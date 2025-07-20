from .SelectionMethod import SelectionMethod
import torch
import numpy as np

class RhoLoss(SelectionMethod):
    """A class for implementing the RhoLoss selection method, which selects samples based on reducible loss.

    This class inherits from `SelectionMethod` and uses an irreducible loss model (ILmodel) and a target model
    to compute reducible loss for sample selection during training. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'ratio', 'budget', 'epochs', 'ratio_scheduler',
                  'warmup_epochs', 'iter_selection', 'balance'.
                - 'rho_loss': Dictionary with key 'training_budget'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
                - 'networks': Dictionary with key 'params' containing 'm_type'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.ratio = config['method_opt']['ratio']
        self.budget = config['method_opt'].get('budget', 0.1)
        self.epochs = config['method_opt'].get('epochs', 5)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.ratio_scheduler = config['method_opt'].get('ratio_scheduler', 'constant')
        self.warmup_epochs = config['method_opt'].get('warmup_epochs', 0)
        self.iter_selection = config['method_opt'].get('iter_selection', False)
        self.balance = config['method_opt'].get('balance', 'none')

        # rho loss specific parameters
        self.training_budget = config['rho_loss'].get('training_budget', 0.1)
        self.il_epochs = max(1, int(self.training_budget * self.epochs))

        # device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build blank target and IL models
        self.target_model = self._build_model().to(self.device)
        self.ILmodel = self._build_model().to(self.device)

    def _build_model(self):
        """Builds a neural network model based on dataset and model type from config.

        Returns:
            torch.nn.Module: A neural network model for the specified dataset and model type.

        Raises:
            KeyError: If the dataset name or model type is not supported.
        """
        dataset_name = self.config['dataset']['name']
        m_type = self.config['networks']['params']['m_type']
        num_classes = self.config['dataset']['num_classes']

        if dataset_name in ['cifar10', 'cifar100']:
            from networks.CIFAR import get_model
            model = get_model(m_type, num_classes=num_classes)
        elif dataset_name in ['imagenet100', 'imagenet']:
            from networks.ImageNet import get_model
            model = get_model(m_type, num_classes=num_classes)
        else:
            raise KeyError(f"Unsupported dataset: {dataset_name}")

        return model

    def get_ILmodel(self, inputs, targets, path=''):
        """Trains or loads the irreducible loss model (ILmodel).

        If a path is provided, attempts to load a pre-trained ILmodel from the specified path.
        Otherwise, trains the ILmodel from scratch using the provided inputs and targets.

        Args:
            inputs (torch.Tensor): Input data for training the ILmodel.
            targets (torch.Tensor): Target labels for training the ILmodel.
            path (str, optional): Path to load/save the ILmodel state dictionary. Defaults to ''.

        Raises:
            FileNotFoundError: If the specified model path does not exist.
            Exception: For other errors during model loading.
        """
        if path:
            try:
                self.logger.info(f'Loading irreducible loss model from {path}')
                self.ILmodel.load_state_dict(torch.load(path))
                self.ILmodel.to(self.device)
                self.logger.info('Loaded irreducible loss model')
                return
            except FileNotFoundError:
                self.logger.info(f'Irreducible loss model not found at {path}')
            except Exception as e:
                self.logger.info(f'Failed to load irreducible loss model from {path}, error: {e}')

        self.logger.info(f'Training irreducible loss model from scratch for {self.il_epochs} epochs')
        optimizer = torch.optim.Adam(self.ILmodel.parameters(), lr=1e-3)
        self.ILmodel.train()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        for epoch in range(self.il_epochs):
            optimizer.zero_grad()
            outputs = self.ILmodel(inputs)
            il_loss = self.criterion(outputs, targets).mean()
            il_loss.backward()
            optimizer.step()

        self.logger.info('Finished training irreducible loss model')
        if path:
            torch.save(self.ILmodel.state_dict(), path)
            self.logger.info(f'Saved irreducible loss model to {path}')
        else:
            self.logger.info('No path provided, irreducible loss model not saved')

    def select(self, inputs, targets, epoch):
        """Selects a subset of samples based on reducible loss.

        Computes the reducible loss as the difference between total loss (from target model)
        and irreducible loss (from ILmodel), then selects the top samples based on the ratio
        for the current epoch.

        Args:
            inputs (torch.Tensor): Input data for computing losses.
            targets (torch.Tensor): Target labels for computing losses.
            epoch (int): Current training epoch.

        Returns:
            list: Indices of selected samples, sorted by descending reducible loss.
        """
        self.ILmodel.eval()
        self.target_model.eval()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            il_outputs = self.ILmodel(inputs)
            irreducible_loss = self.criterion(il_outputs, targets).cpu().numpy()
            target_outputs = self.target_model(inputs)
            total_loss = self.criterion(target_outputs, targets).cpu().numpy()
            reducible_loss = total_loss - irreducible_loss

        ratio = self.get_ratio_per_epoch(epoch)
        n = max(1, int(ratio * len(targets)))
        selected_indices = np.argsort(-reducible_loss)[:n]
        return selected_indices.tolist()

    def get_ratio_per_epoch(self, epoch):
        """Determines the sample selection ratio for the current epoch.

        Supports multiple ratio scheduling strategies: constant, linear increase/decrease,
        and exponential increase/decrease.

        Args:
            epoch (int): Current training epoch.

        Returns:
            float: The selection ratio for the current epoch.

        Raises:
            NotImplementedError: If an unsupported ratio scheduler is specified.
        """
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio, max_ratio = self.ratio
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio, max_ratio = self.ratio
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio, max_ratio = self.ratio
            progress = epoch / self.epochs
            scale = (np.exp(progress) - 1) / (np.e - 1)
            return min_ratio + (max_ratio - min_ratio) * scale
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio, max_ratio = self.ratio
            progress = epoch / self.epochs
            scale = (np.exp(progress) - 1) / (np.e - 1)
            return max_ratio - (max_ratio - min_ratio) * scale
        else:
            raise NotImplementedError(f"Unsupported ratio scheduler: {self.ratio_scheduler}")

    def before_batch(self, i, inputs, targets, indexes, epoch):
        """Selects a subset of samples from a batch for training.

        Adjusts the input batch based on the reducible loss selection criteria and the
        current epoch's ratio.

        Args:
            i (int): Batch index.
            inputs (torch.Tensor): Input data for the batch.
            targets (torch.Tensor): Target labels for the batch.
            indexes (np.ndarray or torch.Tensor or list): Indices of samples in the batch.
            epoch (int): Current training epoch.

        Returns:
            tuple: (inputs, targets, indexes) of the selected samples.
        """
        ratio = self.get_ratio_per_epoch(epoch)
        self.logger.info(f'selecting samples for epoch {epoch}, ratio {ratio}')

        if type(indexes) not in [np.ndarray, torch.Tensor]:
            indexes = np.array(indexes)

        if ratio >= 1.0:
            self.logger.info(f'ratio {ratio} >= 1.0, using all samples')
            return inputs, targets, indexes

        sample_size = inputs.shape[0]
        selected_indices = self.select(inputs, targets, epoch)
        subsample_size = len(selected_indices)

        inputs = inputs[selected_indices]
        targets = targets[selected_indices]
        indexes = indexes[selected_indices]

        self.logger.info(f'selected {subsample_size}/{sample_size} samples from batch {i} in epoch {epoch}')
        return inputs, targets, indexes