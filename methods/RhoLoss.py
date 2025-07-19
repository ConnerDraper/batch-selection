from .SelectionMethod import SelectionMethod
import torch
import numpy as np

class RhoLoss(SelectionMethod):
    def __init__(self, model, config, logger):
        """
        Initialize RhoLoss selection method.

        Args:
            model (torch.nn.Module): The blank model to be used for selection (for architecture).
            config (dict): Configuration dictionary containing method options.
            logger (logging.Logger): Logger for logging information.
        """
        super().__init__(config, logger)
        self.ratio = config['method_opt']['ratio']
        self.budget = config['method_opt'].get('budget', 0.1)
        self.epochs = config['method_opt'].get('epochs', 5)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.target_model = model  # your main learner
        self.ILmodel = model # irreducible loss model

    def get_ILmodel(self, inputs, targets, path=''):
        """
        Train or load the irreducible loss model.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target labels.
            indexes (list): Indexes of the data points.
            path (str): Path to load/save the irreducible loss model.
        """
        if path:
            try:
                self.logger.info(f'Loading irreducible loss model from {path}')
                self.ILmodel = torch.load(path)
                self.logger.info('Loaded irreducible loss model')
                return
            except FileNotFoundError:
                self.logger.info(f'Irreducible loss model not found at {path}')
            except Exception as e:
                self.logger.info(f'Failed to load irreducible loss model from {path}, error: {e}')

        self.logger.info('Training irreducible loss model from scratch')
        self.ILmodel = self._build_model()
        optimizer = torch.optim.Adam(self.ILmodel.parameters(), lr=1e-3)

        self.ILmodel.train()
        for epoch in range(self.epochs // 10):
            optimizer.zero_grad()
            outputs = self.ILmodel(inputs)
            il_loss = self.criterion(outputs, targets).mean()
            il_loss.backward()
            optimizer.step()

        self.logger.info('Finished training irreducible loss model')
        if path:
            torch.save(self.ILmodel, path)
            self.logger.info(f'Saved irreducible loss model to {path}')
        else:
            self.logger.info('No path provided, irreducible loss model not saved')

    def select(self, inputs, targets):
        """
        Select samples based on reducible loss.

        Args:
            inputs (torch.Tensor): Input data.
            targets (torch.Tensor): Target labels.
        
        Returns:
            list: Selected indices based on reducible loss.
        """

        if self.ILmodel is None:
            self.get_ILmodel(inputs, targets)

        self.ILmodel.eval()
        self.target_model.eval()

        with torch.no_grad():
            il_outputs = self.ILmodel(inputs)
            irreducible_loss = self.criterion(il_outputs, targets).cpu().numpy()

            target_outputs = self.target_model(inputs)
            total_loss = self.criterion(target_outputs, targets).cpu().numpy()

            reducible_loss = total_loss - irreducible_loss

        n = int(self.budget * len(targets))
        selected_indices = np.argsort(-reducible_loss)[:n]
        return selected_indices.tolist()
    
    def get_ratio_per_epoch(self, epoch):
        """
        Get the selection ratio for the current epoch.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: Selection ratio for the current epoch.
        """
        if epoch < self.warmup_epochs:
            self.logger.info('warming up')
            return 1.0
        if self.ratio_scheduler == 'constant':
            return self.ratio
        elif self.ratio_scheduler == 'increase_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'decrease_linear':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * epoch / self.epochs
        elif self.ratio_scheduler == 'increase_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return min_ratio + (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        elif self.ratio_scheduler == 'decrease_exp':
            min_ratio = self.ratio[0]
            max_ratio = self.ratio[1]
            return max_ratio - (max_ratio - min_ratio) * np.exp(epoch / self.epochs)
        else:
            raise NotImplementedError
    
    def before_batch(self, i, inputs, targets, indexes, epoch):
        """
        Hook to be called before processing each batch.

        Args:
            i (int): Batch index.
            inputs (torch.Tensor): Input data for the batch.
            targets (torch.Tensor): Target labels for the batch.
            indexes (list): Indexes of the data points in the batch.
            epoch (int): Current epoch number.

        Returns:
            tuple: (inputs, targets, indexes) after selection.
        """
        # get ratio for current epoch
        ratio = self.get_ratio_per_epoch(epoch)
        self.logger.info(f'balance: {self.balance}')
        self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))

        # select samples based on RHO loss
        grad_mean, grad = self.calc_grad(inputs, targets, indexes)
        selected_num_samples = int(inputs.shape[0] * ratio)
        indices = self.select(grad_mean, grad, selected_num_samples)
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]

        self.logger.info(f'selected {selected_num_samples}/{inputs.shape[0]} samples from batch {i} in epoch {epoch}')
        return inputs, targets, indexes