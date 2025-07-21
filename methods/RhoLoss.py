from .SelectionMethod import SelectionMethod
import torch
import numpy as np
import torch
import copy

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
    method_name = 'RhoLoss'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0

        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False

    def get_holdout_model(self, config, logger):
        """"Retrieve the holdout model for computing irreducible loss.
        Args:
            config (dict): Configuration dictionary containing model parameters.
            logger (logging.Logger): Logger instance for logging information.
        """

        self.holdout_model = copy.deepcopy(self.model)

        # if holdout model exists for given config
        # return self.holdout_model

        # if holdout model is not specified, create a new one

        # raise NotImplementedError("Holdout model retrieval not implemented.")
        

    def get_ratio_per_epoch(self, epoch):
        """Get the ratio of samples to select for the current epoch based on the configured scheduler.
        Args:
            epoch (int): Current epoch number.
        Returns:
            float: The ratio of samples to select for the current epoch.
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

    def get_reducible_loss(self, inputs, targets):
        """Compute the reducible loss for the current model using the holdout model.
        Args:
            inputs (torch.Tensor): Input data for which to compute the reducible loss.
            targets (torch.Tensor): Corresponding target labels for the input data.
        Returns:
            torch.Tensor: The computed reducible loss.
        """
        total_loss = self.model(inputs, targets)
        irreducible_loss = self.holdout_model(inputs, targets)
        reducible_loss = total_loss - irreducible_loss
        return reducible_loss
        
    def selection(self, inputs, targets):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        reducible_loss = self.get_reducible_loss(inputs, targets)
        _, indices = torch.topk(reducible_loss, int(self.ratio * inputs.shape[0]))
        return indices
    
    def before_batch(self, i, inputs, targets, indexes, epoch):
        """Prepare the batch for training by selecting samples based on reducible loss.
        Args:
            i (int): Current batch index.
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
            indexes (torch.Tensor): Indices of the samples in the current batch.
            epoch (int): Current epoch number.
        Returns:
            tuple: Selected inputs, targets, and indexes for the current batch.
        """
        
        if self.iter_selection:
            # Get the ratio for the current epoch
            ratio = self.get_ratio_per_epoch(epoch)
            if ratio == 1.0:
                if i == 0:
                    self.logger.info('using all samples')
                return super().before_batch(i, inputs, targets, indexes, epoch)
            else:
                if i == 0:
                    self.logger.info(f'balance: {self.balance}')
                    self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))
            
            # Get indices based on reducible loss
            selected_num_samples = int(inputs.shape[0] * ratio)
            indices = self.selection(inputs, targets, selected_num_samples)
            inputs = inputs[indices]
            targets = targets[indices]
            indexes = indexes[indices]
            return inputs, targets, indexes

        else:
            return super().before_batch(i, inputs, targets, indexes, epoch)