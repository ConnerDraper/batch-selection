from .SelectionMethod import SelectionMethod
import models
import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
from os import path

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

        # holdout model parameters
        self.holdout_ratio = config['holdout']['holdout_ratio'] if 'holdout' in config else 0.1
        self.holdout_epochs = config['holdout']['holdout_epochs'] if 'holdout' in config else 10
        self.holdout_batch_size = config['holdout']['holdout_batch_size'] if 'holdout' in config else 128
        self.holdout_num_workers = config['holdout']['holdout_num_workers'] if 'holdout' in config else 4
        self.holdout_model_path = config['holdout']['holdout_model_path'] if 'holdout' in config else None
        
        # Load or train holdout model
        self.split_train_holdout()
        if self.holdout_model_path and path.exists(self.holdout_model_path):
            self.logger.info(f'Loading holdout model from {self.holdout_model_path}')
            self.load_holdout_model()
        else:
            self.logger.info('No valid holdout model path provided or file does not exist. Training a new holdout model.')
            self.train_holdout_model()


    def load_holdout_model(self):
        """Load the holdout model from the specified path.
        Args:
            model_path (str): Path to the holdout model file.
        Returns:
            torch.nn.Module: The loaded holdout model.
        """
        self.logger.info(f'Loading holdout model from {self.holdout_model_path}')
        holdout_model = torch.load(self.holdout_model_path)
        holdout_model.eval()
        self.holdout_model = holdout_model

    def split_train_holdout(self):
        """Split the training dataset into training and holdout subsets using ratio."""
        total_len = len(self.train_dset)
        holdout_ratio = self.holdout_ratio
        train_ratio = 1.0 - holdout_ratio

        self.train_dset, self.holdout_dset = random_split(
            self.train_dset,
            [train_ratio, holdout_ratio],
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.holdout_loader = DataLoader(self.holdout_dset, batch_size=self.batch_size, shuffle=False)
        self.train_loader = DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=False)
        self.logger.info(f'Split training dataset into {len(self.train_dset)} training samples and {len(self.holdout_dset)} holdout samples')


    def train_holdout_model(self):
        self.logger.info('Training new holdout model')
        model_type = self.config['networks']['type']
        model_args = self.config['networks']['params']
        ho_model = getattr(models, model_type)(**model_args).to(self.device).train()

        opt = self.config['holdout']['optim_params']
        crit_params = self.config['holdout']['loss_params']
        optimizer = torch.optim.SGD(ho_model.parameters(), lr=opt.get('lr', 0.01),
                                    momentum=opt.get('momentum', 0.9),
                                    weight_decay=opt.get('weight_decay', 5e-4))
        criterion = torch.nn.CrossEntropyLoss(**crit_params)

        for epoch in range(self.holdout_epochs):
            total_loss = correct = total = 0
            for batch in self.holdout_loader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)

                optimizer.zero_grad()
                outputs = ho_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(dim=1) == targets).sum().item()
                total += inputs.size(0)

            avg = total_loss / total
            acc = 100 * correct / total
            self.logger.info(f"[Holdout Epoch {epoch+1}/{self.holdout_epochs}] "
                             f"Loss: {avg:.4f}, Acc: {acc:.2f}%")

        self.holdout_model = ho_model.eval()
        if self.holdout_model_path:
            torch.save(self.holdout_model, self.holdout_model_path)
            self.logger.info(f"Saved holdout model to {self.holdout_model_path}")


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
        with torch.no_grad():
            logits_main = self.model(inputs)
            total_loss = self.criterion(logits_main, targets)

            logits_holdout = self.holdout_model(inputs)
            irreducible_loss = self.criterion(logits_holdout, targets)

        reducible_loss = total_loss - irreducible_loss
        self.logger.debug(
            f"Reducible loss stats: mean={reducible_loss.mean():.4f}, max={reducible_loss.max():.4f}, min={reducible_loss.min():.4f}"
        )
        return reducible_loss

    def selection(self, inputs, targets, selected_num_samples):
        """Select sub-batch with highest reducible loss.
        Args:
            inputs (torch.Tensor): Input data for the current batch.
            targets (torch.Tensor): Corresponding target labels for the current batch.
        Returns:
            torch.Tensor: Indices of the selected samples.
        """
        reducible_loss = self.get_reducible_loss(inputs, targets)
        _, indices = torch.topk(reducible_loss, selected_num_samples)
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
                if i % 50 == 0:
                    self.logger.info(f'balance: {self.balance}')
                    self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))
            
            # Get indices based on reducible loss
            selected_num_samples = round(inputs.shape[0] * ratio)
            selected_num_samples = max(1, min(selected_num_samples, inputs.shape[0]))
            indices = self.selection(inputs, targets, selected_num_samples)
            self.logger.debug(f"Selected {len(indices)} samples out of {inputs.shape[0]} for batch {i}")

            inputs = inputs[indices]
            targets = targets[indices]
            indexes = indexes[indices]
            return inputs, targets, indexes

        else:
            return super().before_batch(i, inputs, targets, indexes, epoch)