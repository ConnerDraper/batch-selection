from .SelectionMethod import SelectionMethod
import torch
import numpy as np
# import copy

class DivBS(SelectionMethod):
    """A class for implementing the DivBS selection method, which selects samples based on diversity and balance.

    This class inherits from `SelectionMethod` and uses a greedy selection strategy to choose samples
    that maximize diversity while maintaining balance across classes. It supports various ratio scheduling strategies
    for dynamic sample selection and handles model training and loading for specific datasets.

    Args:
        config (dict): Configuration dictionary containing method and dataset parameters.
            Expected keys include:
                - 'method_opt': Dictionary with keys 'balance', 'iter_selection', 'epoch_selection',
                  'num_epochs_per_selection', 'ratio', 'ratio_scheduler', 'warmup_epochs',
                  'reduce_dim'.
                - 'dataset': Dictionary with keys 'name' and 'num_classes'.
        logger (logging.Logger): Logger instance for logging training and selection information.
    """
    method_name = 'DivBS'
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.balance = config['method_opt']['balance']
        self.ratio = config['method_opt']['ratio']
        self.ratio_scheduler = config['method_opt']['ratio_scheduler'] if 'ratio_scheduler' in config['method_opt'] else 'constant'
        self.warmup_epochs = config['method_opt']['warmup_epochs'] if 'warmup_epochs' in config['method_opt'] else 0

        self.current_train_indices = np.arange(self.num_train_samples)
        self.reduce_dim = config['method_opt']['reduce_dim'] if 'reduce_dim' in config['method_opt'] else False
        
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

    def calc_grad(self, inputs, targets, indexes):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.eval()
        outputs, features = model.feat_nograd_forward(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        with torch.no_grad():
            grad_out = torch.autograd.grad(loss, outputs, retain_graph=True)[0] 
            grad = grad_out.unsqueeze(-1) * features.unsqueeze(1)
            grad = grad.view(grad.shape[0], -1)
        model.train()
        if self.reduce_dim:
            dim = grad.shape[1]
            dim_reduced = dim // self.reduce_dim
            index = np.random.choice(dim, dim_reduced, replace=False)
            grad = grad[:, index]
        grad_mean = grad.mean(dim=0)
        return grad_mean, grad
    
    def greedy_selection(self, grad_mean, grad, number_to_select):
        # print(grad.shape)
        residual = grad_mean.unsqueeze(-1) if grad_mean.dim() == 1 else grad_mean
        index_selected = []
        D = grad.t()
        selected_element = []
        for i in range(number_to_select):
            correlations = torch.abs(torch.matmul(D.t(), residual))
            try:
                idx = torch.multinomial(correlations.squeeze(), 1)
            except:
                break
            # print(f'idx: {idx.shape}')
            index_selected.append(idx.item())
            if len(selected_element) >0:
                selected_element_matrix = torch.cat(selected_element, dim=1) # n_features, n_selected
                D_selected = D[:, idx] - torch.matmul(selected_element_matrix,torch.matmul(selected_element_matrix.t(), D[:, idx]))
            else:
                D_selected = D[:, idx]
            D_selected = D_selected/ torch.norm(D_selected)
            selected_element.append(D_selected)

            
            residual = residual - torch.matmul(D_selected.t(), residual)* D_selected
        
        if len(selected_element) < number_to_select:
            num_random = number_to_select - len(selected_element)
            self.logger.info(f'number_to_select: {number_to_select}, len(selected_element): {len(selected_element)}, num_random: {num_random}')
            remaining_indices = list(set(range(grad.shape[0])) - set(index_selected))
            random_indices = np.random.choice(remaining_indices, num_random, replace=False)
            index_selected.extend(random_indices)

        return index_selected

    def before_batch(self, i, inputs, targets, indexes, epoch):
        ratio = self.get_ratio_per_epoch(epoch)
        if ratio == 1.0:
            if i == 0:
                self.logger.info('using all samples')
            return super().before_batch(i, inputs, targets, indexes, epoch)
        else:
            if i == 0:
                self.logger.info(f'balance: {self.balance}')
                self.logger.info('selecting samples for epoch {}, ratio {}'.format(epoch, ratio))
        grad_mean, grad = self.calc_grad(inputs, targets, indexes)
        selected_num_samples = int(inputs.shape[0] * ratio)
        indices = self.greedy_selection(grad_mean, grad, selected_num_samples)
        inputs = inputs[indices]
        targets = targets[indices]
        indexes = indexes[indices]
        return inputs, targets, indexes