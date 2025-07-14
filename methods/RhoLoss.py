from .SelectionMethod import SelectionMethod
import torch
import numpy as np

class RhoLoss(SelectionMethod):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.ratio = config['method_opt']['ratio']
        self.budget = config['method_opt'].get('budget', 0.1)
        self.epochs = config['method_opt'].get('epochs', 5)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.target_model = self._build_model()  # your main learner
        self.ILmodel = None  # irreducible loss model placeholder

    def _build_model(self, input_shape=(1, 28, 28), num_classes=10):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )

    def get_ILmodel(self, inputs, targets, indexes, path=''):
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
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.ILmodel(inputs)
            il_loss = self.criterion(outputs, targets).mean()
            il_loss.backward()
            optimizer.step()

        self.logger.info('Finished training irreducible loss model')
        if path:
            torch.save(self.ILmodel, path)
            self.logger.info(f'Saved irreducible loss model to {path}')

    def select(self, inputs, targets, indexes):
        if self.ILmodel is None:
            self.get_ILmodel(inputs, targets, indexes)

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
        return [indexes[i] for i in selected_indices]
