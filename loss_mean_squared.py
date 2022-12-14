import numpy as np
from loss import Loss


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):

        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        # Number of outputs for each sample
        outputs = len(dvalues[0])

        self.dinputs = -2*(y_true - dvalues)/outputs

        self.dinputs = self.dinputs/samples
