import numpy as np


class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss

    # Regularization loss calculation
    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.weight_regularizer_L1 > 0:
            regularization_loss += layer.weight_regularizer_L1 * \
                np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_L2 > 0:
            regularization_loss += layer.weight_regularizer_L2 * \
                np.sum(np.abs(layer.weights))

        if layer.bias_regularizer_L1 > 0:
            regularization_loss += layer.bias_regularizer_L1 * \
                np.sum(np.abs(layer.weights))

        if layer.bias_regularizer_L2 > 0:
            regularization_loss += layer.bias_regularizer_L2 * \
                np.sum(np.abs(layer.weights))

        return regularization_loss
