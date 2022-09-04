import numpy as np
from activation_softmax import Activation_Softmax
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy


class Activation_Softmax_Loss_categoricalCrossentropy():
    # Softmax classifier - combined softmax activation and cross-entropy loss for faster backward step

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layers activation function
        self.activation.forward(inputs)
        # set the output
        self.outputs = self.activation.output
        # calculate the loss and return values
        return self.loss.calculate(self.outputs, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # number of samples
        samples = len(dvalues)

        # if Labels are one hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()

        # Calculate the gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize the gradient
        self.dinputs = self.dinputs/samples
