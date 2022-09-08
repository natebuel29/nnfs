import numpy as np


class Activation_Linear:

    def predictions(self, outputs):
        return outputs

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        #derivative is 1
        self.dinputs = dvalues.copy()
