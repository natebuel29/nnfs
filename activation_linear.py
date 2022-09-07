import numpy as np


class Activation_Linear:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        #derivative is 1
        self.dinputs = dvalues.copy()
