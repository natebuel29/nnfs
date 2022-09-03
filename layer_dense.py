import numpy as np


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing weights and biases
        self. weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
