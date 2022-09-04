import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy
from activation_softmax_loss_cross_entropy import Activation_Softmax_Loss_categoricalCrossentropy

nnfs.init()
# Create dataset
X, y = spiral.create_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values (neurons)
dense1 = Layer_Dense(2, 3)

# Create Relu Activation
activation1 = Activation_ReLU()

# Create a second Dense layer with 3 input featues (as we take output of previous layer here) and 3 output layers
dense2 = Layer_Dense(3, 3)

# Create softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_categoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# make a forward pass through the second dense layer
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)
print("Loss:", loss)

# Calculate the accuracy from output of activation2 and targets
# calculate the values across the first axis (rows)
predictions = np.argmax(loss_activation.outputs, axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy: ", accuracy)

# backward pass
loss_activation.backward(loss_activation.outputs, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
