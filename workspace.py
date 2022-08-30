import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax

nnfs.init()
#Create dataset
X,y = spiral.create_data(samples=100,classes=3)

#Create Dense layer with 2 input features and 3 output values (neurons)
dense1 = Layer_Dense(2,3)

#Create Relu Activation
activation1 = Activation_ReLU()

#Create a second Dense layer with 3 input featues (as we take output of previous layer here) and 3 output layers
dense2 = Layer_Dense(3,3)

#Create a softmax activation 
activation2 = Activation_Softmax()

#Perform a forward pass of our training data through this layer
dense1.forward(X)

#Make a forward pass through activation function
#it takes the output of first dense layer here
activation1.forward(dense1.output)

#make a forward pass through the second dense layer
dense2.forward(activation1.output)

#make a forward pass through activation function
activation2.forward(dense2.output)

print(activation2.output[:5])
