import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy

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

loss_function = Loss_CategoricalCrossentropy()
print(y)
loss = loss_function.calculate(activation2.output,y)
print("Loss:",loss)

##Calculate the accuracy from output of activation2 and targets
# calculate the values across the first axis (rows)
predictions = np.argmax(activation2.output,axis=1)

if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print("Accuracy: ",accuracy)