import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from activation_sigmoid import Activation_Sigmoid
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy
from loss_binary_cross_entropy import Loss_BinaryCrossentropy
from activation_softmax_loss_cross_entropy import Activation_Softmax_Loss_CategoricalCrossentropy
from sgd_optimizer import Optimizer_SGD
from adagrad_optimizer import Optimizer_Adagrad
from rms_prop_optimizer import Optimizer_RMSprop
from adam_optimizer import Optimizer_Adam
from layer_dropout import Layer_Dropout

nnfs.init()
# Create dataset
X, y = spiral.create_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Innerlist contains one output (either 0 or 1)
# per each neuron, 1 in this  case
y = y.reshape(-1, 1)

# Create Dense layer with 2 inputs and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)

# Create ReLU activation function
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features and 3 output values
dense2 = Layer_Dense(64, 1)

# Create Sigmoid activation function
activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

# Create Optimizer
optimizer = Optimizer_Adam(decay=5e-7)

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = (activation2.output > 0.5)*1

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"data_loss: {data_loss:.3f}, " +
              f"reg_loss: {regularization_loss:.3f}, " +
              f"lr: {optimizer.current_learning_rate}")

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model

# Create test dataset
X_test, y_test = spiral.create_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Innerlist contains one output (either 0 or 1)
# per each neuron, 1 in this  case
y_test = y_test.reshape(-1, 1)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5)*1

accuracy = np.mean(predictions == y_test)

print(f"validation acc: {accuracy:.3f}, loss: {loss:.3f}")
