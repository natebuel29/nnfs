import numpy as np
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy
from activation_softmax_loss_cross_entropy import Activation_Softmax_Loss_CategoricalCrossentropy
from sgd_optimizer import Optimizer_SGD
from adagrad_optimizer import Optimizer_Adagrad
from rms_prop_optimizer import Optimizer_RMSprop
from adam_optimizer import Optimizer_Adam
from layer_dropout import Layer_Dropout

nnfs.init()
# Create dataset
X, y = spiral.create_data(samples=1000, classes=3)

# Create Dense layer with 2 inputs and 64 output values
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
                     bias_regularizer_l2=5e-4)

# Create ReLU activation function
activation1 = Activation_ReLU()

# Create dropout layer
dropout1 = Layer_Dropout(0.1)

# Create second Dense layer with 64 input features and 3 output values
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation function
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create Optimizer
optimizer = Optimizer_Adam(learning_rate=.09, decay=5e-5)

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(
        dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"data_loss: {data_loss:.3f}, " +
              f"reg_loss: {regularization_loss:.3f}, " +
              f"lr: {optimizer.current_learning_rate}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# Validate the model

# Create test dataset
X_test, y_test = spiral.create_data(samples=100, classes=3)

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == y_test)

print(f"validation acc: {accuracy:.3f}, loss: {loss:.3f}")
