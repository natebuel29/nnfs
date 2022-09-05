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

nnfs.init()
# Create dataset
X, y = spiral.create_data(samples=100, classes=3)

# Create Dense layer with 2 inputs and 64 output values
dense1 = Layer_Dense(2, 64)

# Create ReLU activation function
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features and 3 output values
dense2 = Layer_Dense(64, 3)

# Create Softmax classifier's combined loss and activation function
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create Optimizer
optimizer = Optimizer_Adam(learning_rate=.05, decay=5e-7)

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}" +
              f"lr: {optimizer.current_learning_rate}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
