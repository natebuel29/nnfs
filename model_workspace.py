import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import sine_data, spiral_data
from layer_dense import Layer_Dense
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from activation_sigmoid import Activation_Sigmoid
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy
from loss_binary_cross_entropy import Loss_BinaryCrossentropy
from loss_mean_squared import Loss_MeanSquaredError
from activation_linear import Activation_Linear
from accuracy_regression import Accuracy_Regression
from accuracy_categorical import Accuracy_Categorical

from activation_softmax_loss_cross_entropy import Activation_Softmax_Loss_CategoricalCrossentropy
from sgd_optimizer import Optimizer_SGD
from adagrad_optimizer import Optimizer_Adagrad
from rms_prop_optimizer import Optimizer_RMSprop
from adam_optimizer import Optimizer_Adam
from layer_dropout import Layer_Dropout
from model import Model
nnfs.init()

X_lin, y_lin = sine_data()
model1 = Model()
model1.set(loss=Loss_MeanSquaredError(), optimizer=Optimizer_Adam(
    learning_rate=0.005, decay=1e-3), accuracy=Accuracy_Regression())
# Add layers
model1.add(Layer_Dense(1, 64))
model1.add(Activation_ReLU())
model1.add(Layer_Dense(64, 64))
model1.add(Activation_ReLU())
model1.add(Layer_Dense(64, 1))
model1.add(Activation_Linear())

model1.finalize()

model1.train(X_lin, y_lin, epochs=500, print_every=100)


X_bin, y_bin = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y_bin = y_bin.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model2 = Model()

model2.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4,
           bias_regularizer_l2=5e-4))

model2.add(Activation_ReLU())
model2.add(Layer_Dense(64, 1))
model2.add(Activation_Sigmoid())

model2.set(loss=Loss_BinaryCrossentropy(), optimizer=Optimizer_Adam(
    decay=5e-7), accuracy=Accuracy_Categorical(binary=True))

model2.finalize()

model2.train(X_bin, y_bin, validation_data=(
    X_test, y_test), epochs=10000, print_every=100)


# Create dataset
X_cat, y_cat = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model3 = Model()

# Add layers
model3.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                       bias_regularizer_l2=5e-4))
model3.add(Activation_ReLU())
model3.add(Layer_Dropout(0.1))
model3.add(Layer_Dense(512, 3))
model3.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model3.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model3.finalize()

# Train the model
model3.train(X_cat, y_cat, validation_data=(X_test, y_test),
             epochs=10000, print_every=100)
