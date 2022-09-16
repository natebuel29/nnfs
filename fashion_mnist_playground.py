from zipfile import ZipFile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from layer_dense import Layer_Dense
from accuracy_categorical import Accuracy_Categorical
from activation_relu import Activation_ReLU
from activation_softmax import Activation_Softmax
from activation_sigmoid import Activation_Sigmoid
from loss_categorical_cross_entropy import Loss_CategoricalCrossentropy
from loss_binary_cross_entropy import Loss_BinaryCrossentropy
from loss_mean_squared import Loss_MeanSquaredError
from activation_linear import Activation_Linear
from activation_softmax_loss_cross_entropy import Activation_Softmax_Loss_CategoricalCrossentropy
from sgd_optimizer import Optimizer_SGD
from adagrad_optimizer import Optimizer_Adagrad
from rms_prop_optimizer import Optimizer_RMSprop
from adam_optimizer import Optimizer_Adam
from layer_dropout import Layer_Dropout
from model import Model

# import urllib
# import urllib.request

# URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
# FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

# if not os.path.isfile(FILE):
#     print(f"Downloading {URL} and saving as {FILE}...")
#     urllib.request.urlretrieve(URL, FILE)

#     print('Unzipping images...')
#     with ZipFile(FILE) as zip_images:
#         zip_images.extractall(FOLDER)


def create_data_mnist(path):

    X, y = load_mnist_dataset("train", path)
    X_test, y_test = load_mnist_dataset("test", path)

    return X, y, X_test, y_test


def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(
                path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)

    return np.array(X), np.array(y).astype('uint8')


X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(),
          optimizer=Optimizer_Adam(decay=1e-3), accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y, validation_data=(X_test, y_test),
            epochs=15, batch_size=128, print_every=100)
