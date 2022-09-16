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
# FOLDER = 'fashion_mnist_images'

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


fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Read an image
image_data = cv2.imread('pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

# Load the model
model = Model.load('fashion_mnist.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
