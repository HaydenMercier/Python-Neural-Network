import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
import tensorflow_datasets as datasets
mnist = datasets.load(name='mnist')

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()

class_names = ["T-Shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=1000)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("10 000 image Test accuracy:", test_acc)