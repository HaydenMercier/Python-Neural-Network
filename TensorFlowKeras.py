import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
import numpy as np
X = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)
model = tf.keras.Sequential()
model.add(Dense(4, imput_dim=3, activaton="relu"))