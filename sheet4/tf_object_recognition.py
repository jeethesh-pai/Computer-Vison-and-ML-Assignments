#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np

from utils import *

#
# Task 3
#
# Create a network to recognize single handwritten digits (0-9)

# train data      , test data
# (images, digits), (images, digits)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
input_shape = x_train[0].shape
num_classees = len(set(y_test))  # digits 0, 1, .. 9


# TODO: Normalize all input data to [0, 1].
#
# ???
#


# Now it's time to create the actual TensorFlow model.
# To get to know the syntax, you shall create the same network using the two available APIs.

# TODO: Use the sequential model API to create your model
#  (https://www.tensorflow.org/guide/keras/sequential_model)
def build_model_sequential(input_shape, num_output_classes):
    # Create an empty sequential model
    model = REPLACE_THIS_MODEL
    #
    # ???
    #
    # Add an input `flatten` layer with the provided `input_shape` to the model
    #
    # ???
    #
    # Add a hidden `dense` layer of size 128 with "relu" activation to the model
    #
    # ???
    #
    # Add an output `dense` layer of size `num_output_classes` with "softmax" activation to the model
    #
    # ???
    #
    return model


# TODO: Use the functional model API to create the *same* model as above
#  (https://www.tensorflow.org/guide/keras/functional)
def build_model_functional(input_shape, num_output_classes):
    # Start by creating an `input` layer with the provided `input_shape`
    #
    # ???
    #
    # Then create the same layers as in `build_model_sequential`
    #
    # ???
    #
    # Finally, build and return the actual model using the input layer and the last (output) layer
    return REPLACE_THIS_MODEL  # tf.keras.Model(...


model_seq = build_model_sequential(input_shape, num_classees)
model_fun = build_model_functional(input_shape, num_classees)

img_groups = []

for model in [model_seq, model_fun]:

    if hasattr(model, 'summary'):
        model.summary()  # Tipp: If this function fails, the above created model is not complete (e.g. input_shape information might be missing)

    # TODO: Compile the model using
    #  "sgd" as `optimizer`
    #  "sparse_categorical_crossentropy" as `loss` function
    #  "accuracy" as additional `metrics`
    #
    # ???
    #

    # TODO: Train the model using the `x/y_train` data for 5 epocs.
    #  Attach a callback to enable evaluation of the training progress using `TensorBoard`.
    #
    # ???
    #

    # Use the trained model to recognize the digit of some random images
    num_samples = 9
    sample_idx = np.random.randint(0, len(x_test), num_samples)
    img_groups = img_groups + [
        [("GT: " + str(y_test[i]) + " / Detected: " + str(np.argmax(model(x_test[[i], :, :]))), x_test[i]) for i in
         sample_idx]]

for imgs in img_groups:
    plt.figure(figsize=(6, 6))
    showImages(imgs, 3, False)
plt.show()
