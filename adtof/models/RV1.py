import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt


class rv1(object):
    """
    Richard Vogl model
    """

    def __init__(self):
        pass

    def createModel(self):
        """
        TODO
        """
        #TODO
        # When to apply the dropout?
        # How to handle the bidirectionnal aggregation ? Sum, or nothing ?
        # How to handle the ocntext for the learning 400 samples before learning?
        context = 25
        n_bins = 84
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(context, n_bins)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(60)),
            tf.keras.layers.Dense(3)
        ])

        model.compile(optimizer="adam", loss="sigmoid_cross_entropy_with_logits")
        return model
    
    def train(x, y):
        