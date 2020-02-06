#!/usr/bin/env python
# encoding: utf-8
"""
TODO
"""
import argparse
import datetime
import io
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import itertools

from adtof.deepModels import dataLoader
from adtof.deepModels.peakPicking import PeakPicking
from adtof.deepModels.rv1tf import RV1TF
from adtof.io import mir
from adtof.io.converters.converter import Converter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.config.experimental_run_functions_eagerly(True)
logging.basicConfig(filename='logs/conversion.log', level=logging.DEBUG)


def log_layers(epoch, input, model, activation_model,file_writer):
    log_layer_activation(epoch, input, model, activation_model, file_writer)
    log_layer_weights(epoch, input, model, activation_model, file_writer)

def log_layer_activation(epoch, input, model, activation_model, file_writer):
    # Create a figure to contain the plot.
    layer_names = ["activation 0 input"] + ["activation " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers)]
    layer_activations = [input] + activation_model.predict(input)
    for iLayer, (layer_name, layer_activation) in enumerate(zip(layer_names, layer_activations)):  # Displays the feature maps
        # Start next subplot.
        figure = plt.figure(figsize=(20, 10))

        if len(layer_activation.shape) == 4:
            _, sizeW, sizeH, nKernels = layer_activation.shape
            for iKernel in range(nKernels):
                plt.subplot(nKernels // 5 + 1, 5, iKernel + 1, title=str(iKernel))
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(layer_activation[0, :, :, iKernel], cmap='viridis')

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image(layer_name, image, step=epoch)


def log_layer_weights(epoch, input, model, activation_model, file_writer):
    # Create a figure to contain the plot.
    layer_names = ["weights " + str(i + 1) + " " + layer.name for i, layer in enumerate(model.layers) if len(layer.get_weights())]
    layer_weigths = [layer.get_weights()[0] for layer in model.layers if len(layer.get_weights())]
    for iLayer, (layer_name, layer_weigth) in enumerate(zip(layer_names, layer_weigths)):  # Displays the feature maps
        # Start next subplot.
        figure = plt.figure(figsize=(20, 10))

        if len(layer_weigth.shape) == 4:
            sizeW, sizeH, nChannel, nKernels = layer_weigth.shape
            for iChannel, iKernel in itertools.product(range(1), range(nKernels)):
                plt.subplot(1, nKernels, iChannel * nKernels + iKernel + 1, title=str(iKernel) + "-" + str(iChannel))
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(layer_weigth[:, :, iChannel, iKernel], cmap='viridis')

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)

        with file_writer.as_default():
            tf.summary.image(layer_name, image, step=epoch)


def main():
    """
    Entry point of the program
    Parse the arguments and call the conversion
    """
    parser = argparse.ArgumentParser(description='todo')
    parser.add_argument('folderPath', type=str, help="Path.")
    args = parser.parse_args()
    labels = [36]  #[36, 40, 41, 46, 49]
    sampleRate = 50

    # dataLoader.vizDataset(args.folderPath, labels=labels, sampleRate=sampleRate)
    # Plot the first image of the dataset
    # for x, y in dataset:
    #     file_writer = tf.summary.create_file_writer(log_dir)
    #     with file_writer.as_default():
    #         tf.summary.image(str(list(np.reshape(y, (batch_size)))), x, step=0, max_outputs=20, description=str(list(np.reshape(y, (batch_size)))))

    # Get the data
    # classWeight = dataLoader.getClassWeight(args.folderPath)
    dataset = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=True, labels=labels, sampleRate=sampleRate), (tf.float64, tf.float64),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(labels), )))
    )
    dataset_test = tf.data.Dataset.from_generator(
        dataLoader.getTFGenerator(args.folderPath, train=False, labels=labels, sampleRate=sampleRate), (tf.float64, tf.float64),
        output_shapes=(tf.TensorShape((None, None, 1)), tf.TensorShape((len(labels), )))
    )
    batch_size = 100
    dataset = dataset.batch(batch_size).repeat()
    dataset_test = dataset_test.batch(batch_size).repeat()
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset_test = dataset_test.prefetch(buffer_size=batch_size)

    # Get the model
    model = RV1TF().createModel(output=len(labels))
    checkpoint_path = "models/rv1.ckpt"
    log_dir = os.path.join("logs", "fit", "rv1.2")  #datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    file_writer = tf.summary.create_file_writer(log_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)

    # Get the debug activation model
    layer_outputs = [layer.output for layer in model.layers]  # Extracts the outputs of the top 12 layers
    activation_model = tf.keras.models.Model(
        inputs=model.input, outputs=layer_outputs
    )  # Creates a model that will return these outputs, given the model input
    viz_example, _ = next(dataLoader.getTFGenerator(args.folderPath, train=False, labels=labels, sampleRate=sampleRate)())
    viz_example = viz_example.reshape([1] + list(viz_example.shape))  #Adding mini-batch

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True),
        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
        ),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log_layer_activation(epoch, viz_example, model, activation_model, file_writer))
    ]

    model.fit(
        dataset,
        epochs=100,
        initial_epoch=0,
        steps_per_epoch=100,
        callbacks=callbacks,
        validation_data=dataset_test,
        validation_steps=20
        # class_weight=classWeight
    )

    # for x, y in dataset_test:
    #     predictions = model.predict(x)

    #     import matplotlib.pyplot as plt
    #     f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #     ax1.plot(predictions)
    #     ax2.plot(y)
    #     plt.show()
    #     print("Done!")


if __name__ == '__main__':
    main()
