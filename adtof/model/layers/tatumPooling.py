import tensorflow as tf
from tensorflow import keras
from keras import backend as K


class TatumPooling(keras.layers.Layer):
    def __init__(self) -> None:
        """
        Performs pooling on 2D spatial data to convert from a frame-level to a tatum-level.
        The pooling is performed with variable window size and no overlap corresponding to the tatum locations.
        See: Ishizuka - 2021 - Global Structure-Aware Drum Transcription Based on Self-Attention Mechanisms

        poolingOperation specifies which pooling operation to use
        """
        super().__init__()
        # self.poolingOperation = poolingOperation

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({"poolingOperation": self.poolingOperation})
    #     return config

    @tf.function
    def _poolBatch(self, featureMap, tatumBoundaries):
        """
        Applies the pooling operation to a single track and its tatum indexes
        """
        # Input is a spec_tensor (all the tatum have 2 boundaries)
        # output is a spec_tensor (all the max pooling op return features dim)
        return tf.map_fn(lambda e: K.max(featureMap[e[0] : e[1], :], axis=0), tatumBoundaries, fn_output_signature=tf.float32)

    @tf.function
    def call(self, featureMaps, tatumsBoundaries):
        """
        input:
        featureMaps shape is [batch, frame, features]
        tatumsBoundaries shape is [batch, tatum, 2 (start and stop boundaries)]
        Because the number of samples covering the tatums depends on the tempo of the track, featureMaps has to be padded with 0 to have the same length.

        output:
        pooledFeatures of shape [batch, tatum, features]
        pooledFeatures has the same amount of tatums for each track.
        (this is needed for the following layers)
        """
        # call _poolBatch(featureMaps[i], tatumsBoundaries[i]) for each batch i.
        # the batch i is retreived by putting the tensors into the same tuple and specifying their dtype
        # The output type has to be specified since it is different from the input type
        return tf.map_fn(lambda tuple: self._poolBatch(tuple[0], tuple[1]), (featureMaps, tatumsBoundaries), fn_output_signature=tf.float32, parallel_iterations=10)
        # return tf.vectorized_map(lambda tuple: self._poolBatch(tuple[0], tuple[1]), (featureMaps, tatumsBoundaries)) # Doesn't work whith different size output elements
