import numpy as np
import tensorflow as tf


def get_angles_std(pos, i, d_model):
    """
    Get the std positional encoding --- nonlinear stripes patterns in the encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_angles_linear(pos, i):
    """
    Get the tatum-synchronous positional encoding --- linear stripes patterns in the encoding
    """
    angle_rates = np.pi / (2 + np.floor(i / 2))
    return pos * angle_rates


def positional_encoding(position, d_model, encoding="std"):
    if encoding == "std":
        angle_rads = get_angles_std(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    elif encoding == "linear":
        angle_rads = get_angles_linear(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :])
    else:
        raise ValueError("Unknown encoding type")
        
    # apply sin to even indices in the array and cos to odd
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)
