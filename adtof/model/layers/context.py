from keras import backend as K


def _add_context(x, context_frames):
    """
    Based on https://github.com/mcartwright/dafx2018_adt/blob/057ac6b1e39cd0c80554d52535cc9d88b6316c74/large_vocab_adt_dafx2018/model.py#L64
    """
    # Do not perform same padding by not padding extra frames
    # x = K.temporal_padding(x, (pad_frames, pad_frames))

    # Duplicate x multiple times equal to the number of context frames, with an offset.
    # The column i in to_concat[0] is at index i-1 in to_concat[1]. So i in to_concat[:] contains one frame and the following frames i+1, ..., i+n given as context
    to_concat = [x[:, offset : -(context_frames - offset - 1), :] for offset in range(context_frames - 1)]
    to_concat.append(x[:, (context_frames - 1) :, :])

    # Stack vertically the context frames together. Now the frame i has the following frames stacked on to of it (the columns contains duplicated values from the following ones)
    x = K.concatenate(to_concat, axis=2)
    return x
