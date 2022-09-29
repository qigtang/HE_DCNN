import numpy as np
import tensorflow as tf
import itertools


class PadMixin(object):

  @classmethod
  def get_padding_as_op(cls, x, pads):
    x_rank = len(x.get_shape())
    x_shape = x.get_shape().as_list()
    spatial_size = x_rank - 2

    if len(pads) == spatial_size:
        pads = list(itertools.chain(*[[pad, pad] for pad in pads]))
    assert len(pads) // 2 == spatial_size

    num_dim = int(len(pads) / 2)
    tf_pads = np.transpose(np.array(pads).reshape([2, num_dim]))
    tf_pads = [0, 0, 0, 0] + tf_pads.flatten().tolist()

    padding = tf.constant(
        np.array(tf_pads).reshape([num_dim + 2, 2])
        .astype(np.int32))  # tf requires int32 paddings
    return tf.pad(x, padding)
