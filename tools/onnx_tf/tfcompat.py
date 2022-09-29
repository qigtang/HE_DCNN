# 2020/11/30
# created by jinlj
# Add compatibility to onnxtf-1.7 for tf-1.4 and tf-1.13
# Tested for yolov5, lab, unet, retinanet (without focus, postprocessing).

import functools
import tensorflow as tf

from tensorflow.python.framework import ops
import collections.abc as collections_abc

# 增加对 math 兼容性
if not hasattr(tf, "math"):
  class TfMath:
      ceil = tf.ceil
      floor = tf.floor
      count_nonzero = tf.count_nonzero
      is_inf = tf.is_inf
      is_nan = tf.is_nan
      erf = tf.erf
      floormod = tf.floormod
      truediv = tf.truediv
      floordiv = tf.floordiv
      pow = tf.pow
      abs = tf.abs
      minimum = tf.minimum
      maximum = tf.maximum
      reduce_min = tf.reduce_min
      reduce_max = tf.reduce_max
      add = tf.add
      subtract = tf.subtract
      multiply = tf.multiply
      reciprocal = tf.reciprocal
      log = tf.log
      logical_xor = tf.logical_xor
      cumsum = tf.cumsum
  tf.math = TfMath

# 增加对 nn 模块兼容性
if not hasattr(tf.nn, "space_to_batch"):
  tf.nn.space_to_batch = tf.space_to_batch
  tf.nn.depth_to_space = tf.depth_to_space

# 增加对 random 模块兼容性
if not hasattr(tf, "random"):
  class TfRandom:
      normal = tf.random_normal
      uniform = tf.random_uniform
  tf.random = TfRandom

if not hasattr(tf, "uint32"):
    tf.uint32 = tf.int32
    tf.uint64 = tf.int64

def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  # Performance is fast-pathed for common cases:
  # `None`, `list`, `tuple` and `int`.
  if value is None:
    return [1] * (n + 2)

  # Always convert `value` to a `list`.
  if isinstance(value, list):
    pass
  elif isinstance(value, tuple):
    value = list(value)
  elif isinstance(value, int):
    value = [value]
  elif not isinstance(value, collections_abc.Sized):
    value = [value]
  else:
    value = list(value)  # Try casting to a list.

  len_value = len(value)

  # Fully specified, including batch and channel dims.
  if len_value == n + 2:
    return value

  # Apply value to spatial dims only.
  if len_value == 1:
    value = value * n  # Broadcast to spatial dimensions.
  elif len_value != n:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, len_value))

  # Add batch and channel dims (always 1).
  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]

# 增加对 max_pool 兼容性
def max_pool_v2(input, ksize, strides, padding, data_format=None, name=None):
  """Performs the max pooling on the input.
  """
  if input.shape is not None:
    n = len(input.shape) - 2
  elif data_format is not None:
    n = len(data_format) - 2
  else:
    raise ValueError(
        "The input must have a rank or a data format must be given.")
  if not 1 < n <= 3:
    raise ValueError("Input tensor must be of rank 4 or 5 but was {}.".format(n + 2))

  if data_format is None:
    channel_index = n + 1
  else:
    channel_index = 1 if data_format.startswith("NC") else n + 1

  ksize = _get_sequence(ksize, n, channel_index, "ksize")
  strides = _get_sequence(strides, n, channel_index, "strides")

  max_pooling_ops = {
      #1: max_pool1d,
      2: tf.nn.max_pool,
      3: tf.nn.max_pool3d
  }

  op = max_pooling_ops[n]
  return op(
      input,
      ksize=ksize,
      strides=strides,
      padding=padding,
      data_format=data_format,
      name=name)

if not hasattr(tf.nn, "max_pool_v2"):
    tf.nn.max_pool_v2 = max_pool_v2

# ------------------------------------------------------------------
def check_tfversion():
    return (tf.__version__.startswith("1.4") or tf.__version__.startswith("1.13"))
    return True

# 增加对 pool 兼容性
def pool_func(func, *args, **kwargs):
    if "dilations" in kwargs and check_tfversion():
        kwargs["dilation_rate"] = kwargs.pop("dilations")
    return func(*args, **kwargs)

tf.nn.pool = functools.partial(pool_func, tf.nn.pool)

# 增加对 resize 兼容性
def resize_func(func, *args, **kwargs):
    if check_tfversion() and "half_pixel_centers" in kwargs:
        kwargs.pop("half_pixel_centers")
        print(kwargs)
    return func(*args, **kwargs)

tf.image.resize_bilinear = functools.partial(resize_func, tf.image.resize_bilinear)
tf.image.resize_bicubic = functools.partial(resize_func, tf.image.resize_bicubic)
tf.image.resize_nearest_neighbor = functools.partial(resize_func, tf.image.resize_nearest_neighbor)