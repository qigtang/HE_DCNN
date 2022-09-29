import copy

import numpy as np
import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description
from onnx_tf.handlers.handler import tf_func
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("Upsample")
@tf_func(tf.image.resize_images)
@partial_support(True)
@ps_description("Upsample required 4D input in Tensorflow.")
class Upsample(BackendHandler):

  @classmethod
  def args_check(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Upsample without 4D input", "Tensorflow")

    if node.attrs.get("mode", "nearest").lower() not in ["nearest", "bilinear", "linear"]:
      exception.OP_UNSUPPORTED_EXCEPT("Upsample without nearest or bilinear",
                                      "Tensorflow")


  @classmethod
  def version_1(cls, node, **kwargs):
      print("==> Upsample only support NCHW format!! ==")
      x = kwargs["tensor_dict"][node.inputs[0]]
      x_shape = x.get_shape().as_list()
      if hasattr(node,"attrs"):
          attrs = copy.deepcopy(node.attrs)

      if len(node.inputs) == 2:
          scales = kwargs["tensor_dict"][node.inputs[1]]
          if True:
              h_w_scale = scales[2:]
              h_w_shape = x_shape[2:]
              new_h_w_shape = tf.cast(h_w_scale * h_w_shape, tf.int32)

              mode = attrs.get("mode", "nearest")
              if mode.lower() == "bilinear":
                  mode = tf.image.ResizeMethod.BILINEAR
              else:
                  mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

              attrs["size"] = new_h_w_shape
              attrs["method"] = mode

              # Remove scale.
              upsample_node = copy.deepcopy(node)
              import pdb; pdb.set_trace()
              del upsample_node.inputs[1]
              return [
                  cls.make_tensor_from_onnx_node(
                      upsample_node, attrs=attrs, c_last_only=True, **kwargs)
              ]
      else:
          if "scales" in attrs:
              # version_7
              scales = attrs["scales"] # (nb, nc, nh, nw)
              new_height = np.floor(x_shape[2] * scales[2])
              new_width  = np.floor(x_shape[3] * scales[3])
          elif "height_scale" in attrs  and "width_scale" in attrs:
              # version_<7
              new_height = np.floor(x_shape[2] * attrs["height_scale"])
              new_width  = np.floor(x_shape[3] * attrs["width_scale"])
          else:
              raise Exception("Unsupported Upsample!!!")

          mode = attrs.get("mode", "nearest")
          if mode.lower() == "bilinear":
              mode = tf.image.ResizeMethod.BILINEAR
          else:
              mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

          attrs["size"] = np.array((new_height, new_width), dtype=np.int32)
          attrs["method"] = mode

          return [
              cls.make_tensor_from_onnx_node(
                  node, attrs=attrs, c_last_only=True, **kwargs)
          ]


  @classmethod
  def version_7(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    attrs = copy.deepcopy(node.attrs)
    scales = attrs["scales"]
    new_height = np.floor(x_shape[2] * scales[2])
    new_weight = np.floor(x_shape[3] * scales[3])

    mode = attrs.get("mode", "nearest")
    if mode.lower() == "bilinear" or mode.lower() == "linear":
      mode = tf.image.ResizeMethod.BILINEAR
    else:
      mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    attrs["size"] = np.array((new_height, new_weight), dtype=np.int32)
    attrs["method"] = mode

    return [
        cls.make_tensor_from_onnx_node(
            node, attrs=attrs, c_last_only=True, **kwargs)
    ]

  @classmethod
  def version_9(cls, node, **kwargs):
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = tf_shape(x)
    attrs = copy.deepcopy(node.attrs)
    scales = kwargs["tensor_dict"][node.inputs[1]]

    h_w_scale = scales[2:]
    h_w_shape = x_shape[2:]
    new_h_w_shape = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype),
                            tf.int32)

    mode = attrs.get("mode", "nearest")
    if mode.lower() == "bilinear" or mode.lower() == "linear":
      mode = tf.image.ResizeMethod.BILINEAR
    else:
      mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    attrs["size"] = new_h_w_shape
    attrs["method"] = mode

    # Remove scale.
    upsample_node = copy.deepcopy(node)
    del upsample_node.inputs[1]
    return [
        cls.make_tensor_from_onnx_node(
            upsample_node, attrs=attrs, c_last_only=True, **kwargs)
    ]
