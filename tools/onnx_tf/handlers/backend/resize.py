import tensorflow as tf

from onnx_tf.common import exception
from onnx_tf.common import data_type
from onnx_tf.common import sys_config
from onnx_tf.common.tf_helper import tf_shape
from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import partial_support
from onnx_tf.handlers.handler import ps_description


@onnx_op("Resize")
@partial_support(True)
@ps_description(
    "Resize required 4D input in Tensorflow. " +
    "For opset 11, only the following attributes and inputs " +
    "conbination are supported in Tensorflow:\n\t1. mode=nearest, " +
    "coordinate_transformation_mode=align_corners, nearest_mode=" +
    "round_prefer_ceil, can use scales(*) or sizes.\n" +
    "\t2. mode=nearest, coordinate_transformation_mode=asymmetric, " +
    "nearest_mode=floor, can use scales(*) or sizes.\n" +
    "\t3. mode=nearest, coordinate_transformation_mode=" +
    "tf_half_pixel_for_nn, nearest_mode=floor, can use scales(*) " +
    "or sizes.\n\t4. mode=linear, coordinate_transformation_mode=" +
    "align_corners, can use scales(*) or sizes.\n\t5. mode=linear, " +
    "coordinate_transformation_mode=asymmetric, can use scales(*) " +
    "or sizes.\n\t6. mode=linear, coordinate_transformation_mode=" +
    "half_pixel, can use scales(*) or sizes.\n\t7. mode=cubic, " +
    "coordinate_transformation_mode=align_corners, " +
    "cubic_coeff_a=-0.5, exclude_outside=1, can use scales(*) " +
    "or sizes.\n\t8. mode=cubic, coordinate_transformation_mode=" +
    "asymmetric, cubic_coeff_a=-0.5, exclude_outside=1, can use " +
    "scales(*) or sizes.\n\t9. mode=cubic, coordinate_transformation_mode=" +
    "half_pixel, cubic_coeff_a=-0.5, exclude_outside=1, can use " +
    "scales(*) or sizes.\n\t10. mode=nearest, " +
    "coordinate_transformation_mode=tf_crop_and_resize, " +
    "extrapolation_value=any_float_value, nearest_mode=round_prefer_ceil, " +
    "can use scales or sizes.\n\t11. mode=linear, " +
    "coordinate_transformation_mode=tf_crop_and_resize, " +
    "extrapolation_value=any_float_value, can use scales or sizes." +
    "\n\t- Note (*): The accuracy of your model will go down, if the " +
    "height and the width of the new sizes(scales * origial sizes) " +
    "are not in whole numbers.")

class Resize(BackendHandler):
  # setup supported_types and cast_map for x and roi
  x_supported_types = [
      tf.uint8, tf.uint16, tf.int8, tf.int16, tf.int32, tf.int64, tf.float16,
      tf.float32, tf.float64
  ]
  x_cast_map = {tf.uint32: tf.int64, tf.bool: None, tf.string: None}
  roi_supported_types = [tf.float32]
  roi_cast_map = {tf.float16: tf.float32}

  @classmethod
  def args_check(cls, node, **kwargs):
    # update cast_map base on auto_cast flag
    cls.x_cast_map[tf.uint64] = tf.int64 if sys_config.auto_cast else None
    cls.x_cast_map[tf.complex64] = tf.float64 if sys_config.auto_cast else None
    cls.x_cast_map[tf.complex128] = tf.float64 if sys_config.auto_cast else None
    cls.roi_cast_map[tf.float64] = tf.float32 if sys_config.auto_cast else None

    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    x_dtype = x.dtype
    if len(x_shape) != 4:
      exception.OP_UNSUPPORTED_EXCEPT("Resize required 4D input", "Tensorflow")
    if x_dtype in cls.x_cast_map and cls.x_cast_map[x_dtype] is None:
      exception.DTYPE_NOT_CAST_EXCEPT(
          "Resize input " + node.inputs[0] + " with data type '" +
          data_type.tf_to_np_str(x_dtype) + "'",
          data_type.tf_to_np_str_list(cls.x_supported_types))

    if cls.SINCE_VERSION >= 11:
      # supported attributes combination
      # ____________________________________________________________________________________________________________________________________________________
      # | mode    | coordinate_transformation_mode | cubic_coeff_a | exclude_outside | extrapolation_value | nearest_mode      | scales        | sizes     |
      # |_________|________________________________|_______________|_________________|_____________________|___________________|_______________|___________|
      # | nearest | align_corners                  | not apply     | 0               | not apply           | round_prefer_ceil | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | asymmetric                     | not apply     | 0               | not apply           | floor             | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | tf_half_pixel_for_nn           | not apply     | 0               | not apply           | floor             | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | align_corners                  | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | asymmetric                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | half_pixel                     | not apply     | 0               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | align_corners                  | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | asymmetric                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | cubic   | half_pixel                     | -0.5          | 1               | not apply           | not apply         | supported (1) | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | nearest | tf_crop_and_resize             | not apply     | 0               | any float value     | round_prefer_ceil | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # | linear  | tf_crop_and_resize             | not apply     | 0               | any float value     | not apply         | supported     | supported |
      # |---------|--------------------------------|---------------|-----------------|---------------------|-------------------|---------------|-----------|
      # Note:
      # 1. The accuracy of your model will go down, if the height and the width of the new sizes(scales * origial sizes) are not in whole numbers.
      coordinate_transformation_mode = node.attrs.get(
          "coordinate_transformation_mode", "half_pixel")
      cubic_coeff_a = node.attrs.get("cubic_coeff_a", -0.75)
      exclude_outside = node.attrs.get("exclude_outside", 0)
      mode = node.attrs.get("mode", "nearest")
      nearest_mode = node.attrs.get("nearest_mode", "round_prefer_floor")

      if coordinate_transformation_mode == "tf_crop_and_resize":
        roi = kwargs["tensor_dict"][node.inputs[1]]
        roi_dtype = roi.dtype
        if roi_dtype in cls.roi_cast_map and cls.roi_cast_map[roi_dtype] is None:
          exception.DTYPE_NOT_CAST_EXCEPT(
              "Resize input " + node.inputs[1] + " with data type '" +
              data_type.tf_to_np_str(roi_dtype) + "'",
              data_type.tf_to_np_str_list(cls.roi_supported_types))

      if coordinate_transformation_mode == "pytorch_half_pixel":
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize coordinate_transformation_mode=pytorch_half_pixel",
            "Tensorflow")
      if (coordinate_transformation_mode == "half_pixel" and mode == "nearest"
         ) or (coordinate_transformation_mode == "tf_half_pixel_for_nn" and
               mode in ["linear", "cubic"]) or (
                   coordinate_transformation_mode == "tf_crop_and_resize" and
                   mode == "cubic"):
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize coordinate_transformation_mode=" +
            coordinate_transformation_mode + " and  mode=" + mode, "Tensorflow")
      if (exclude_outside == 1 and
          mode in ["nearest", "linear"]) or (exclude_outside == 0 and
                                             mode == "cubic"):
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize mode=" + mode + " and exclude_outside=" +
            str(exclude_outside), "Tensorflow")
      if cubic_coeff_a != -0.5 and mode == "cubic":
        exception.OP_UNSUPPORTED_EXCEPT(
            "Resize mode=cubic and cubic_coeff_a=" + cubic_coeff_a,
            "Tensorflow")
      if mode == "nearest":
        if (nearest_mode in [
            "round_prefer_floor", "ceil"
        ]) or (coordinate_transformation_mode in [
            "align_corners", "tf_crop_and_resize"
        ] and nearest_mode == "floor") or (coordinate_transformation_mode in [
            "asymmetric", "tf_half_pixel_for_nn"
        ] and nearest_mode == "round_prefer_ceil"):
          exception.OP_UNSUPPORTED_EXCEPT(
              "Resize coordinate_transformation_mode=" +
              coordinate_transformation_mode +
              ", mode=nearest and nearest_mode=" + nearest_mode, "Tensorflow")

  @classmethod
  def version_10(cls, node, **kwargs):
    # x, roi and scales are all in NCHW format
    x = kwargs["tensor_dict"][node.inputs[0]]
    x_shape = x.get_shape().as_list()
    x_dtype = x.dtype
    scales = kwargs["tensor_dict"][node.inputs[1]]

    # get the new size from scales
    h_w_scale = scales[2:]
    h_w_shape = x_shape[2:]
    new_h_w_shape = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)

    mode = node.attrs.get("mode", "nearest")
    if mode.lower() == "linear":
      mode = tf.image.ResizeMethod.BILINEAR
    else:
      mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    # process tf.image.resize unsupported datatype for x
    x = tf.cast(x, cls.x_cast_map[x_dtype]) if x_dtype in cls.x_cast_map else x

    # The input image is in NCHW format. But tf.image.resize only
    # support channel last data format. Therefore need to transpose
    # to NHWC format first then process resize and then transpose
    # back to NCHW format.
    x_t = tf.transpose(x, perm=[0, 2, 3, 1])
    y = tf.image.resize_images(x_t, size=new_h_w_shape, method=mode)
    output = tf.transpose(y, perm=[0, 3, 1, 2])
    # cast output back to the original x.dtype
    output = tf.cast(output, x_dtype) if x_dtype is not tf.float32 else output
    return [output]

    # ----------------------------------------------------------------------
    n_in_scales_is_one = tf.equal(scales[0], 1)
    c_in_scales_is_one = tf.logical_or(tf.equal(scales[1], 1),
                                       tf.equal(scales[3], 1))
    assert_n_c_in_scales_are_ones = tf.Assert(
        tf.logical_and(n_in_scales_is_one, c_in_scales_is_one), [scales])

    with tf.control_dependencies([assert_n_c_in_scales_are_ones]):
      x_in_NCHW_format = tf.equal(scales[1], 1)
      h_w_scale = tf.where(x_in_NCHW_format, scales[2:], scales[1:3])
      h_w_shape = tf.where(x_in_NCHW_format, x_shape[2:], x_shape[1:3])
      new_h_w_shape = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype),
                              tf.int32)

      mode = node.attrs.get("mode", "nearest")
      if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
      else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

      def process_NCHW_format(x):
        x_t = tf.transpose(x, perm=[0, 2, 3, 1])
        y = tf.image.resize_images(x_t, size=new_h_w_shape, method=mode)
        y_t = tf.transpose(y, perm=[0, 3, 1, 2])
        return y_t

      def process_NHWC_format(x):
        y = tf.image.resize_images(x, size=new_h_w_shape, method=mode)
        return y

      output = tf.cond(x_in_NCHW_format, lambda: process_NCHW_format(x),
                       lambda: process_NHWC_format(x))

      return [output]
    # ----------------------------------------------------------------------

  @classmethod
  def version_11(cls, node, **kwargs):
    # x, roi, scales and sizes are all in NCHW format
    tensor_dict = kwargs["tensor_dict"]
    x = tensor_dict[node.inputs[0]]
    x_shape = tf_shape(x)
    x_dtype = x.dtype
    roi = tensor_dict[node.inputs[1]]
    roi_dtype = roi.dtype
    scales = tensor_dict[node.inputs[2]]
    sizes = tensor_dict[node.inputs[3]] if len(
        node.inputs) == 4 else tf.constant([], dtype=tf.int64)
    coordinate_transformation_mode = node.attrs.get(
        "coordinate_transformation_mode", "half_pixel")
    extrapolation_value = node.attrs.get("extrapolation_value", 0.0)
    mode = node.attrs.get("mode", "nearest")

    param = scales if len(node.inputs) == 3 else sizes

    if mode.lower() == "linear":
      tf_resize = tf.image.resize_bilinear
      mode = "bilinear"
    elif mode.lower() == "cubic":
      tf_resize = tf.image.resize_bicubic
    else:
      tf_resize = tf.image.resize_nearest_neighbor

    x_in_NCHW_format = tf.equal(param[1], 1)

    if len(node.inputs) == 3:  # only scales is defined
      h_w_scale = scales[2:]
      h_w_shape = x_shape[2:]
      new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)
    else:  # sizes is defined
      # The number of elements of 'sizes' should be the same as the rank of input 'X'
      sizes.set_shape(x_shape.shape)
      new_size = tf.cast(sizes[2:], tf.int32)
    # Tensorflow require the shape of "size" in the "tf.image.resize" must be known at
    # graph creation time. However in the dynamic shape situation, the shape of "new_size"
    # will be "None", the actual shape can only be determine at runtime. But we know
    # "new_size" should always contain [h, w], therefore the shape must be 2.
    new_size.set_shape([2])

    # process tf.image.resize and tf.image.crop_and_resize unsupported datatype for x
    x = tf.cast(x, cls.x_cast_map[x_dtype]) if x_dtype in cls.x_cast_map else x

    # The input image is in NCHW format. But tf.image.crop_and_resize,
    # tf.image.resize and tf.compat.v1.image.resize_xx only support
    # channel last data format. Therefore need to transpose to NHWC
    # formar first then process resize and then transpose back to
    # NCHW format.
    x_t = tf.transpose(x, perm=[0, 2, 3, 1])
    if coordinate_transformation_mode == "tf_crop_and_resize":
      # process tf.image.crop_and_resize unsupported datatype for boxes (roi in onnx resize)
      roi = tf.cast(
          roi,
          cls.roi_cast_map[roi_dtype]) if roi_dtype in cls.roi_cast_map else roi
      # get boxes for crop
      indices = []
      x_rank = len(x.get_shape())
      for i in range(2, x_rank):
        indices.insert(i - 2, i)
        indices.insert(i, i + x_rank)
      boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
      # get box_indices for crop
      box_indices = tf.cast(tf.range(0, x_shape[0]), dtype=tf.int32)
      # run ctop and resize
      y = tf.image.crop_and_resize(x_t, boxes, box_indices, new_size, mode,
                                   extrapolation_value)
    elif coordinate_transformation_mode == "align_corners":
      y = tf_resize(x_t,
                    size=new_size,
                    align_corners=True,
                    half_pixel_centers=False)
    elif coordinate_transformation_mode == "asymmetric":
      y = tf_resize(x_t,
                    size=new_size,
                    align_corners=False,
                    half_pixel_centers=False)
    else:  # half_pixel or tf_half_pixel_for_nn
      y = tf_resize(x_t,
                    size=new_size,
                    align_corners=False,
                    half_pixel_centers=True)
    output = tf.transpose(y, perm=[0, 3, 1, 2])
    # cast output back to the original x.dtype
    output = tf.cast(output, x_dtype) if x_dtype is not tf.float32 else output

    return [output]