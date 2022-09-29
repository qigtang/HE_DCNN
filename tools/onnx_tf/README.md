## 1. 简介

Facebook 主推的 Pytorch 简洁而不简单，相较于 Google 的 tensorflow 活跃度日益增加，很多新方法（如 yolov5, nanodet)、 新框架（mmdetection， mmsegmentation， detectron2) 基于 pytorch 实现。
Pytorch 已经成为优选的炼丹框架。 但是由于历史兼容性原因，现有部署框架基于 tensorflow 1.4 实现。

如何将两者结合起来，充分利用 pytorch 进行训练，利用 tensorflow 进行部署？ Microsoft 的 onnx 模型可以作为架起两个框架的桥梁。 Onnx 是一个开放格式，定义了一组公共的深度学习和机器学习算子，和一个公共的数据文件格式，使 AI 开发者可以在不同框架、工具、运行时、编译器等下使用模型。

Pytorch 库中内置了导出 onnx 模型的接口，用户可以直接使用内置的 `torch.onnx.export` 导出训练好的模型文件到 onnx 格式。但是 Tensorflow 库官方并不直接支持 onnx。借助 onnx 组织的 [onnx-tensorflow(简称 onnxtf 或 onnx_tf)](https://github.com/onnx/onnx-tensorflow) 可以将 onnx 格式模型转换到 tensorflow 支持的 pb 格式。

但是，onxntf 依赖 onnx 和 tensorflow。不同 onnxtf 版本依赖的 onnx 和  tensorflow 版本不同，兼容性成为一个公共而又暂未解决的问题。如为了生成 tensorflow 1.4 支持的 pb 模型，需要使用 onnx2tf 1.2 左右，同时依赖的 onnx 版本较低。现有检测框架（mmdetection等）依赖依赖的 pytorch 版本比较高。当 onnx 版本较低时，无法加载高版本 pytorch 训练好并导出的 onnx 模型。从而中断了转换链。

经过尝试，发现 onnx 的 opset_version=9 时，对常见的深度学习算子支持较为完善。而 pytorch 1.x 基本都支持导出 opset_version = 9 格式的 onnx 模型文件。而 onnxtf 的最高版本 1.7 (一个月前修改时还是 1.6) 支持 opset_version = 9，且对 opset_version = 11 等也有支持。因此尝试使用 onnxtf 1.7 转换模型。 onnxtf1.7 官方支持 tensorflow 2.3.1 ，而很多 tensorflow 2.x 的函数在 tensorflow 1.4 中没有。 因此需要修改兼容性，增加对 tensorflow 1.4 的兼容性（同时发现，`导出的 tf1.4 的模型一般可以直接在 tf1.13 上运行`）。

## 2. 主要修改
主要兼容性修改有：

### 2.1 移除 `tf.compat.v1` 相关的前缀，替换为对应的函数。

```
# 高版本用到了 compat：
# ret = tf.compat.v1.image.crop_and_resize
# 低版本移除 compat：
ret = tf.image.crop_and_resize
```

### 2.2 增加对 math、random、nn 等相关模块兼容性。

如：

```python
# 增加对 random 模块兼容性
if not hasattr(tf, "random"):
  class TfRandom:
      normal = tf.random_normal
      uniform = tf.random_uniform
  tf.random = TfRandom
```

### 2.3 增加 max_pool_v2 操作兼容性。

```python
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
```

### 2.4 增加对 pool、resize_xxx 函数兼容性

```python
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
```

### 2.5 增加对 conv、pad、upsample 等操作的修改。

### 2.6 集成两个便捷函数 `onnx2tf` 和  `runTf`。

其中 onnx2tf 直接将 onnx 模型转换到  pb 模型，并提示输出输出节点; runTf 函数则加载 pb 文件运行推理。


## 3. 用法

### 3.1 搭建训练和转换环境

训练环境： pytorch(如 1.7), onnx(如 1.8)
转换环境： tensorflow-gpu(如 1.4, 1.13), onnx( 如1.8 )

如果没有现有环境，可以使用我在 100 服务器上的备用 conda 环境。
其中，训练 python 环境 (pytorch 1.7)：

```bash
/opt/miniconda3/envs/th/bin/python3.7
```

转换 python 环境 (tensorflow-gpu 1.4.1)：

```bash
/opt/miniconda3/envs/tf104/bin/python3.6
```

### 3.2 训练环境下，使用 pytorch 导出 onnx 格式。

在 pytorch (如1.7) 环境下训练模型，导出为 onnx 格式文件。
注意，为了更好地转换到 onnx，并最终转换到 tensorflow，尽量避免导出含有切片等动态操作的层（如 yolov5 的 focus 层）。
同时，测试发现对 opset_version=9 或 11 支持比较好，并尽量使用 opset_version=9。

```bash
torch.onnx.export(model, x, "exported.onnx", opset_version=9)
```

### 3.3 测试环境下，使用 onnxtf 转换 onnx 到 ob 格式。

在 tf1.4 或 tf1.13 (GPU) 环境下，运行该工具，导出为 pb 文件。

验证过：lab 模型， mmdetection 版 yolov5 模型，可转换并正常推理。
待验证：mmdetection 版  retinanet 可转换，但尚未验证。

```py
#export CUDA_VISIBLE_DEVICES=0

fpath  = "exported.onnx"
fpath2 = "exported.pb"

# 添加 onnxtf 路径, 导入 onnxtf
import sys
sys.path.insert(0, "THE_ONNXTF_ROOT")

from onnx_tf import onnx2tf
onnx2tf(fpath, fpath2)
```

### 3.3 依赖与安装

依赖： onnx>=1.5，tensorflow-1.4 （1.13 也可）。

安装: 可以将 onnx_tf 软链接到 python 路径下，也可以设置 PYTHONPATH 环境变量，包含该 onnx_tf 路径。