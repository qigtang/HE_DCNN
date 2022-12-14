import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self):
        super(GlobalAveragePooling, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs



@NECKS.register_module()
class GlobalAveragePooling_jzx(nn.Module):  # 这里不用pool， 放在head中
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling.
    We do not use `squeeze` as it will also remove the batch dimension
    when the tensor has a batch dimension of size 1, which can lead to
    unexpected errors.
    """

    def __init__(self):
        super(GlobalAveragePooling_jzx, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):


        # if isinstance(inputs, tuple):
        #     outs = tuple([self.gap(x) for x in inputs])
        #     outs = tuple(
        #         [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        # elif isinstance(inputs, torch.Tensor):
        #     outs = self.gap(inputs)
        #     outs = outs.view(inputs.size(0), -1)
        # else:
        #     raise TypeError('neck inputs should be tuple or torch.tensor')
        # return outs

        return inputs

