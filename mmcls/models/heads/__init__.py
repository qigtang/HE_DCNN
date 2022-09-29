from .cls_head import ClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .jzx_mutilclass_linear_head import jzx_mutilclass_LinearClsHead
from .jzx_cls_head import jzx_ClsHead


__all__ = [
    'ClsHead', 'LinearClsHead', 'MultiLabelClsHead', 'MultiLabelLinearClsHead', 'jzx_mutilclass_LinearClsHead', 'jzx_ClsHead'
]
