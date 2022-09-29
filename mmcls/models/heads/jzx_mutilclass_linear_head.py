import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
import numpy as np

from ..builder import HEADS
from .cls_head import ClsHead
from .jzx_cls_head import jzx_ClsHead

@HEADS.register_module()
class jzx_mutilclass_LinearClsHead(jzx_ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes, # 不用管
                 num_classes_list,  # todo: list, eg:[4,4,3,2,5,2,2,2,3]

                 cross_entropy_weights,
                 hanming_weights,
                 loss_weights,

                 cost_sensitive,
                 in_channels,
                 loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk=(1, )):
        super(jzx_mutilclass_LinearClsHead, self).__init__(loss=loss, topk=topk)
        self.in_channels = in_channels
        self.num_classes_list = num_classes_list
        self.num_classes = num_classes   # 不用管

        self.cross_entropy_weights = cross_entropy_weights
        self.hanming_weights = hanming_weights
        self.loss_weights = loss_weights

        self.cost_sensitive = cost_sensitive

        self.gap = nn.AdaptiveAvgPool2d((1, 1)) # todo:

        if len(self.num_classes_list) <= 0:
            raise ValueError(
                f'num_classes_list={num_classes_list} must be a positive integer')

        self._init_layers()



    def _init_layers(self):

        self.fc_list = nn.ModuleList(nn.Linear(self.in_channels, num_classes) for num_classes in self.num_classes_list)

        # for i, num_classes in enumerate(self.num_classes_list): # 遍历加层
        #     layer_name = f'fc{i + 1}'
        #     linear_layer = nn.Linear(self.in_channels, num_classes)
        #     self.add_module(layer_name, linear_layer)




    def init_weights(self):

        normal_init(self.fc_list, mean=0, std=0.01, bias=0)

        # for i, num_classes in enumerate(self.num_classes_list): # 遍历加层
        #     layer_name = f'fc{i + 1}'
        #     normal_init(layer_name, mean=0, std=0.01, bias=0)


    def simple_test(self, img):

        # """Test without augmentation."""
        # cls_score = self.fc(img)
        # if isinstance(cls_score, list):
        #     cls_score = sum(cls_score) / float(len(cls_score))
        # pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        # if torch.onnx.is_in_onnx_export():
        #     return pred
        # pred = list(pred.detach().cpu().numpy())
        # return pred

        pred_list = []
        pred_argmax_list = []
        for i, fc in enumerate(self.fc_list): # 遍历加层

            outs = self.gap(img)
            outs = outs.view(img.size(0), -1)
            cls_score = fc(outs)

            if isinstance(cls_score, list):
                cls_score = sum(cls_score) / float(len(cls_score))
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
            pred_numpy = pred.detach().cpu().numpy()
            pred_list.append(pred_numpy)

            pred_argmax = torch.argmax(pred, dim=1) if pred is not None else None
            pred_argmax_list.append(pred_argmax.cpu().numpy())

        pred_argmax_list_numpy = np.asarray(pred_argmax_list)
        pred_argmax_list_numpy = np.transpose(pred_argmax_list_numpy)
        pred_argmax_list_last = pred_argmax_list_numpy.tolist()


        pred_list_0_numpy = np.array(pred_list[0])
        for i, pred_list_cu in enumerate(pred_list):
            if i > 0:
                pred_list_cu_numpy = np.array(pred_list_cu)
                pred_list_0_numpy = np.concatenate( (pred_list_0_numpy, pred_list_cu_numpy), 1)

        pred_list_0_numpy_list = pred_list_0_numpy.tolist()

        return pred_list_0_numpy_list
        # return pred_argmax_list_last


    # 汉明距离
    def hanming_loss(self, cls_score0, cu_gt_label0, num_class0):
        if isinstance(cls_score0, list):
            cls_score0 = sum(cls_score0) / float(len(cls_score0))
        cls_score_softmax = F.softmax(cls_score0, dim=1) if cls_score0 is not None else None
        cls_score_argmax= torch.argmax(cls_score_softmax, dim=1) if cls_score_softmax is not None else None

        cu_max = (cu_gt_label0.cpu().max()).data.numpy()
        if cu_max >= num_class0:
            print('ooo')




        cls_score_argmax_onehot = F.one_hot(cls_score_argmax, num_classes=num_class0)
        cu_gt_label_onehot = F.one_hot(cu_gt_label0, num_classes=num_class0)

        loss = (cls_score_argmax_onehot != cu_gt_label_onehot).sum()
        loss = torch.div(loss.float(), cls_score0.shape[0] * 2)  # 要多除以一个2, 因为一般1个元素不一致，导致onehot中至少两个值不一致
        return loss



    def forward_train(self, x, gt_label):

        losses_total = dict()
        if self.cal_acc:
            for k in self.topk:
                a = torch.tensor(0.).cuda()
                losses_total['accuracy'] = {f'top-{k}': a}
        losses_total['loss'] = torch.tensor(0.).cuda()
        losses_total['losses_hanming_total'] = torch.tensor(0.).cuda()


        for i, fc in enumerate(self.fc_list): # 遍历加层

            outs = self.gap(x)
            outs = outs.view(x.size(0), -1)
            cls_score = fc(outs)
            cu_gt_label = gt_label[:, i]
            cu_cost_sensitive = self.cost_sensitive[str(i)]


            losses_hanming = self.hanming_loss(cls_score, cu_gt_label, self.num_classes_list[i])
            losses_total['losses_hanming_' + str(i)] = losses_hanming  # 单个特征的losses_hanming
            losses_total['losses_hanming_total'] += losses_hanming * self.hanming_weights[i]  # 所有特征的总losses_hanming




            losses_total['loss'+str(i)] = torch.tensor(0.).cuda() # 单个特征的loss
            losses = self.loss(cls_score, cu_gt_label, cu_cost_sensitive)  # 对于'accuracy'-’top-1‘计算有误， 特别是在两维的时候

            cls_argmax_ = torch.argmax(cls_score, dim=1) if cls_score is not None else None
            loss_ = (cls_argmax_ == cu_gt_label).sum()
            losses['accuracy']['top-1'] = torch.div(loss_.float(), cls_score.shape[0] * 0.01 )


            lo = 1
            for key, value in losses.items():
                if key == 'loss':
                    losses_total['loss' + str(i)] = value # 单个特征的loss
                    losses_total[key] += value * self.cross_entropy_weights[i]# 所有特征的总loss

                elif key == 'accuracy':
                    losses2 = losses[key]
                    losses_total2 = losses_total[key]
                    for key2, value2 in losses2.items():
                        dd = losses_total2[key2]
                        ddd = value2
                        losses_total2[key2] = ddd + dd
                    losses_total[key] = losses_total2

        losses_total['accuracy'][key2] = losses_total['accuracy'][key2] / len(self.num_classes_list)
        losses_total['loss'] = losses_total['loss']*self.loss_weights[0] + losses_total['losses_hanming_total']*self.loss_weights[1]

        return losses_total





    # 备份
    def forward_train_bak(self, x, gt_label):

        losses_total = dict()
        if self.cal_acc:
            for k in self.topk:
                a = torch.tensor(0.).cuda()
                losses_total['accuracy'] = {f'top-{k}': a}
        losses_total['loss'] = torch.tensor(0.).cuda()


        for i, fc in enumerate(self.fc_list): # 遍历加层
            cls_score = fc(x)
            cu_gt_label = gt_label[:, i]
            cu_cost_sensitive = self.cost_sensitive[str(i)]

            num_class = self.num_classes_list[i]
            losses_hanming = self.hanming_loss(cls_score, cu_gt_label, num_class)


            losses = self.loss(cls_score, cu_gt_label, cu_cost_sensitive)
            for key, value in losses.items():
                if key == 'loss':
                    losses_total[key] += value

                elif key == 'accuracy':
                    losses2 = losses[key]
                    losses_total2 = losses_total[key]
                    for key2, value2 in losses2.items():
                        dd = losses_total2[key2]
                        ddd = value2
                        dddd = ddd + dd
                        losses_total2[key2] = dddd / len(self.num_classes_list)
                    losses_total[key] = losses_total2

        losses_total['loss'] = losses_total['loss'] / len(self.num_classes_list)
        return losses_total