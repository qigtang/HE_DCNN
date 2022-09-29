
import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, \
    cohen_kappa_score, precision_recall_fscore_support

import scipy
import torch.nn.functional as F



def AIresult_postpress(shiyan_type, true_label_list, features_blobs0, pic_id_list):
    #
    if shiyan_type == 'baochong_feibao_2':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [0, 1, 2]:
                true_label_list_re.append(0)
            elif i in [3, 4]:
                true_label_list_re.append(1)
            features_blobs0_.append([ features_blobs0[index][0], features_blobs0[index][1], features_blobs0[index][2], features_blobs0[index][3], features_blobs0[index][4] ])
            pic_id_list_re.append(pic_id_list[index])
        scores_ = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        scores = []
        for index, _ in enumerate(scores_):
            value1 = scores_[index][0] + scores_[index][1] + scores_[index][2]
            value2 = scores_[index][3] + scores_[index][4]
            scores.append([value1, value2])
        pred_label = np.argmax(np.array(scores), axis=1)
        scores = np.array(scores)


    elif shiyan_type == 'pao_feibao_2':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [0, 3, 4]:
                if i == 0:
                    true_label_list_re.append(0)
                else:
                    true_label_list_re.append(1)
                features_blobs0_.append([features_blobs0[index][0], features_blobs0[index][3] , features_blobs0[index][4]])
                pic_id_list_re.append(pic_id_list[index])
        scores_ = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        scores = []
        for index, _ in enumerate(scores_):
            value1 = scores_[index][0]
            value2 = scores_[index][1] + scores_[index][2]
            scores.append([value1, value2])
        pred_label = np.argmax(scores, axis=1)
        scores = np.array(scores)

    elif shiyan_type == 'nang_pao_feibao_3':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [1, 2]:
                true_label_list_re.append(0)
            elif i in [0]:
                true_label_list_re.append(1)
            elif i in [3, 4]:
                true_label_list_re.append(2)
            features_blobs0_.append([ features_blobs0[index][0], features_blobs0[index][1], features_blobs0[index][2], features_blobs0[index][3], features_blobs0[index][4] ])
            pic_id_list_re.append(pic_id_list[index])
        scores_ = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        scores = []
        for index, _ in enumerate(scores_):
            value1 = scores_[index][1] + scores_[index][2]
            value2 = scores_[index][0]
            value3 = scores_[index][3] + scores_[index][4]
            scores.append([value1, value2, value3])
        pred_label = np.argmax(np.array(scores), axis=1)
        scores = np.array(scores)

    elif shiyan_type == 'pao_liangfeibao_efeibao_3':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [0, 3, 4]:
                if i in [0]:
                    true_label_list_re.append(0)
                elif i in [3]:
                    true_label_list_re.append(1)
                elif i in [4]:
                    true_label_list_re.append(2)
                features_blobs0_.append([features_blobs0[index][0], features_blobs0[index][3], features_blobs0[index][4]])
                pic_id_list_re.append(pic_id_list[index])
        scores = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        pred_label = np.argmax(scores, axis=1)

    elif shiyan_type == 'nang_pao_liangfeibao_efeibao_4':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [1, 2]:
                true_label_list_re.append(0)
            elif i in [0]:
                true_label_list_re.append(1)
            elif i in [3]:
                true_label_list_re.append(2)
            elif i in [4]:
                true_label_list_re.append(3)
            features_blobs0_.append([ features_blobs0[index][0], features_blobs0[index][1], features_blobs0[index][2], features_blobs0[index][3], features_blobs0[index][4] ])
            pic_id_list_re.append(pic_id_list[index])
        scores_ = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        scores = []
        for index, _ in enumerate(scores_):
            value1 = scores_[index][1] + scores_[index][2]
            value2 = scores_[index][0]
            value3 = scores_[index][3]
            value4 = scores_[index][4]
            scores.append([value1, value2, value3, value4])
        pred_label = np.argmax(np.array(scores), axis=1)
        scores = np.array(scores)

    elif shiyan_type == 'nang_nang_2':
        true_label_list_re = []
        features_blobs0_ = []
        pic_id_list_re = []
        for index, i in enumerate(true_label_list):
            if i in [1, 2]:
                if i in [1]:
                    true_label_list_re.append(0)
                elif i in [2]:
                    true_label_list_re.append(1)
                features_blobs0_.append([features_blobs0[index][1], features_blobs0[index][2]])
                pic_id_list_re.append(pic_id_list[index])
        scores = (F.softmax(torch.tensor(np.array(features_blobs0_)))).data.cpu().numpy()
        pred_label = np.argmax(scores, axis=1)

    elif shiyan_type == 'pao_nang1_2_lfeibao_efeibao_5':
        true_label_list_re = true_label_list
        scores = (F.softmax(torch.tensor(features_blobs0))).data.cpu().numpy()
        pred_label = np.argmax(scores, axis=1)
        pic_id_list_re = pic_id_list

    return true_label_list_re, scores, pred_label, pic_id_list_re





# 计算AI结果
def cal_AIresult_by_shiyan_data_type(num_classes, scores, pred_label, true_label_list, org_patience_dir, pic_id_list):
    # 每个病人id 的推理结果
    org_patience_list = os.listdir(org_patience_dir)
    save_line_1 = []  # 病人id
    save_line_2 = []  # 病人id的预测类别
    save_line_3 = []  # 病人id的真实类别
    save_line_4 = []  # 病人id的预测类别得分
    for cu_patience in org_patience_list:
        cu_patience_flag = False  # 判断该病人目录下的图片，是否在dataload中;
        # print(cu_patience)
        cu_patience_dir = os.path.join(org_patience_dir, cu_patience)
        cu_patience_pic_lables = []
        cu_patience_pic_lables_org = []
        cu_patience_pic_scores = [0.0]*num_classes
        for cu_patience_pic in os.listdir(cu_patience_dir):
            if cu_patience_pic in pic_id_list:
                cu_patience_flag = True

                cu_index = pic_id_list.index(cu_patience_pic)
                cu_patience_pic_scores += scores[cu_index]
                cu_patience_pic_lable = pred_label[cu_index]
                cu_patience_pic_lables.append(cu_patience_pic_lable)
                cu_patience_pic_lable_org = true_label_list[cu_index]
                cu_patience_pic_lables_org.append(cu_patience_pic_lable_org)
        if cu_patience_flag:
            cu_patience_pic_scores_average = (cu_patience_pic_scores / len(cu_patience_pic_lables)).tolist()
            maxlabel = cu_patience_pic_scores_average.index(max(cu_patience_pic_scores_average))
            maxlabel_org = max(cu_patience_pic_lables_org, key=cu_patience_pic_lables_org.count)
            save_line_1.append(cu_patience)
            save_line_2.append(maxlabel)
            save_line_3.append(maxlabel_org)
            save_line_4.append(cu_patience_pic_scores_average)

    round_save_line_4 = np.around(save_line_4, decimals=4)
    return save_line_1, save_line_2, save_line_3, round_save_line_4



def save_data_in_sheet(excel_patience, num_classes, lable_list, save_line_1, save_line_3, save_line_4):
    #
    for i in range(num_classes):
        excel_patience.write(0, i+1, lable_list[i])  # todo:
    excel_patience.write(0, num_classes+1, '真实标签')

    for i, data in enumerate(save_line_1):
        excel_patience.write(i + 1, 0, save_line_1[i])
        save_line_4_data = save_line_4[i]

        for j in range(num_classes):
            excel_patience.write(i + 1, j + 1, save_line_4_data[j])
        excel_patience.write(i + 1, num_classes+1, round(save_line_3[i]))



def cal_evaluation_two_class(shiyan_type, data_type, save_line_2, save_line_3, save_line_4):
    #
    colors = ['purple', 'g', 'darkorange', 'y']
    labels = ['Validation', 'External test', 'Internal test', 'Prospective test']

    if data_type == 'val':
        plt.figure()
        color = colors[0]
        label = labels[0]
    elif data_type == 'test_waibu':
        color = colors[1]
        label = labels[1]
    elif data_type == 'test_neibu':
        color = colors[2]
        label = labels[2]
    elif data_type == 'test_qianzhan':
        color = colors[3]
        label = labels[3]

    class_precision_list = []
    class_recall_list = []
    class_specificity_list = []
    class_yingxing_list = []
    fpr_list = []
    tpr_list = []
    if True:
        class_id = 0
        class_pre = [0 if i == class_id else 1 for i in save_line_2]
        class_org = [0 if i == class_id else 1 for i in save_line_3]
        class_matrix = confusion_matrix(class_org, class_pre)
        TP, FN, FP, TN = class_matrix[0, 0], class_matrix[0, 1], class_matrix[1, 0], class_matrix[1, 1]
        if TP + FP == 0:
            class_precision = 0.0
        else:
            class_precision = 1.0 * TP / (TP + FP)
        if TP + FN == 0:
            class_recall = 0.0
        else:
            class_recall = 1.0 * TP / (TP + FN)
        if FP + TN == 0:
            class_specificity = 0.0
        else:
            class_specificity = 1.0 * TN / (FP + TN)
        if TN + FN == 0:
            class_yingxing = 0.0
        else:
            class_yingxing = 1.0 * TN / (TN + FN)
        class_precision_list.append(class_precision)
        class_recall_list.append(class_recall)
        class_specificity_list.append(class_specificity)
        class_yingxing_list.append(class_yingxing)
        class_pre_scores = [1 - i[class_id] for i in save_line_4]
        auc = roc_auc_score(class_org, class_pre_scores)
        fpr, tpr, thresholds = roc_curve(class_org, class_pre_scores)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

        auc_print = '0·' + (str(round(auc, 3))).split('.')[-1]
        plt.plot(fpr, tpr, color=color, lw=2, linestyle='-', label=label + ',AUC=' + auc_print)
        # plt.plot(fpr, tpr, color=color, lw=2, linestyle='-')


    if data_type == 'test_qianzhan':
        ax = plt.axes()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # plt.plot([0, 1], [0, 1], color='lightcoral', lw=2, linestyle='--')

        plt.legend(loc="lower right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.gca().xaxis.set_major_formatter(
            plt.FixedFormatter(['0·0', '0·2', '0·4', '0·6', '0·8', '1·0']))
        plt.gca().yaxis.set_major_formatter(
            plt.FixedFormatter(['0·0', '0·2', '0·4', '0·6', '0·8', '1·0']))

        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt_save_name = './out/yes_wbl_caijian_adam_small/' + \
                        'yes_wbl' + '_' + \
                        shiyan_type + '_' \
                        '_ROC曲线.svg'
        plt.savefig(plt_save_name, dpi=300)
        plt.close()


    # 以病人id， 计算相关指标
    my_precision, my_recall, my_f1, _ = precision_recall_fscore_support(save_line_3, save_line_2, average=None)
    my_confusion_matrix = confusion_matrix(save_line_3, save_line_2)
    my_accuracy = accuracy_score(save_line_3, save_line_2)
    my_kappa = cohen_kappa_score(save_line_3, save_line_2)
    my_precision_average = class_precision_list[0]
    my_recall_average = class_recall_list[0]
    my_specificity_average = class_specificity_list[0]
    my_yingxing_average = class_yingxing_list[0]
    my_f1_average = my_f1[0]

    return fpr,tpr,auc,\
           my_accuracy,\
           my_precision_average,\
           my_recall_average,\
           my_specificity_average,\
           my_yingxing_average,\
           my_f1_average,\
           my_kappa,\
           my_confusion_matrix




def cal_evaluation_mutil_class(num_classes, shiyan_type, data_type, save_line_2, save_line_3, save_line_4):
    #
    plt.figure()
    class_precision_list = []
    class_recall_list = []
    class_specificity_list = []
    class_yingxing_list = []
    fpr_list = []
    tpr_list = []
    auc_list = []
    weight_list = []  # 权重
    for class_id in range(num_classes):
        weight_list.append(save_line_3.count(class_id))
        class_pre = [0 if i == class_id else 1 for i in save_line_2]
        class_org = [0 if i == class_id else 1 for i in save_line_3]
        class_matrix = confusion_matrix(class_org, class_pre)
        TP, FN, FP, TN = class_matrix[0, 0], class_matrix[0, 1], class_matrix[1, 0], class_matrix[1, 1]

        if TP + FP == 0:
            class_precision = 0.0
        else:
            class_precision = 1.0 * TP / (TP + FP)
        if TP + FN == 0:
            class_recall = 0.0
        else:
            class_recall = 1.0 * TP / (TP + FN)
        if FP + TN == 0:
            class_specificity = 0.0
        else:
            class_specificity = 1.0 * TN / (FP + TN)
        if TN + FN == 0:
            class_yingxing = 0.0
        else:
            class_yingxing = 1.0 * TN / (TN + FN)

        class_precision_list.append(class_precision)
        class_recall_list.append(class_recall)
        class_specificity_list.append(class_specificity)
        class_yingxing_list.append(class_yingxing)
        class_pre_scores = [1 - i[class_id] for i in save_line_4]
        auc = roc_auc_score(class_org, class_pre_scores)
        fpr, tpr, thresholds = roc_curve(class_org, class_pre_scores)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)

        colors = ['purple', 'g', 'darkorange', 'y', 'r']
        if shiyan_type == 'nang_pao_feibao_3':
            labels = ['CE', 'AE', 'Non-HE FLL']
        elif shiyan_type == 'pao_liangfeibao_efeibao_3':
            colors = ['g', 'darkorange', 'y']
            labels = ['AE', 'Benign non-HE FLL', 'Malignant FLL']
        elif shiyan_type == 'nang_pao_liangfeibao_efeibao_4':
            labels = ['CE', 'AE', 'Benign non-HE FLL', 'Malignant FLL']
        elif shiyan_type == 'pao_nang1_2_lfeibao_efeibao_5':
            labels = ['AE', 'CE1', 'CE2','Benign non-HE FLL', 'Malignant FLL']

        auc_print = '0·' + (str(round(auc, 3))).split('.')[-1]
        plt.plot(fpr, tpr, color=colors[class_id], lw=2, linestyle='-', label=labels[class_id] + ',AUC=' + auc_print)
        # plt.plot(fpr, tpr, color=colors[class_id], lw=2, linestyle='-')


    fpr_average = np.unique(np.concatenate([fpr_list[i] for i in range(num_classes)]))
    tpr_average = np.zeros_like(fpr_average)
    for i in range(num_classes):
        tpr_average += scipy.interp(fpr_average, fpr_list[i], tpr_list[i])
    tpr_average = tpr_average / num_classes
    auc_average = sklearn.metrics.auc(fpr_average, tpr_average)

    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    auc_print = '0·' + (str(round(auc_average, 3))).split('.')[-1]
    plt.plot(fpr_average, tpr_average, color='b', lw=2,label='Macro average,AUC=' + auc_print)
    plt.legend(loc="lower right")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().xaxis.set_major_formatter(
        plt.FixedFormatter(['0·0', '0·2', '0·4', '0·6', '0·8', '1·0']))
    plt.gca().yaxis.set_major_formatter(
        plt.FixedFormatter(['0·0', '0·2', '0·4', '0·6', '0·8', '1·0']))


    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt_save_name = './out/yes_wbl_caijian_adam_small/' + \
                    'yes_wbl' + '_' + \
                    shiyan_type + '_' \
                    + data_type + \
                    '_ROC曲线.svg'
    plt.savefig(plt_save_name, dpi=300)
    plt.close()

    # 以病人id， 计算相关指标
    my_precision, my_recall, my_f1, _ = precision_recall_fscore_support(save_line_3, save_line_2, average=None)  # 可以作为对比结果
    my_confusion_matrix = confusion_matrix(save_line_3, save_line_2)
    my_accuracy = accuracy_score(save_line_3, save_line_2)
    my_kappa = cohen_kappa_score(save_line_3, save_line_2)
    weight_list = [x / sum(weight_list) for x in weight_list]
    my_precision_average = sum(np.multiply(np.array(class_precision_list), np.array(weight_list)))
    my_recall_average = sum(np.multiply(np.array(class_recall_list), np.array(weight_list)))
    my_specificity_average = sum(np.multiply(np.array(class_specificity_list), np.array(weight_list)))
    my_yingxing_average = sum(np.multiply(np.array(class_yingxing_list), np.array(weight_list)))
    my_f1_average = sum(np.multiply(np.array(my_f1), np.array(weight_list)))

    return fpr,tpr,auc,  \
           fpr_average, tpr_average, auc_average, \
           class_precision_list,class_recall_list,class_specificity_list,class_yingxing_list,my_f1,auc_list, \
           my_accuracy, my_precision_average,my_recall_average,my_specificity_average,my_yingxing_average,my_f1_average,my_kappa,auc_average,my_confusion_matrix



def save_evaluation_in_sheet_two_class(sheet_w, data_type, data_name, shiyan_name,
                             my_accuracy, my_precision_average, my_recall_average,
                             my_specificity_average,my_yingxing_average,my_f1_average,my_kappa,auc,my_confusion_matrix,style):
    # 只写一次，不然报错
    if data_type == 'val':
        sheet_w.write(0, 0, '实验类型')
        sheet_w.write(0, 1, '数据类型')
        sheet_w.write(0, 2, 'accuracy')
        sheet_w.write(0, 3, 'precision')
        sheet_w.write(0, 4, 'recall')
        sheet_w.write(0, 5, 'specificity')
        sheet_w.write(0, 6, '阴性预测率')
        sheet_w.write(0, 7, 'F1-score')
        sheet_w.write(0, 8, 'kappa系数')
        sheet_w.write(0, 9, 'AUC值')
        sheet_w.write(0, 10, 'confusion_matrix')
        sheet_w.write(1, 0, shiyan_name)  # 写入实验名称
    if data_type == 'val':
        data_index = 1
    elif data_type == 'test_waibu':
        data_index = 2
    elif data_type == 'test_neibu':
        data_index = 3
    elif data_type == 'test_qianzhan':
        data_index = 4
    sheet_w.write(data_index, 1, data_name, style)
    sheet_w.write(data_index, 2, round(my_accuracy, 3), style)
    sheet_w.write(data_index, 3, round(my_precision_average, 3), style)
    sheet_w.write(data_index, 4, round(my_recall_average, 3), style)
    sheet_w.write(data_index, 5, round(my_specificity_average, 3), style)
    sheet_w.write(data_index, 6, round(my_yingxing_average, 3), style)
    sheet_w.write(data_index, 7, round(my_f1_average, 3), style)
    sheet_w.write(data_index, 8, round(my_kappa, 3), style)
    sheet_w.write(data_index, 9, round(auc, 3), style)
    sheet_w.write(data_index, 10, str(my_confusion_matrix), style)


def save_evaluation_in_sheet_mutil_class(sheet_w, data_type, data_name,shiyan_name,
                                         my_accuracy,my_precision_average,my_recall_average,my_specificity_average,
                                         my_yingxing_average,my_f1_average,my_kappa,auc_average,my_confusion_matrix,
                                         class_precision_list,class_recall_list,class_specificity_list,class_yingxing_list,my_f1,auc_list,style
                                         ):
    if data_type == 'val':  #
        sheet_w.write(0, 0, '实验类型')
        sheet_w.write(0, 1, '数据类型')
        sheet_w.write(0, 2, 'accuracy')
        sheet_w.write(0, 3, 'precision(single/average)')
        sheet_w.write(0, 4, 'recall(single/average)')
        sheet_w.write(0, 5, 'specificity(single/average)')
        sheet_w.write(0, 6, '阴性预测率(single/average)')
        sheet_w.write(0, 7, 'F1-score(single/average)')
        sheet_w.write(0, 8, 'kappa系数')
        sheet_w.write(0, 9, 'AUC值(single/average)')
        sheet_w.write(0, 10, 'confusion_matrix')
        sheet_w.write(1, 0, shiyan_name)  # 写入实验名称

    if data_type == 'val':
        data_index = 1
    elif data_type == 'test_waibu':
        data_index = 2
    elif data_type == 'test_neibu':
        data_index = 3
    elif data_type == 'test_qianzhan':
        data_index = 4
    sheet_w.write(data_index, 1, data_name, style)
    sheet_w.write(data_index, 2, round(my_accuracy, 3), style)

    sheet_w.write(data_index, 3, str([[round(my_precision_first, 3) for my_precision_first in class_precision_list],
                                      round(my_precision_average, 3)]), style)
    sheet_w.write(data_index, 4, str(
        [[round(my_recall_first, 3) for my_recall_first in class_recall_list], round(my_recall_average, 3)]), style)
    sheet_w.write(data_index, 5, str(
        [[round(my_specificity_first, 3) for my_specificity_first in class_specificity_list],
         round(my_specificity_average, 3)]), style)
    sheet_w.write(data_index, 6, str(
        [[round(my_yingxing_first, 3) for my_yingxing_first in class_yingxing_list], round(my_yingxing_average, 3)]),
                  style)
    sheet_w.write(data_index, 7, str([[round(my_f1_first, 3) for my_f1_first in my_f1], round(my_f1_average, 3)]),
                  style)
    sheet_w.write(data_index, 8, round(my_kappa, 3), style)
    sheet_w.write(data_index, 9, str([[round(auc_first, 3) for auc_first in auc_list], round(auc_average, 3)]), style)
    sheet_w.write(data_index, 10, str(my_confusion_matrix), style)


#
def calculate_confusion_matrix(pred, target, num_classes):
    pred_label = np.argmax(pred, axis=1)
    assert (pred_label.shape) == (target.shape)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
    with torch.no_grad():
        for t, p in zip(target, pred_label):
            confusion_matrix[t, p] += 1
    return confusion_matrix




# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():

    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    #
    shiyan_type_list = ['baochong_feibao_2',
                   'pao_feibao_2',
                   'nang_pao_feibao_3',
                   'pao_liangfeibao_efeibao_3',
                   'nang_pao_liangfeibao_efeibao_4',
                   'nang_nang_2',
                   'pao_nang1_2_lfeibao_efeibao_5']

    data_type_list =['val',
                   'test_neibu']


    #
    for shiyan_type in shiyan_type_list:

        print(shiyan_type)

        # save
        import xlwt
        excel_w = xlwt.Workbook()
        style = xlwt.XFStyle() # font
        font = xlwt.Font()
        font.colour_index = 4
        style.font = font
        sheet_evaluation = excel_w.add_sheet(shiyan_type)  # 评价指标的表格

        for data_type in data_type_list:

            print(data_type)

            sheet_pre = excel_w.add_sheet(data_type)  # 概率值的表格

            # shiyan_type
            if shiyan_type == 'baochong_feibao_2':
                shiyan_name = '包虫 非包虫二分类'
                lable_list = ['包虫', '非包虫']
                num_classes = 2
            elif shiyan_type == 'pao_feibao_2':
                shiyan_name = '泡包 非包虫二分类'
                lable_list = ['泡包虫', '非包虫']
                num_classes = 2
            elif shiyan_type == 'nang_pao_feibao_3':
                shiyan_name = '囊包 泡包 非包虫三分类'
                lable_list = ['囊包虫', '泡包虫', '非包虫']
                num_classes = 3
            elif shiyan_type == 'pao_liangfeibao_efeibao_3':
                shiyan_name = '泡包 良性非包 恶性非包虫三分类'
                lable_list = ['泡包虫', '良性非包虫', '恶性非包虫']
                num_classes = 3
            elif shiyan_type == 'nang_pao_liangfeibao_efeibao_4':
                shiyan_name = '囊包 泡包 良性非包 恶性非包虫四分类'
                lable_list = ['囊包虫', '泡包虫', '良性非包虫', '恶性非包虫']
                num_classes = 4
            elif shiyan_type == 'nang_nang_2':
                shiyan_name = '囊包1 囊包2二分类'
                lable_list = ['囊包虫1', '囊包虫2']
                num_classes = 2
            elif shiyan_type == 'pao_nang1_2_lfeibao_efeibao_5':
                shiyan_name = '泡包 囊包1 囊包2 良性非包 恶性非包虫五分类'
                lable_list = ['泡包虫', '囊包虫1', '囊包虫2', '良性非包虫', '恶性非包虫']
                num_classes = 5

            # data_type
            if data_type == 'val':
                data_name = '验证集'
                cfg_data_dir = cfg.data.val
                org_patience_dir = './data/val/patience'
            elif data_type == 'test_waibu':
                data_name = '测试集_外部'
                cfg_data_dir = cfg.data.test_waibu
                org_patience_dir = ''
            elif data_type == 'test_neibu':
                data_name = '测试集_内部'
                cfg_data_dir = cfg.data.test_neibu
                org_patience_dir = './data/test/patience'
            elif data_type == 'test_qianzhan':
                data_name = '测试集_前瞻'
                cfg_data_dir = cfg.data.test_qianzhan
                org_patience_dir = ''


            ################################## begin
            # build the dataloader
            dataset = build_dataset(cfg_data_dir)
            data_loader = build_dataloader(
                dataset,
                samples_per_gpu=cfg.data.samples_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False,
                round_up=False)
            # build the model and load checkpoint
            model = build_classifier(cfg.model)
            fp16_cfg = cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint['meta']:
                CLASSES = checkpoint['meta']['CLASSES']
            else:
                from mmcls.datasets import ImageNet
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use imagenet by default.')
                CLASSES = ImageNet.CLASSES
            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                model.CLASSES = CLASSES
                show_kwargs = {} if args.show_options is None else args.show_options
                # hook
                features_blobs0 = []
                def hook_feature0(module, input, output):
                    features_blobs0.append(output.data.cpu().numpy())
                    # features_blobs0.append(output)
                modules_out10 = model._modules.get('module')
                modules_out20 = modules_out10._modules.get('head')
                modules_out30 = modules_out20._modules.get('fc')
                modules_out30.register_forward_hook(hook_feature0)
                # hook over
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                          **show_kwargs)
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                         args.gpu_collect)
            rank, _ = get_dist_info()
            if rank == 0:
                if args.metrics:
                    results = dataset.evaluate(outputs, args.metrics,
                                               args.metric_options)
                    for k, v in results.items():
                        if k != 'gt_label':
                            print(f'\n{k} : {v:.2f}')
                else:
                    warnings.warn('Evaluation metrics are not specified.')
                    scores = np.vstack(outputs)
                    pred_score = np.max(scores, axis=1)
                    pred_label = np.argmax(scores, axis=1)
                    pred_class = [CLASSES[lb] for lb in pred_label]
                    results = {
                        'pred_score': pred_score,
                        'pred_label': pred_label,
                    }
                    if not args.out:
                        print('\nthe predicted result for the first element is '
                              f'pred_score = {pred_score[0]:.2f}, '
                              f'pred_label = {pred_label[0]} '
                              'Specify --out to save all results to files.')


                import time
                start_time = time.time()

                # 导入数据
                pic_id_list = []
                for i, data in enumerate(data_loader):
                    img_metas = data['img_metas'].data[0]
                    for img_meta in img_metas:
                        ori_filename = img_meta['ori_filename']
                        pic_id_list.append(ori_filename)

                # 推理结果 vs 标签
                features_blobs0 = np.vstack(features_blobs0)
                true_label_list = results['gt_label']

                # 后处理
                true_label_list_re, scores, pred_label, pic_id_list_re = AIresult_postpress(shiyan_type, true_label_list, features_blobs0, pic_id_list)

                # 计算几个line
                save_line_1, save_line_2, save_line_3, save_line_4 =\
                    cal_AIresult_by_shiyan_data_type(num_classes,  scores,  pred_label, true_label_list_re, org_patience_dir, pic_id_list_re)

                end_time = time.time()
                spend_time = (end_time - start_time)/60
                print('spend_time:{}'.format(spend_time))

                # creat sheet， 保存概率值
                save_data_in_sheet(sheet_pre, num_classes, lable_list, save_line_1,
                                   save_line_3, save_line_4)

                # 计算评价指标,分两种情况，一个是二分类，一个是三或者四分类
                if shiyan_type in ['baochong_feibao_2','pao_feibao_2','nang_nang_2']:
                    # 二分类
                    fpr, tpr, auc, \
                    my_accuracy, my_precision_average, my_recall_average, \
                    my_specificity_average, my_yingxing_average, my_f1_average, \
                    my_kappa, my_confusion_matrix = cal_evaluation_two_class(shiyan_type, data_type,save_line_2, save_line_3, save_line_4)

                    save_evaluation_in_sheet_two_class(sheet_evaluation, data_type, data_name, shiyan_name,
                                             my_accuracy, my_precision_average, my_recall_average,
                                             my_specificity_average, my_yingxing_average, my_f1_average, my_kappa, auc,
                                             my_confusion_matrix, style)
                else:
                    # 三或者四分类
                    fpr, tpr, auc, \
                    fpr_average, tpr_average, auc_average, \
                    class_precision_list,class_recall_list,class_specificity_list,class_yingxing_list,my_f1,auc_list,\
                    my_accuracy, my_precision_average, my_recall_average, my_specificity_average, my_yingxing_average, \
                    my_f1_average, my_kappa, auc_average, my_confusion_matrix = cal_evaluation_mutil_class(num_classes, shiyan_type, data_type, save_line_2, save_line_3, save_line_4)

                    save_evaluation_in_sheet_mutil_class(sheet_evaluation, data_type, data_name, shiyan_name,
                                                         my_accuracy, my_precision_average, my_recall_average,
                                                         my_specificity_average,
                                                         my_yingxing_average, my_f1_average, my_kappa, auc_average,
                                                         my_confusion_matrix,
                                                         class_precision_list, class_recall_list,
                                                         class_specificity_list, class_yingxing_list, my_f1, auc_list,
                                                         style)

        excel_save_name = './out/yes_wbl_caijian_adam_small/' + shiyan_name + '.xlsx'
        excel_w.save(excel_save_name)  # 保存
        print('ok')


    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)



if __name__ == '__main__':
    main()

