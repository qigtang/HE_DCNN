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

import cv2
from torch.nn import functional as F


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

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    # dataset = build_dataset(cfg.data.val)
    dataset = build_dataset(cfg.data.test_neibu)


    data_loader = build_dataloader(
        dataset,
        samples_per_gpu= 1,
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


        def returnCAM(feature_conv, weight_softmax, class_idx):
            # generate the class activation maps upsample to 256x256
            size_upsample = (256, 256)
            bz, nc, h, w = feature_conv.shape
            output_cam = []
            for idx in class_idx:
                cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
                cam = cam.reshape(h, w)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                output_cam.append(cv2.resize(cam_img, size_upsample))
            return output_cam


        model.eval()
        # print(model)

        # 在forward之前注册一个hook， 该hook在forward执行以后自动被执行
        features_blobs = []
        def hook_feature(module, input, output):
            features_blobs.append(output.data.cpu().numpy())
        modules_out1 = model._modules.get('module')
        modules_out2 = modules_out1._modules.get('backbone')
        modules_out3 = modules_out2._modules.get('layer4')
        modules_out3.register_forward_hook(hook_feature)

        features_blobs0 = []
        def hook_feature0(module, input, output):
            features_blobs0.append(output)

        modules_out10 = model._modules.get('module')
        modules_out20 = modules_out10._modules.get('head')
        modules_out30 = modules_out20._modules.get('fc')
        modules_out30.register_forward_hook(hook_feature0)

        # get the softmax weight
        params = list(model.parameters())
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())


        for i, data in enumerate(data_loader):
            with torch.no_grad():
                result = model(return_loss=False, **data)


            import torch.nn.functional as F
            sf = F.softmax(features_blobs0[0])
            sf = sf.data.cpu().numpy()

            pred_score = np.max(result)
            pred_label = np.argmax(result)

            CAMs = returnCAM(features_blobs[-1], weight_softmax, [pred_label])

            img_name = data['img_metas']._data[0][0]['filename']
            img = cv2.imread(img_name)

            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5

            if i <= 100:
                cv2.imwrite('out/CAM/neibu/' + img_name.split('/')[-1], result)
                cv2.imwrite('out/CAM/neibu/' + (img_name.split('/')[-1]).split('.jpg')[0] + '_org.jpg', img)




if __name__ == '__main__':
    main()
