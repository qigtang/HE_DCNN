
## Introduction
This repository contains source code for the paper "Robust ultrasound identification of hepatic echinococcosis and subtypes using a deep convolutional neural network model: a large-scale multicenter study".
HE_DCNN is designed for echinococcosis classification.The code architecture is from an open source image classification toolbox named mmclassification(https://github.com/open-mmlab/mmclassification)


## Citation
If you find it useful in your research, please consider citing: "Robust ultrasound identification of hepatic echinococcosis and subtypes using a deep convolutional neural network model: a large-scale multicenter study"


## Usage
Pytorch>=1.5.0
torchvision>=0.6.0
numpy>=1.18.1
pillow>=7.0.0
opencv>=4.5.1
scikit-learn>=0.23.1

## Installation
Please refer to [install.md](docs/install.md) for installation and dataset preparation.
In last, run shell "python setup.py develop" .


## Prepare datasets
The folder structure is as follws.
```
HE_DCNN
├── mmcls
├── tools
├── configs
├── data
│   ├── meta
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   ├── train
│   │   ├── 0
│   │   ├── 1
│   │   ├── 2
│   │   ├── 3
│   │   ├── 4
│   ├── train_pic
│   ├── val
│   │   │   ├── patience
│   │   │   ├── pic
│   ├── test
│   │   │   ├── patience
│   │   │   ├── pic
```

## Train a model
All outputs (log files and checkpoints) will be saved to the working directory, which is specified by `work_dir` in the config file.
By default we evaluate the model on the validation set after each epoch, you can change the evaluation interval by adding the interval argument in the training config.
If you use resnet50, you need write a resnet50-config in [configs] directory, we write an example shell named "resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py"
Notes: before training, you need to download the pretrained model checkpoints from "https://mmclassification.readthedocs.io/en/latest/model_zoo.html"
```shell
# python tools/train.py configs/resnet/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py
```


### Test a dataset
You can use the following commands to test a dataset.
```shell
# python tools/test.py configs/resnet/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py work_dirs/resnet50_b32x8_pao_nang2_jbl2_classs5/epoch_200.pth --metrics accuracy precision recall
```


### Show the CAM
```shell
# python tools/show_CAM.py configs/resnet/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py work_dirs/resnet50_b32x8_pao_nang2_jbl2_classs5/epoch_200.pth --metrics accuracy precision recall
```

