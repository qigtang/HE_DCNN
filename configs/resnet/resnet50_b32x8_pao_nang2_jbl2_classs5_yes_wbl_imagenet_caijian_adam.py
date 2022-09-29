

#  python tools/train.py  configs/resnet/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py

# python tools/test_pao_nang2_jbl2_classs5_yes_wbl_caijian_adam.py  configs/resnet/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_imagenet_caijian_adam.py work_dirs/resnet50_b32x8_pao_nang2_jbl2_classs5_yes_wbl_caijian_adam/200.pth --metrics accuracy precision recall --out out/yes_wbl_caijian/pao_nang2_jbl2_classs5_yes_wbl_caijian_adam.json




_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/default_runtime.py'
]


# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,  #todo:
        in_channels=2048,

        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 1),  #todo: topk < num_classes
    ))


# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[117.4, 117.4, 117.4],  #todo:
     std=[78.7, 78.6, 78.7],  #todo:
     to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),  #todo:
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),  #todo:
    dict(type='CenterCrop', crop_size=224),  #todo:
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]


data = dict(
    samples_per_gpu=5,
    workers_per_gpu=0,

    #
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='Filelist_gbc',
            data_prefix='data/train_pic',  # todo:
            ann_file='data/meta/train.txt',  # todo:
            pipeline=train_pipeline),
    ),

    #
    val=dict(
        type=dataset_type,
        data_prefix='data/val/pic',  #todo:
        ann_file='data/meta/val.txt',  #todo:
        pipeline=test_pipeline),

    #
    test_qianzhan=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='',
        pipeline=test_pipeline),

    test_waibu=dict(
        type=dataset_type,
        data_prefix='',
        ann_file='',
        pipeline=test_pipeline),

    test_neibu=dict(
        type=dataset_type,
        data_prefix='data/test/pic',
        ann_file='data/meta/test.txt',
        pipeline=test_pipeline)

)



evaluation = dict(interval=5, metric='accuracy')  # This evaluate the model per 5 epoch
# optimizer
optimizer = dict(type='Adam', lr=0.001) #todo:
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
# checkpoint saving
checkpoint_config = dict(interval=25)
# yapf:disable,  interval=100
log_config = dict(
    interval=80,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

resume_from = 'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
work_dir = 'work_dirs/resnet50_b32x8_pao_nang2_jbl2_classs5/'