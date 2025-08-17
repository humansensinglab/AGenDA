_base_ = [
    '../../../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../../../mmdetection/configs/_base_/datasets/coco_detection.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_2x.py', 
    '../../../mmdetection/configs/_base_/default_runtime.py'
]



# TRAIN DATASET
data_root_train = '../Data/Real/LINZ/train'

# VAL DATASET
data_root_val   = '../Data/Real/LINZ/validation/'

# TEST DATASET
## LINZ
# '../Data/Real/LINZ/test/' # Use this for testing on the real LINZ test set with ground truth annotations
data_root_test  = '../Data/Synthetic/LINZ-with-cars/' # Use this for labeling synthetic LINZ test set without ground truth annotations


max_epochs = 1000 # 40
train_batch_size_per_gpu = 64
validation_batch_size_per_gpu = 64
test_batch_size_per_gpu = 64
num_workers = 8


class_name = ('small',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

img_scale = (128, 128)

affine_scale = 0.9

load_from = 'https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_2x_coco'


# model settings
model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))



dataset_type = 'CocoDataset'

backend_args = None

# Original
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=img_scale, keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

last_transform = [
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOXHSVRandomAug'), # ???
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction'
        )
    )
]

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
    ),
    dict(
        type='RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='MixUp',
        img_scale=img_scale,
    ),
    *last_transform
]


# Original
# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=num_workers,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root_train,
#         ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_BalancedRatio:0.2000.json',
#         data_prefix=dict(img='images/'),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=train_pipeline,
#         metainfo=metainfo,
#         backend_args=backend_args
#     )
# )

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root_train,
            ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
            # ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_BalancedRatio:0.2000.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            metainfo=metainfo,
            backend_args=backend_args,
            pipeline=pre_transform
        ),
        pipeline=train_pipeline,
    )
)



test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor'
        )
    )
]


val_dataloader = dict(
    batch_size=validation_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    )
)

test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_test,
        ann_file='annotations_coco_Empty.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
        backend_args=backend_args
    )
)

# test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_val + 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_test + 'annotations_coco_Empty.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

# test_evaluator = val_evaluator



# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.2,
        momentum=0.9,
        weight_decay=0.0001
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(enable=False, base_batch_size=train_batch_size_per_gpu)


default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1, 
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50']
    ),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    # param_scheduler=dict(
    #     max_epochs=max_epochs, 
    #     warmup_mim_iter=1000,
    #     lr_factor=lr_factor
    # ),
    logger=dict(type='LoggerHook', interval=5))

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')


work_dir = '../work_dirs/faster-rcnn/LINZ2UGRC/faster-rcnn_train_real_linz_test_real_linz'

