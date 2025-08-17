_base_ = [
    '../../../mmdetection/configs/_base_/default_runtime.py',
    '../../../mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

# VAL DATASET
data_root_val = '/var/storage/Common/SatelliteVehicles/Datasets/Real/Real-Utah_112px_0.125m_RndSmpl_Imgs:all_Anno:small-only/validation_subset025.0_seed0/'
ann_file_val = 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json'

## TEST DATASET
data_root_test  = '/var/storage/Common/SatelliteVehicles/Datasets/Real/Real-Utah_112px_0.125m_RndSmpl_Imgs:all_Anno:small-only/test/'
ann_file_test   = 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json'


train_batch_size_per_gpu = 24
val_batch_size_per_gpu = 12
test_batch_size_per_gpu = 60

num_workers = 8

max_epochs = 100


img_scale = (128, 128)

affine_scale = 0.9

class_name = ('small',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])



load_from = 'https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth'


# MODEL SETTINGS
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)

batch_augments = [
    dict(type='BatchFixedSizePad', size=img_scale, pad_mask=True)
]

model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=img_scale[0],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_rel_pos=True,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint=load_from,
        )
    ),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=num_classes
        ),
        mask_head=None
    )
)

custom_hooks = [dict(type='Fp16CompresssionHook')]


## 
dataset_type = 'CocoDataset'
backend_args = None

# Original
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(
#         type='LoadAnnotations', 
#         with_bbox=True, 
#         # with_mask=True
#         with_mask=False
#     ),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomResize',
#         scale=img_scale,
#         ratio_range=(0.1, 2.0),
#         keep_ratio=True),
#     dict(
#         type='RandomCrop',
#         crop_type='absolute_range',
#         crop_size=img_scale,
#         recompute_bbox=True,
#         allow_negative_crop=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
#     dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
#     dict(type='PackDetInputs')
# ]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False)
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



# TRAIN DATASET
data_root_A_train = '../Data/Synthetic/UGRC-with-cars/' # 
ann_file_A_train  = 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-FasterRCNN-SynUGRC-STACKDAAMHeatMaps-Clf-Refine.json'

data_root_B_train = '../Data/Synthetic/UGRC-without-cars/' #
ann_file_B_train = 'annotations_coco_Empty.json'


dataset_A_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root_A_train,
            ann_file=ann_file_A_train,
            data_prefix=dict(img='images/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=pre_transform
        ),
        pipeline=train_pipeline
    )
)

dataset_B_train = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root_B_train,
            ann_file=ann_file_B_train,
            data_prefix=dict(img='images/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=pre_transform
        ),
        pipeline=train_pipeline
    )
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        # _delete_=True, # Deletes the settings inherited from the _base_ configuration
        type='ConcatDataset',
        datasets=[dataset_A_train, dataset_B_train],
        # separate_eval=False
    )
)


test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(
        type='LoadAnnotations', 
        with_bbox=True,
        # with_mask=True
        with_mask=False
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file=ann_file_val,
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_test,
        ann_file=ann_file_test,
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_val + ann_file_val,
    metric='bbox',
    format_only=False)
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_test + ann_file_test,
    metric='bbox',
    format_only=False
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.7,
        'decay_type': 'layer_wise',
        'num_layers': 12,
    },
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        # lr=0.01,
        # lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# max_iters = 184375
# interval = 5000
max_iters = 100000

# interval = 2000
interval = 1000

dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001,
        by_epoch=False, 
        begin=0, 
        end=250
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        
        end=max_iters,
        # end=max_epochs,
        
        by_epoch=False,
        # by_epoch=True,
        
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        # milestones=[20, 29],
        # milestones=[5000, 6000],
        milestones=[1000, 2000],
        gamma=0.1
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals
)
# train_cfg = dict(
#     type='EpochBasedTrainLoop', 
#     max_epochs=max_epochs, 
#     val_interval=1
# )

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(
        type='LoggerHook', 
        interval=50,
        log_metric_by_epoch=False
    ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        # by_epoch=True,
        save_last=True,
        # interval=1,
        interval=interval,
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50'],
        max_keep_ckpts=2
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
)

log_processor = dict(
    type='LogProcessor', 
    window_size=50, 
    by_epoch=False
    # by_epoch=True
)

auto_scale_lr = dict(base_batch_size=64)
work_dir = '../work_dirs/vitdet/LINZ2UGRC/vitdet_train_syn_ugrc_test_real_ugrc'


