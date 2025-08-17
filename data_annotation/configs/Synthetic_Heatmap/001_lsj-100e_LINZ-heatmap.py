_base_ = [
    '../../../mmdetection/configs/_base_/default_runtime.py',
]

# dataset settings
dataset_type = 'CocoDataset'
# data_root = 'data/coco/'

# Dataset
# TRAIN DATASET
data_root_train = '../Data/Synthetic/LINZ-with-cars/'

# VAL DATASET
data_root_val   = '../Data/Synthetic/LINZ-with-cars/'

# TEST DATASET
data_root_test = '../Data/Synthetic/UGRC-with-cars/'

image_size = (128, 128)


class_name = ('small',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])


backend_args = None

train_batch_size_per_gpu = 48
test_batch_size_per_gpu = 48
num_workers = 4

max_epochs = 100


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations', 
        with_bbox=True, 
        with_mask=False
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
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

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_train,
        ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-ViTDet-SynLINZ-STACKDAAMHeatMaps-ConfThresh:0.60.json',
        data_prefix=dict(img='daam_stack_heatmaps/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        metainfo=metainfo,
    )
)

val_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_val,
        ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-ViTDet-SynLINZ-STACKDAAMHeatMaps-ConfThresh:0.60.json',
        data_prefix=dict(img='daam_stack_heatmaps/'),
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
        ann_file='annotations_coco_Empty.json',
        data_prefix=dict(img='daam_stack_heatmaps/'),
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=metainfo,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_val + 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-ViTDet-SynLINZ-STACKDAAMHeatMaps-ConfThresh:0.60.json',
    metric='bbox',
    format_only=False)
# test_evaluator = val_evaluator
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_test + 'annotations_coco_Empty.json',
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
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ))

# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# max_iters = 184375
# interval = 5000
max_iters = 50000

# interval = 2000
interval = 500

dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        
        end=max_iters,
        # end=max_epochs,
        
        by_epoch=False,
        # by_epoch=True,
        
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        milestones=[20, 29],
        gamma=0.1)
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
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        # by_epoch=True,
        save_last=True,
        # interval=1,
        interval=interval,
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50'],
        max_keep_ckpts=1))

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
