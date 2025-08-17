_base_ = '../../../mmyolo/configs/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75
last_stage_out_channels = 768

affine_scale = 0.9
mixup_prob = 0.1


img_scale = (128, 128) #_base_.img_scale
# img_scale = (640, 640) #_base_.img_scale
num_classes = 1
class_name = ('small',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

train_batch_size_per_gpu = 192
val_batch_size_per_gpu = 384
test_batch_size_per_gpu = 384

train_num_workers = 4
val_num_workers = 4
test_num_workers = 4

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.0001
lr_factor = 0.01  # Learning rate scaling factor
max_epochs = 500  # Maximum training epochs

# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

save_epoch_intervals = 2
max_keep_ckpts = 2

# validation intervals in stage 2
val_interval_stage2 = 1

# TRAIN DATASET
data_root_train = '../Data/Synthetic/LINZ-with-cars/'
ann_file_train = 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-FasterRCNN-SynLINZ-STACKDAAMHeatMaps-ConfThresh:0.60.json'

# VAL DATASET
data_root_val = '../Data/Synthetic/LINZ-with-cars/'
ann_file_val = 'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500_Pseudo-FasterRCNN-SynLINZ-STACKDAAMHeatMaps-ConfThresh:0.60.json'

# TEST DATASET
data_root_test  = '../Data/Synthetic/UGRC-with-cars/'
ann_file_test = 'annotations_coco_Empty.json'


load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth'


# =======================Unmodified in most cases==================
pre_transform = _base_.pre_transform
last_transform = _base_.last_transform

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels]
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels])
    ),
    train_cfg=dict(
        assigner=dict(
            num_classes=num_classes
        )
    )
)

mosaic_affine_transform = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

# enable mixup
train_pipeline = [
    *pre_transform, *mosaic_affine_transform,
    dict(
        type='YOLOv5MixUp',
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)
    ),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)
    ), 
    *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root_train,
        ann_file=ann_file_train,
        data_prefix=dict(img='daam_stack_heatmaps/'),
        filter_cfg=dict(filter_empty_gt=False),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)

# _base_.test_pipeline[1].img_scale = img_scale
# _base_.test_pipeline[2].scale = img_scale

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root_val,
        ann_file=ann_file_val,
        data_prefix=dict(img='daam_stack_heatmaps/'),
        metainfo=metainfo,
        # filter_cfg=dict(filter_empty_gt=False), # Does this make a change?
        filter_cfg=dict(filter_empty_gt=True), # Does this make a change?
        pipeline=test_pipeline,
    )
)

test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=test_num_workers,
    dataset=dict(
        data_root=data_root_test,
        ann_file=ann_file_test,
        data_prefix=dict(img='daam_stack_heatmaps/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),  # Does this make a change?
        pipeline=test_pipeline,
    )
)


optim_wrapper = dict(
    optimizer=dict(
        lr=base_lr,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
)


default_hooks = dict(
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=max_epochs
    ),
    checkpoint=dict(
        interval=save_epoch_intervals,
        max_keep_ckpts=max_keep_ckpts,
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50']
    )
)

_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs
_base_.custom_hooks[1].switch_pipeline = train_pipeline_stage2

val_evaluator = dict(
    ann_file=data_root_val + ann_file_val,
)

test_evaluator = dict(
    ann_file= data_root_test + ann_file_test,
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[
        ((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)
    ]
)


visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'), 
        dict(type='TensorboardVisBackend')
    ]
)

work_dir = '../work_dirs/yolov8/LINZ2UGRC/yolov8_train_syn_linz_hmap_test_syn_ugrc_hmap' # change here