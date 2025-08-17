_base_ = '../../../mmyolo/configs/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco.py'
deepen_factor = 0.67
widen_factor = 0.75

# TRAIN DATASET
data_root_train = '../Data/Real/LINZ/train'

# VAL DATASET
data_root_val   = '../Data/Real/LINZ/validation/'

# TEST DATASET
# '../Data/Real/LINZ/test/' # Use this for testing on the real LINZ test set with ground truth annotations
data_root_test  = '../Data/Synthetic/LINZ-with-cars/' # Use this for labeling synthetic LINZ test set without ground truth annotations


class_name = ('small',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

img_scale = (128, 128)
# img_scale = (112, 112)

# Estimated with " python ./tools/analysis_tools/optimize_anchors.py --input-shape 128 128 --augment-args 0.1 1.9  --algorithm v5-k-means configs/..."
# anchors = [[(25, 32), (53, 69), (159, 220)], [(235, 166), (242, 242), (310, 337)], [(365, 375), (230, 681), (679, 324)]]
# anchors = [[(157, 155), (239, 133), (136, 238)], [(240, 165), (170, 237), (236, 191)], [(206, 240), (241, 217), (242, 242)]]
anchors = [[(31, 28), (32, 37), (27, 48)], [(48, 27), (47, 34), (34, 48)], [(41, 48), (49, 41), (48, 48)]]

max_epochs = 1000 # 40
train_batch_size_per_gpu = 200
validation_batch_size_per_gpu = 100
test_batch_size_per_gpu = 200 #768 #384
train_num_workers = 8

num_det_layers = 3

# Learning rate
base_lr   = 0.01 #0.01
lr_factor = 0.1

load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_m-v61_syncbn_fast_8xb16-300e_coco/yolov5_m-v61_syncbn_fast_8xb16-300e_coco_20220917_204944-516a710f.pth'

batch_shapes_cfg = dict(
    img_size=img_scale[0],
    batch_size=train_batch_size_per_gpu
)

pre_transform = _base_.pre_transform
affine_scale = _base_.affine_scale
mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform, 
    *mosaic_affine_pipeline,
    dict(
        type='YOLOv5MixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_pipeline]),
    dict(
        type='mmdet.Albu',
        transforms=_base_.albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

_base_.test_pipeline[next(i for i, v in enumerate(_base_.test_pipeline) if v.type=='YOLOv5KeepRatioResize')].scale = img_scale
_base_.test_pipeline[next(i for i, v in enumerate(_base_.test_pipeline) if v.type=='LetterResize')].scale = img_scale


model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
    ),
    bbox_head=dict(
        head_module=dict(
            widen_factor=widen_factor,
            num_classes=num_classes,
            featmap_strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            num_base_priors=3
        ),
        prior_generator=dict(
            base_sizes=anchors,
            strides=[
                8,
                16,
                32,
            ],
        ),
        loss_obj=dict(
            loss_weight=_base_.loss_obj_weight * ((img_scale[0] / 640)**2 * 3 / num_det_layers)
        ),
        loss_cls=dict(
            loss_weight=_base_.loss_cls_weight * (num_classes / 80 * 3 / num_det_layers)
        ),
        loss_bbox=dict(
            loss_weight=_base_.loss_bbox_weight * (3 / num_det_layers),
        ),
        obj_level_weights=[
            4.0,
            1.0,
            0.4,
        ],
    ),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
        multi_label=False,
    ),
    
)


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root_train,
            ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
            data_prefix=dict(img='images/'),
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False),
            pipeline=train_pipeline
        )
    )
)

val_dataloader = dict(
    batch_size=validation_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root_val,
        metainfo=metainfo,
        ann_file='annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
        data_prefix=dict(img='images/'),
        pipeline=_base_.test_pipeline
    )
)

test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root_test,
        metainfo=metainfo,
        ann_file='annotations_coco_Empty.json',
        data_prefix=dict(img='images/'),
        batch_shapes_cfg=batch_shapes_cfg,
        pipeline=_base_.test_pipeline
    )
)

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.optim_wrapper.optimizer.lr = base_lr

val_evaluator  = dict(
    ann_file=data_root_val+'annotations_coco_FakeBBoxes:42.36px_ForIoU:0.500.json',
)
test_evaluator = dict(
    ann_file=data_root_test+'annotations_coco_Empty.json',
)


default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1, 
        save_best=['coco/bbox_mAP', 'coco/bbox_mAP_50']
    ),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(
        max_epochs=max_epochs, 
        warmup_mim_iter=1000,
        lr_factor=lr_factor
    ),
    logger=dict(type='LoggerHook', interval=5))

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

work_dir = '../work_dirs/yolov5/LINZ2UGRC/yolov5_train_real_linz_test_real_linz'

