_base_ = [
    '../../../mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    './001_lsj-100e_LINZ-heatmap.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)


image_size = (128, 128)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]



load_from = 'https://download.openmmlab.com/mmdetection/v3.0/vitdet/vitdet_mask-rcnn_vit-b-mae_lsj-100e/vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth'
# load_from=None
# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type='ViT',
        img_size=128,
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
            checkpoint=load_from
            # checkpoint='detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth'      
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
            num_classes=1
        ),
        mask_head=None
    )
)

custom_hooks = [dict(type='Fp16CompresssionHook')]
work_dir = '../work_dirs/vitdet/LINZ2UGRC/vitdet_train_syn_linz_hmap_test_syn_ugrc_hmap'
