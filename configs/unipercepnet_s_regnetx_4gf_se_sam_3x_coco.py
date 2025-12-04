_base_ = [
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_3x.py', 
    './_base_/default_runtime.py',
]

import os
import sys

HOME_DIR = os.environ['HOME']
sys.path.append(os.path.join(HOME_DIR, 'dev/YOLOF-MaskV2-mmcv'))

custom_imports = dict(
    imports=['src'],
    allow_failed_imports=False)

model = dict(
    type='src.UniPercepNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='src.RegNet',
        arch='regnetx_4.0gf',
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        se_on=True,
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://regnetx_4.0gf')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=1360,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='YOLOFHead',
        num_classes=80,
        in_channels=512,
        reg_decoded_bbox=True,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[1, 2, 4, 8, 16],
            strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='DIoULoss', loss_weight=1.5)),
    roi_head=dict(
        type='src.StandardRoIHead',
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[32]),
        mask_head=dict(
            type='src.FCNMaskHead',
            num_convs=4,
            in_channels=512,
            conv_out_channels=256,
            num_classes=80,
            sam_on=True,
            upsample_cfg=dict(
                type='deconv', scale_factor=2),
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.2))),
    train_cfg=dict(
        bbox_head=dict(
            assigner=dict(
                type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        roi_head=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        bbox_head=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        roi_head=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=4)

lr = 1e-4
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.01))

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=1, save_best='coco/segm_mAP', rule='greater'))
