default_scope = "mmdet"

custom_imports = dict(
    imports=["rgbt_atss_runtime"],
    allow_failed_imports=False,
)

classes = ("ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane")

img_scale = (640, 512)
img_mean = [123.675, 116.28, 103.53]
img_std = [58.395, 57.12, 57.375]
pretrained_path = None

model = dict(
    type="ATSS",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="DualStreamResNet",
        out_channels=(256, 512, 1024, 2048),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        pretrained_path=pretrained_path,
        backbone_cfg=dict(
            type="ResNet",
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            style="pytorch",
            init_cfg=None,
        ),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="ATSSHead",
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
    ),
    train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=9), allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

train_pipeline = [
    dict(type="LoadRGBTConcatenatedImageFromFile", to_float32=True, to_rgb=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="NormalizeConcatenatedRGBT", mean=img_mean + img_mean, std=img_std + img_std),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadRGBTConcatenatedImageFromFile", to_float32=True, to_rgb=True),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="NormalizeConcatenatedRGBT", mean=img_mean + img_mean, std=img_std + img_std),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=None,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=None,
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    metric="bbox",
    proposal_nums=(100, 300, 1000),
)
test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=200),
    dict(type="MultiStepLR", begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="coco/bbox_mAP_50"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

env_cfg = dict(cudnn_benchmark=False)
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
launcher = "none"
