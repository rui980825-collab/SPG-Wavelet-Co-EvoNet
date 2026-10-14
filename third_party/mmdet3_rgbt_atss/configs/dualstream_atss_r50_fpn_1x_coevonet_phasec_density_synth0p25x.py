_base_ = ["./dualstream_atss_r50_fpn_1x_coevonet_uavrgbt.py"]

dataset_type = "CoEvoNetUAVRGBTDataset"
classes = ("ship", "car", "cyclist", "pedestrian", "bus", "drone", "plane")
img_mean = [123.675, 116.28, 103.53]
img_std = [58.395, 57.12, 57.375]

train_pipeline = [
    dict(type="LoadRGBTConcatenatedImageFromFile", to_float32=True, to_rgb=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=(640, 512), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="NormalizeConcatenatedRGBT", mean=img_mean + img_mean, std=img_std + img_std),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadRGBTConcatenatedImageFromFile", to_float32=True, to_rgb=True),
    dict(type="Resize", scale=(640, 512), keep_ratio=True),
    dict(type="NormalizeConcatenatedRGBT", mean=img_mean + img_mean, std=img_std + img_std),
    dict(type="PackDetInputs"),
]

real_train_dataset = dict(
    type=dataset_type,
    ann_file="data/raw/dataset/annotations_coco/instances_00_train2017.json",
    data_prefix=dict(img="data/raw/dataset/images"),
    filename_replacements=[("/00/", "/01/"), ("\\\\00\\\\", "\\\\01\\\\")],
    filter_cfg=dict(filter_empty_gt=True, min_size=1),
    pipeline=train_pipeline,
    metainfo=dict(classes=classes),
)

synth_train_dataset = dict(
    type=dataset_type,
    ann_file="data/synth/uav_rgbir/v25lite_phaseC_tau027_density_20k/annotations/instances_synth_realcat.json",
    data_prefix=dict(img="data/synth/uav_rgbir/v25lite_phaseC_tau027_density_20k/images/rgb"),
    tir_data_prefix="data/synth/uav_rgbir/v25lite_phaseC_tau027_density_20k/images/tir",
    filter_cfg=dict(filter_empty_gt=True, min_size=1),
    pipeline=train_pipeline,
    metainfo=dict(classes=classes),
)

train_dataloader = dict(
    batch_size=1,
    sampler=dict(
        type="SourceAwareSampler",
        shuffle=True,
        real_ratio=4,
        synth_ratio=1,
        deterministic_cycle=True,
    ),
    dataset=dict(
        _delete_=True,
        type="ConcatDataset",
        datasets=[real_train_dataset, synth_train_dataset],
        ignore_keys=["dataset"],
    ),
)

val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        ann_file="data/raw/dataset/annotations_coco/instances_00_test2017.json",
        data_prefix=dict(img="data/raw/dataset/images"),
        filename_replacements=[("/00/", "/01/"), ("\\\\00\\\\", "\\\\01\\\\")],
        test_mode=True,
        pipeline=test_pipeline,
        metainfo=dict(classes=classes),
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(ann_file="data/raw/dataset/annotations_coco/instances_00_test2017.json")
test_evaluator = val_evaluator

work_dir = "runs/mmdet3_rgbt_atss/phasec_density_synth0p25x"
