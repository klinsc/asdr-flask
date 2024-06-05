import random

# Inherit and overwrite part of the config based on this config
_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

data_root = './data/rpod5/'  # dataset root

# dataset category name
class_name = ('11522_tx_dyn1',
              '11522_tx_ynyn0d1',
              '115_1way_ds_w_motor',
              '115_3ways_ds_w_motor',
              '115_breaker',
              '115_buffer',
              '115_cvt_1p',
              '115_cvt_3p',
              '115_ds',
              '115_gs',
              '115_gs_w_motor',
              '115_la',
              '115_vt_1p',
              '115_vt_3p',
              '22_breaker',
              '22_cap_bank',
              '22_ds',
              '22_ds_la_out',
              '22_ds_out',
              '22_gs',
              '22_ll',
              '22_vt_1p',
              '22_vt_3p',
              'BCU',
              'DIM',
              'DPM',
              'LL',
              'NGR',
              'NGR_future',
              'Q',
              'ground',
              'relays',
              'remote_io_module',
              'ss_auto_mode',
              'ss_man_mode',
              'sync_auto_25',
              'sync_manual_bus',
              'tele_protection',
              'terminator_double',
              'terminator_single',
              'terminator_w_future',
              'v_m',
              'v_m_digital')

num_classes = len(class_name)  # dataset category number

# metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
# palette is a display color for category at visualization
# The palette length must be greater than or equal to the length of the classes
# Random RGB color of format (R, G, B).
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]  # noqa
metainfo = dict(classes=class_name, palette=color)

# Adaptive anchor based on tools/analysis_tools/optimize_anchors.py
anchors = [
    [(68, 69), (154, 91), (143, 162)],  # P3/8
    [(242, 160), (189, 287), (391, 207)],  # P4/16
    [(353, 337), (539, 341), (443, 432)]  # P5/32
]

max_epochs = 2000 # use 1000 and got 0.3 loss
train_batch_size_per_gpu = 12
train_num_workers = 4

# load COCO pre-trained weight
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth'  # noqa

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='trainval.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of two weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),

    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),

    # The log printing interval is 5
    logger=dict(type='LoggerHook', interval=5))

# The evaluation interval is 10
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])  # noqa
