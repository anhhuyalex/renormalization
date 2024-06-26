_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=512,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'VideoDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomGroupResize', size_min=256, size_max=320),
    dict(type='GroupRandomCrop', size=224),
    dict(type='GroupColorJitter'),
    dict(type='GroupRandomHorizontalFlip'),
    dict(type='NormalizedStack', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # total 32*8=256
    workers_per_gpu=16,
    drop_last=True,
    train=dict(
        type=dataset_type,
        root='/data5/chengxuz/Dataset/kinetics/comp_jpgs_extracted',
        metafile='/mnt/fs3/chengxuz/kinetics/pt_meta/train_frameno_new.txt',
        num_frames=1,
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnealing', min_lr=0.)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 1000
