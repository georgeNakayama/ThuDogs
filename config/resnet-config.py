exp_name = 0
batch_size = 128
max_epochs = 50
save_interval = 1
val_interval = 1
save_dir = 'Tsinghua-dogs/saved_weights/'
init_lr = 0.004
train_dir = '/mnt/disk/wang/THD-datasets/processed_tsinghuadogs/train'
val_dir = '/mnt/disk/wang/THD-datasets/processed_tsinghuadogs/val'

model = dict(
    type = 'Rnet50'
)
optimizer = dict(
    type = 'SGD',
    lr = 0.003,
    momentum = 0.9,
    weight_decay = 1e-4
)
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = 20500
)