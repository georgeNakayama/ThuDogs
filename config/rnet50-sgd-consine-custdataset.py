exp_name = 'rnet50-sgd-consine-custdataset-500-epochs'
batch_size = 64
max_epochs = 500
save_interval = 25
val_interval = 1
save_dir = 'Tsinghua-dogs/saved_weights/'
work_dir = 'Tsinghua-dogs/runs/'

train_dataset = dict(
    type = 'TsinghuaDogs',
    shuffle = True,
    split='train',
    num_workers = 8,
    batch_size = 64
)

transforms = [
    'Rotate',
    'HorizontalFlip',
    'ColorAugmentation',
    'VerticalFlip',
    'Blur',
    dict(
        type='Resize',
        shape = 224
    )
]

val_dataset = dict(
    type = 'TsinghuaDogs',
    shuffle = True,
    split='validation',
    num_workers = 8,
    batch_size = 2
)

model = dict(
    type = 'Rnet50'
)
optimizer = dict(
    type = 'SGD',
    lr = 0.005,
    momentum = 0.9,
    weight_decay = 1e-4
)
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = 10,
    T_mult = 2,
    start_warmup_lr = 1e-5
)