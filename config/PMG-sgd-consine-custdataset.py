exp_name = 'PMG-sgd-consine-custdataset-500-epochs'
batch_size = 16
max_epochs = 300
save_interval = 25
val_interval = 1
save_dir = 'Tsinghua-dogs/saved_weights/'
work_dir = 'Tsinghua-dogs/runs/'
num_classes=130

train_dataset = dict(
    type = 'TsinghuaDogs',
    shuffle = True,
    split='train',
    num_workers = 8,
    batch_size = 16
)

transforms_train = [
    'HorizontalFlip',
    dict(
        type='Resize',
        shape = 550
    ),
    dict(
        type='Crop',
        size=448
    )
]

transforms_val = [
    dict(
        type='Resize',
        shape = 550
    ),
    dict(
        type='Crop',
        size=448
    )
]

val_dataset = dict(
    type = 'TsinghuaDogs',
    shuffle = True,
    split='validation',
    num_workers = 8,
    batch_size = 16
)

model = dict(
    type = 'PMG',
    model = dict(type='Rnet50', pretrained=True),
    feature_size=512,
    num_classes=num_classes
)
optimizer = dict(
    type = 'SGD',
    lr = 0.002,
    momentum = 0.9,
    weight_decay = 5e-4
)
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = max_epochs,
    T_mult = 1
)