exp_name = 'PMG-sgd-consine-custdataset-200-epochs'
batch_size = 16
max_epoch = 200
save_interval = 1
val_interval = 1
save_dir = 'ThuDogs/saved_weights/'
work_dir = 'ThuDogs/runs/'
num_classes=130
lrs = dict(classifier_concat = 0.002, 
        conv_block1= 0.002,
        classifier1 = 0.002,
        conv_block2 = 0.002,
        classifier2 = 0.002,
        conv_block3 = 0.002,
        classifier3= 0.002,
        features = 0.0002)

train_dataset = dict(
    type = 'TsinghuaDogs',
    shuffle = True,
    split='train',
    num_workers = 8,
    batch_size = 16
)

train_transforms = [
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

val_transforms = [
    dict(
        type='Resize',
        shape = 550
    ),
    dict(
        type='Crop',
        random=False,
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
    T_max = max_epoch,
    T_mult = 1
)

num_chk_points = 2