import shutil 
import numpy as np 
import urllib.request 
import os
from tqdm import tqdm
from PIL import Image, ImageOps
import random
from jittor.dataset import Dataset
import xml.etree.ElementTree as ET
import jittor as jt


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)



def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def create_directories(source, target):
    for d in tqdm(os.scandir(source)):
        if d.is_dir():
            os.system('mkdir -p "{}"'.format(target + d.name))

def fetch(image_path, annotation_path):
    im = Image.open(image_path)
    name = get_name(annotation_path)

    return im, name

def get_name(path):
    tree = ET.parse(path)
    root = tree.getroot()
    obj = root.find('object')
    return obj.find('name').text


def horizontal_flip(im):
    if random.random() < 0.5:
        return im.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return im
def pad(image):
    w,h = image.size
    crop_size = 513
    pad_h = max(crop_size - h, 0)
    pad_w = max(crop_size - w, 0)
    image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
    return image


def crop(image):
    w, h = image.size
    ratio = random.randrange(50, 100, 1)
    crop_w = int(w * (ratio / 100.))
    crop_h = int(h * (ratio / 100.))
    x1 = random.randint(0, w - crop_w)
    y1 = random.randint(0, h - crop_h)
    image = image.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    return image

def scale(image):
    SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    ratio_w = np.random.choice(SCALES)
    ratio_h = np.random.choice(SCALES)
    w, h= image.size
    nw = (int)(w*ratio_w)
    nh = (int)(h*ratio_h)
    image = image.resize((nw, nh), Image.BILINEAR)
    return image     


def get_body_size(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    obj = root.find('object')
    body_bnd_box = obj.find('bodybndbox')
    return tuple([int(body_bnd_box[i].text) for i in range(4)])

def normalize(image):
    mean = (0.485, 0.456, 0.40)
    std = (0.229, 0.224, 0.225)
    np_im = np.array(image).astype(np.float32)

    np_im /= 255.0
    np_im -= mean
    np_im /= std
    return np_im


class Basicdataset(Dataset):
    def __init__(self, split='train', data_root = '/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/', batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.image_root = os.path.join(data_root, 'images')
        self.label_root = os.path.join(data_root, 'annotations')

        self.data_list_path = os.path.join(data_root, 'datalist/' + self.split + '.lst')
        self.image_path = []
        self.anno_path = []

        with open(self.data_list_path, "r") as f:
            lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            file_path = line[[x.isdigit() for x in line].index(True):]
            _img_path = os.path.join(self.image_root, file_path)
            _label_path = os.path.join(self.label_root, file_path + '.xml')
            assert os.path.isfile(_img_path)
            assert os.path.isfile(_label_path)
            self.image_path.append(_img_path)
            self.anno_path.append(_label_path)
        self.total_len = len(self.image_path)
        self.set_attrs(batch_size = self.batch_size, total_len = self.total_len, shuffle = self.shuffle) # bs , total_len, shuffle
    
    def __getitem__(self, image_id):
        return NotImplementedError

class Traindataset(Basicdataset):
    def __init__(self, data_root='/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/', split='train', batch_size=1, shuffle=False):
        super(Traindataset, self).__init__(split,data_root, batch_size, shuffle)

    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        anno_path = self.anno_path[image_id]
        image, label = fetch(image_path, anno_path)
        image = image.resize((512, 512), Image.BILINEAR)
        image = scale(image)
        image = crop(image)
        image = pad(image)
        image = horizontal_flip(image)
        image = image.resize((224, 224), Image.BILINEAR)
        image = normalize(image)
        image = np.array(image).astype(np.float32).transpose(2, 0, 1)
        
        return image, label

def orgainize_data(dest_path):
    image_dir = dest_path + 'tsinghua-dogs-data/low-resolution/'
    annotation_dir = dest_path + 'tsinghua-dogs-data/Low-Annotations/'
    train_val_split_dir= dest_path + 'tsinghua-dogs-data/TrainAndValList/'

    image_dest_dir_train = dest_path + 'processed_tsinghuadogs/train/'
    image_dest_dir_val = dest_path + 'processed_tsinghuadogs/val/'

    create_directories(image_dir, image_dest_dir_train)
    create_directories(image_dir, image_dest_dir_val)

    with open(train_val_split_dir + 'train.lst') as f:
        lineList = f.readlines()
    for line in tqdm(lineList):
        file_path = line[[x.isdigit() for x in line].index(True):-1]
        image_path = os.path.join(image_dir, file_path)
        annotation_path = os.path.join(annotation_dir, file_path)+ '.xml'
        im = process_image(image_path, annotation_path)
        im.save(os.path.join(image_dest_dir_train, file_path))

    with open(train_val_split_dir + 'validation.lst') as f:
        lineList = f.readlines()
    for line in tqdm(lineList):
        file_path = line[[x.isdigit() for x in line].index(True):-1]
        image_path = os.path.join(image_dir, file_path)
        annotation_path = os.path.join(annotation_dir, file_path)+ '.xml'
        im = process_image(image_path, annotation_path)
        im.save(os.path.join(image_dest_dir_val, file_path))

        

def unzip_files(dest_path, filenames):
    print('-------unpacking train validation lists-------')
    shutil.unpack_archive('TrainValSplit.zip', dest_path + 'tsinghua-dogs-data')
    print('-------unpacking low res images-------')
    shutil.unpack_archive(filenames[0], dest_path + 'tsinghua-dogs-data')
    print('-------unpacking low res annotatios-------')
    shutil.unpack_archive(filenames[1], dest_path + 'tsinghua-dogs-data')
    
def download_files():
    #resources=[images, labels]
    resources=[
        'https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1', 
        'https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip'
        ]

    test_val_split='https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip'

    download_url(test_val_split, 'TrainValSplit.zip')
    download_url(resources[0], 'low-res-images.zip')
    download_url(resources[1], 'low-res-annotations.zip')
