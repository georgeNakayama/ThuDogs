import shutil 
import numpy as np 
import urllib.request 
import os
from tqdm import tqdm
from PIL import Image


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
        
        
def process_image(image_path, annotation_path):
    im = np.array(Image.open(image_path))
    xmin, ymin, xmax, ymax = get_body_size(annotation_path)
    im_trim = im[int(ymin): int(ymax),int(xmin): int(xmax)]
    pic = Image.fromarray(im_trim).convert('RGB')
    return pic


def get_body_size(annotation_path):
    import xml.etree.ElementTree as ET
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    obj = root.find('object')
    body_bnd_box = obj.find('bodybndbox')
    return body_bnd_box[0].text, body_bnd_box[1].text, body_bnd_box[2].text, body_bnd_box[3].text

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


def main():
    dest_path='/mnt/disk/wang/THD-datasets/'
    filenames=[
            'low-res-images.zip', 
            'low-res-annotations.zip'
        ]
    #download_files()
    #unzip_files(dest_path, filenames)
    orgainize_data(dest_path)

if __name__ == '__main__':
    main()
