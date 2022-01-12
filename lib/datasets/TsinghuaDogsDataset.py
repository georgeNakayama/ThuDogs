import shutil 
import numpy as np 
import os
from PIL import Image, ImageDraw
from lib.datasets.annotation import Annotation
from lib.datasets.transforms import Compose, Normalize
from jittor.dataset import Dataset
from lib.utils import DATASETS

@DATASETS.register_module()
class TsinghuaDogs(Dataset):
    def __init__(self, transforms=None, dataset_dir='/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/', batch_size=128, shuffle=False, split='train', name=True, head=False, body=False, num_workers=16, drop_last=True, buffer_size=536870912):
        super().__init__(num_workers=num_workers, drop_last=True, buffer_size=buffer_size)
        self.dataset_dir = dataset_dir
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.name=name 
        self.head=head 
        self.body=body
        self.image_root = os.path.join(dataset_dir, 'images')
        self.label_root = os.path.join(dataset_dir, 'annotations')

        self.data_list_path = os.path.join(dataset_dir, 'datalist/{}.lst'.format(self.split))
        self.image_path = []
        self.anno_path = []
        self.class_names = [] 

        for dir in os.listdir(self.image_root):
                if dir[-5:] == '.xlsx':
                    continue
                self.class_names.append(dir)
        self.class_names.sort()
        self.num_classes = len(self.class_names)
        
        self.transforms = Compose(transforms) if transforms is not None else None

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
    

    def collate_batch(self, batch):
        imgs = []
        names = []
        bodies = []
        heads = []
        max_width = 0
        max_height = 0
        for img, labels in batch:
            imgs.append(img)

            [name, head_bbox, body_bbox] = labels

            width, height = img.shape[-1], img.shape[-2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)

            if self.name:
                names.append(self.class_names.index(name))
            if self.head:
                heads.append(np.array(head_bbox))
            if self.body:
                bodies.append(np.array(body_bbox))
            
        img_batch = np.stack(imgs)
        name_batch = np.stack(names)
        body_batch = np.stack(heads) if self.body else None
        head_batch = np.stack(bodies) if self.head else None
        return img_batch, tuple([name_batch, body_batch, head_batch])
            
            

    def __getitem__(self, image_id):
        image_path = self.image_path[image_id]
        anno_path = self.anno_path[image_id]
        image, anno = self.fetch(image_path, anno_path)
        name = os.path.dirname(anno_path).split('/')[-1]
        body_bnd_box = anno.get_bbox('bodybndbox')
        head_bnd_box = anno.get_bbox('headbndbox')
        bboxes = [head_bnd_box, body_bnd_box]
        if self.transforms is not None:
            image, bboxes = self.transforms(image, bboxes)
        normalize = Normalize()
        image = normalize(image)
        im = image.transpose(2, 0, 1)
        return im, tuple([name, bboxes[0], bboxes[1]])
    
    

    def fetch(self, image_path, annotation_path):
        im = Image.open(image_path).convert('RGB')
        anno = Annotation(path=annotation_path)
        return im, anno


if __name__ == '__main__':
    from lib.datasets.transforms import *
    transform = [Rotate(), HorizontalFlip(), VerticalFlip(), Blur(), ColorAugmentation(), Resize(224)]
    train_dataset = TsinghuaDogs(dataset_dir='/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/', transforms=transform, batch_size=2, shuffle=False, head=True, body=True)
    for idx, (inputs, targets) in enumerate(train_dataset):
        pic = np.array(inputs[0])
        target_name = targets[0][0]
        target_head = targets[1][0]
        target_body = targets[2][0]
        print(target_name)
        im = Image.fromarray(np.int8(pic).transpose(1, 2, 0), 'RGB')
        draw = ImageDraw.Draw(im)
        draw.rectangle(list(target_head), outline='red')
        draw.rectangle(list(target_body), outline='blue')
        im.save('./{}.png'.format(idx))
        print('name for ./{}.png is {}'.format(idx, target_name))
        if idx >= 5:
            break