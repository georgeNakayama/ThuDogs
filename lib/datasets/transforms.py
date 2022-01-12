import random
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageStat
import copy
import numpy as np
from lib.datasets.annotation import Annotation
from lib.utils import TRANSFORMS

__all__ = ['Compose', 'Resize', 'Rotate', 'HorizontalFlip', 'VerticalFlip', 'Blur', 'ColorAugmentation', 'Normalize']

@TRANSFORMS.register_module()
class Compose:
    def __init__(self, transforms):
        self.transforms = copy.deepcopy(transforms)

    def __call__(self, image, bboxes=None):
        #bboxes is a list of dict contained the four corners of the bounding boxes
        for transform in self.transforms:
            image, bboxes = transform(image, bboxes)
        return image, bboxes

@TRANSFORMS.register_module()
class Resize:
    def __init__(self, shape):
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            assert isinstance(shape, tuple) and len(shape) == 2, 'shape must be either an integer or a two tuple'
            self.shape = shape 
    def __call__(self, image, bboxes=None):
        if bboxes is None:
            return image.resize(self.shape, resample=Image.BILINEAR)

        width, height = image.size
        ratios = [np.float32(self.shape[0] / width), np.float32(self.shape[1] / height)]
        new_bboxes = []
        for bbox in bboxes:
            new_bboxes.append([np.float32(ratios[i % 2] * bbox[i]) for i in range(4)])
        return image.resize(self.shape, resample=Image.BILINEAR), new_bboxes 

@TRANSFORMS.register_module()
class Rotate:
    def __init__(self, random=True, theta=90):
        self.random = random
        if self.random:
            self.angles = [0, 90, 180, 270]
        else:
            self.angles = [theta]

    def __call__(self, image, bboxes=None):
        angle = random.choice(self.angles)
        if bboxes is None:
            return image.rotate(angle, resample=Image.BILINEAR, expand=True)

        new_bboxes = []
        w, h = image.size
        for bbox in bboxes:
            if angle == 0:
                new_bboxes.append(bbox)
            elif angle == 90:
                new_bboxes.append([bbox[1], w - bbox[2], bbox[3], w - bbox[0]])
            elif angle == 180:
                new_bboxes.append([w - bbox[2], h - bbox[3], w - bbox[0], h - bbox[1]])
            elif angle == 270:
                new_bboxes.append([h - bbox[3], bbox[0], h - bbox[1], bbox[2]])
        return image.rotate(angle, resample=Image.BILINEAR, expand=True), new_bboxes

@TRANSFORMS.register_module()
class HorizontalFlip:
    def __init__(self, p = 0.5):
        self.prob = p

    def __call__(self, image, bboxes=None):
        flag = random.random() < self.prob
        if bboxes is None:
            return image.transpose(Image.FLIP_TOP_BOTTOM) if flag else image

        new_bboxes = []
        w, h = image.size
        new_bboxes = [[w - bbox[2], bbox[1], w - bbox[0], bbox[3]] if flag else bbox for bbox in bboxes ]
        im_flipped = image.transpose(Image.FLIP_LEFT_RIGHT) if flag else image
        return im_flipped, new_bboxes 

@TRANSFORMS.register_module()
class VerticalFlip:
    def __init__(self, p = 0.5):
        self.prob = p

    def __call__(self, image, bboxes=None):
        flag = random.random() < self.prob
        if bboxes is None:
            return image.transpose(Image.FLIP_TOP_BOTTOM) if flag else image

        new_bboxes = []
        w, h = image.size
        new_bboxes = [[bbox[0], h - bbox[3], bbox[2], h - bbox[1]] if flag else bbox for bbox in bboxes ]
        im_flipped = image.transpose(Image.FLIP_TOP_BOTTOM) if flag else image
        return im_flipped, new_bboxes 

@TRANSFORMS.register_module()
class Blur:
    def __init__(self, range=(0, 3)):
        self.range = range
    
    def __call__(self, image, bboxes=None):
        radius = random.randrange(self.range[0], self.range[1])
        if bboxes is not None:
            return image.filter(ImageFilter.GaussianBlur(radius = radius)), bboxes
        else:
            return image.filter(ImageFilter.GaussianBlur(radius = radius))

@TRANSFORMS.register_module()
class ColorAugmentation:
    def __init__(self, range=(0.5, 1.5), step=0.25):
        self.range = range
        self.step = step
    
    def __call__(self, image, bboxes=None):
        factors = [self.range[0] + self.step * i for i in range(int((self.range[1] - self.range[0]) / self.step))]

        mean = int(ImageStat.Stat(image.convert("L")).mean[0] + 0.5)
        contrastor_im= Image.new("L", image.size, mean).convert(image.mode)
        contrasted = Image.blend(contrastor_im, image, random.choice(factors))

        colorer_im = contrasted.convert('L').convert(contrasted.mode)
        colored = Image.blend(colorer_im, contrasted, random.choice(factors))

        brightener_im = Image.new(colored.mode, colored.size, 0)
        brightened = Image.blend(brightener_im, colored, random.choice(factors))

        sharp_im = brightened.filter(ImageFilter.SMOOTH)
        sharpened = Image.blend(sharp_im, brightened, random.choice(factors))

        return sharpened, bboxes 

@TRANSFORMS.register_module()
class Normalize:
    def __init__(self):
        pass 
    def __call__(self, image, bboxes=None):
        image_array = np.asarray(image).astype(np.float32)
        mean = image_array.mean(axis = (0, 1), dtype=np.float32)
        if bboxes is None:
            return (image_array - mean)
        return (image_array - mean), bboxes

@TRANSFORMS.register_module()
class Crop:
    def __init__(self, random=True, size=448):
        self.random = random 
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert isinstance(size, tuple) and len(size) == 2, 'size must be an integer or a two tuple of (width, height)'
            self.size = size
    def __call__(self, image, bboxes=None):
        w, h = image.size
        assert w >= self.size[0] and h >= self.size[1], 'input image size must be greater than {} by {}'.format(self.size[0], self.size[1])
        if self.random:
            x1 = random.randint(0, w - self.size[0])
            y1 = random.randint(0, h - self.size[1])
        else: 
            x1 = (w - self.size[0]) // 2
            y1 = (h - self.size[1]) // 2
        return image.crop((x1, y1, x1 + self.size[0], y1 + self.size[1])), bboxes



if __name__ == '__main__':
    anno_path = '/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/annotations/200-n000008-Airedale/n107026.jpg.xml'
    img_path = '/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/images/200-n000008-Airedale/n107026.jpg'
    im = Image.open(img_path)
    anno = Annotation(anno_path)
    four_corners = ['xmin', 'ymin', 'xmax', 'ymax']
    body = [anno.lookup('bodybndbox')[corner] for corner in four_corners]
    head = [anno.lookup('headbndbox')[corner] for corner in four_corners]
    resize = Resize(shape=550)
    crop = Crop(random=False, size=448)
    #rotate = Rotate(random=False, resample=Image.BILINEAR, theta=90)
    #flip = VerticalFlip(p = 1)
    #blur = Blur()
    #color_aug = ColorAugmentation()
    #im_resized, [head_new, body_new] = resize(im, bboxes=[head, body])
    im_rotated = resize(im)
    im_cropped = crop(im_rotated)
    im_rotated.save('./8.png')
    im_cropped.save('./9.png')