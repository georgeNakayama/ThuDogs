import os
import numpy as np
import random
from PIL import Image

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 


def jigsaw_generator(images, n=1):
    #assumeing images are square sized and the side length is divisible by n
    if n == 1:
        return images
    size = images.shape[-1]
    patch_size = int(size / n)
    idxes = [(i * patch_size, j *  patch_size) for i in range(n) for j in range(n)]
    num_patches = len(idxes)
    random.shuffle(idxes)
    images_cp = images.copy()
    for i in range(n): 
        for j in range(n):
            idx = i * n + j
            temp = images_cp[..., idxes[idx][0]: idxes[idx][0]+ 1 * patch_size, idxes[idx][1]: idxes[idx][1] + 1 * patch_size]
            images[..., i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1)* patch_size] = temp
    return images
if __name__ == '__main__':
    path = '/mnt/disk/wang/THD-datasets/tsinghua-dogs-data/images/200-n000008-Airedale/n107026.jpg'
    im = Image.open(path)
    im = im.resize((200, 200))
    image = np.asarray(im, dtype=np.int8)
    image = image.transpose(2, 0, 1)
    images = image[np.newaxis, ...]
    print(images.shape)
    new_im = jigsaw_generator(images, 4)
    new_im = new_im[0].transpose(1, 2, 0)
    print(new_im.shape)
    im = Image.fromarray(new_im, 'RGB')
    im.save('./new.png')
