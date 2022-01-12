import shutil
from tqdm import tqdm
import argparse
import os
import urllib.request 

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def unzip_files(dest_path):
    print('-------unpacking train validation lists-------')
    shutil.unpack_archive('TrainValSplit.zip', os.path.join(dest_path, 'tsinghua-dogs-data'))
    print('-------unpacking low res images-------')
    shutil.unpack_archive('low-res-images.zip', os.path.join(dest_path, 'tsinghua-dogs-data'))
    print('-------unpacking low res annotatios-------')
    shutil.unpack_archive('low-res-annotations.zip', os.path.join(dest_path, 'tsinghua-dogs-data'))
    
def download_files(save_dir):
    #resources=[images, labels]
    resources=[
        'https://cloud.tsinghua.edu.cn/f/80013ef29c5f42728fc8/?dl=1', 
        'https://cg.cs.tsinghua.edu.cn/ThuDogs/low-annotations.zip'
        ]

    test_val_split='https://cg.cs.tsinghua.edu.cn/ThuDogs/TrainValSplit.zip'

    download_url(test_val_split, os.path.join(save_dir, 'TrainValSplit.zip'))
    download_url(resources[0], os.path.join(save_dir, 'low-res-images.zip'))
    download_url(resources[1], os.path.join(save_dir,'low-res-annotations.zip'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action='store_true'
    )
    parser.add_argument(
        "--zip",
        action='store_true'
    )
    parser.add_argument(
        "--save_dir",
        default=".",
        type=str,
    )
    args = parser.parse_args()

    if args.download:
        download_files(args.save_dir)
    if args.zip:
        unzip_files(args.save_dir)