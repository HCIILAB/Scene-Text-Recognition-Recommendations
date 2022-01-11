import random
import re
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import math
from config import get_args
import cv2
global_args = get_args(sys.argv[1:])

class lmdbDataset(Dataset):

    def __init__(self, alphabets, root=None, augmentation=False, target_transform=None, num_samples=math.inf):
        self.env = lmdb.open(
            root,
            max_readers=16,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.nSamples = min(self.nSamples, num_samples)
        self.augmentation = augmentation
        self.target_transform = target_transform
        self.alphabets = alphabets
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())       
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if global_args.RGB:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            # # data augmentation
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            if self.target_transform is not None:
                label = self.target_transform(label)
            if len(label)>=global_args.max_len:
                print('too long')
                return self[index+1]
            if global_args.alphabets == 'lowercase':
                label = label.lower()
            # also, we need to filter charcters not in the alphabets
            out_of_char = f'[^{self.alphabets}]'
            label = re.sub(out_of_char, '',label)
        return (img, label)

class Padresize(object):
    # 将文本行图像高度缩放到固定尺寸，宽度等比例变化，并进行填充
    # 对于超长样本，则需要缩放场边
    def __init__(self, height, width, interpolation=Image.BILINEAR,mode='RGB'):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.mode = mode
        self.toTensor = transforms.ToTensor()
    def __call__(self, img):
        iw, ih = img.size
        scale = self.height / ih
        target_height = self.height
        target_width = int(scale * iw)
        if target_width <= self.width:
            img = img.resize((target_width, target_height), self.interpolation)
            new_image = Image.new(self.mode,(self.width,self.height))
            new_image.paste(img)
        else:
            img = img.resize((target_width, target_height), self.interpolation)
            new_image = img.resize((self.width, target_height), self.interpolation)
        new_image = self.toTensor(new_image)
        return new_image

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.transform(img)
        img.sub_(0.5).div_(0.5)
        return img

# 如果没有SSD，建议用这个，否则机械硬盘会很慢
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

class alignCollate(object):

    def __init__(self, imgH, imgW):
        self.imgH = imgH
        self.imgW = imgW

    def __call__(self, batch):
        images, labels = zip(*batch)
        if global_args.padresize:
            transform = Padresize(self.imgH, self.imgW)
        elif global_args.keepratioresize: # being done when loading 
            images = torch.cat([t.unsqueeze(0) for t in images], 0)
            return images, labels
        else:
            transform = resizeNormalize((self.imgW, self.imgH))
        images = torch.cat([transform(t).unsqueeze(0) for t in images], 0)
        return images, labels