from re import L
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import math
import torchvision

class lmdbDataset(Dataset):

    def __init__(self, root=None):
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
        self.transform = transforms.ToTensor()

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
                img = Image.open(buf).convert('RGB')
                img = self.transform(img)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            # # data augmentation
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

        return img

if __name__ =="__main__": 
    dataset_path = './data_CVPR2021/training/label/real/11.ReCTS'
    dataset = lmdbDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=1,shuffle=True)
    for i, batch in enumerate(loader):
        img = batch
        torchvision.utils.save_image(img,'ReCTS'+str(i)+'.jpg')
        if i >=2 :
            break
