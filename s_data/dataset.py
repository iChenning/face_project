import os
import random
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from s_data.MaskTheFace.augment_mask import AugmentMask

default_trans_list = [
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]


class MyDataset(Dataset):
    def __init__(self, txt_path, transform_list=None, is_check_label=True):
        assert os.path.exists(txt_path), "nonexistent:" + txt_path
        # load path and label from txt
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        labels = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            if len(words) > 1:
                imgs.append((words[0].encode('utf-8'), int(words[1])))
                labels.append(int(words[1]))
            elif len(words) == 1:
                imgs.append((words[0].encode('utf-8'), 0))
                labels.append(0)
        f.close()

        if is_check_label:
            self.labels_check(labels)

        # imgs and transform
        self.imgs = imgs
        self.aug_mask = AugmentMask('./s_data/MaskTheFace', mask_rate=0.3)
        trans_list = [transforms.ToPILImage()]
        trans_list.extend(transform_list if transform_list is not None else default_trans_list)
        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        wait_read = True
        while wait_read:
            try:
                img = Image.open(f_path).convert("RGB")
                wait_read = False
            except:
                print(f_path)
                f_path, label = self.imgs[random.randint(0, len(self.imgs) - 1)]

        img = np.asarray(img)
        img = self.aug_mask.mask(img)
        img = self.transform(img)

        # return (img, label, f_path, index)
        return (img, label)

    def __len__(self):
        return len(self.imgs)

    def labels_check(self, labels):
        """
        ??????labels????????????????????????0??????????????????????????????
        """
        labels_set = set(labels)
        labels_continuous = set(range(len(labels_set)))
        labels_diff = labels_continuous - labels_set
        assert len(labels_diff) == 0, print(labels_diff, len(labels))
        print('label is from 0 and continue, pass label check!')
