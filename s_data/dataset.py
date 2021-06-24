import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        assert os.path.exists(txt_path), "nonexistent:" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        labels = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            # imgs.append((words[0].encode('utf-8'), int(words[1])))
            imgs.append((words[0].encode('utf-8'), 0))
            # labels.append(int(words[1]))
            labels.append(0)
        f.close()
        # self.n_classes = self.labels_check(labels)
        self.imgs = imgs
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [  # transforms.RandomResizedCrop(112, scale=(0.8, 1.)),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.Resize((112, 112)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        # print(img.shape, f_path)
        # return (img, label, f_path, index)
        return (img, label)

    def __len__(self):
        return len(self.imgs)

    def labels_check(self, labels):
        """
        判断labels是否是连续的，从0开始的，并返回类别数
        """
        labels_set = set(labels)
        labels_continuous = set(range(len(labels_set)))
        labels_diff = labels_continuous - labels_set
        assert len(labels_diff) == 0, print(labels_diff, len(labels))
        return len(labels_set)
