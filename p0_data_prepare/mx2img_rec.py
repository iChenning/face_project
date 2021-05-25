import os
import cv2
import numbers
import argparse
import numpy as np
import mxnet as mx
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, save_dir):
        super(MXFaceDataset, self).__init__()
        self.transform = None
        self.root_dir = root_dir
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

        self.save_dir = save_dir

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        img = mx.image.imdecode(img).asnumpy()

        folder_ = os.path.join(self.save_dir, str(int(label)).zfill(7))
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        num_ = len(os.listdir(folder_))
        name_ = str(num_).zfill(3) + '.jpg'
        # print(os.path.join(folder_, name_), img.shape, label)
        cv2.imwrite(os.path.join(folder_, name_), img[..., ::-1])

        return torch.rand(1, 3, 112, 112), torch.tensor(1, dtype=torch.long)

    def __len__(self):
        return len(self.imgidx)


def main(args):
    # data
    trainset = MXFaceDataset(root_dir=args.root_dir, save_dir=args.save_dir)
    train_loader = DataLoader(trainset, 32, shuffle=False, num_workers=0,
                              pin_memory=True, drop_last=False)

    for step, (img, label) in enumerate(train_loader):
        print(step, len(train_loader))
        # if step > 2:
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--root_dir', type=str, default='/data/cve_data/glint360/glint360_data/')
    parser.add_argument('--save_dir', type=str, default='/data/cve_data/CveTestResult/glint360k')
    args_ = parser.parse_args()

    main(args_)
