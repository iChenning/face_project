import copy
import argparse
from tqdm import tqdm
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset

import sys

sys.path.append('../p3_recognition')

import s_backbones as backbones
from s_utils.load_model import load_normal
from s_data.dataset import MyDataset


device = torch.device('cuda')


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        assert os.path.exists(txt_path), "nonexistent:" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append(words[0].encode('utf-8'))
        f.close()
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        f_path= self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image in tqdm(data_loader):
            model(image.to(device))


test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def main(args):
    # data
    test_dataset = MyDataset(r'E:\data_list\test-1_1-agedb_30.txt', test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=8)

    # net
    f_ = open(args.pruned_info)
    cfg_ = [int(x) for x in f_.read().split()]
    f_.close()
    net = backbones.__dict__[args.network](cfg=cfg_)
    state_dict = load_normal(args.resume)
    net.load_state_dict(state_dict)
    net.dropout = torch.nn.Sequential()

    model = copy.deepcopy(net).to(device)
    model.eval()
    graph_module = torch.fx.symbolic_trace(model)
    qconfig = get_default_qconfig("qnnpack")
    qconfig_dict = {"": qconfig}
    model_prepared = prepare_fx(graph_module, qconfig_dict)
    calibrate(model_prepared, test_loader)  # 这一步是做后训练量化
    model_int8 = convert_fx(model_prepared)
    torch.jit.save(torch.jit.script(model_int8), 'int8.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--pruned_info', type=str, default=r'E:\pruned_info\glint360k-se_iresnet100.txt')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100-pruned\backbone.pth')

    args = parser.parse_args()

    main(args)
