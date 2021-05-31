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

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal
from s_data.dataset_mx import MXFaceDataset


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        count = 0
        for (img, label) in tqdm(data_loader):
            if count > 3000:
                break
            model(img.cuda())
            count += 1



test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def main(args):
    # data
    trainset = MXFaceDataset(root_dir=args.train_txt)
    train_loader = DataLoader(trainset, args.bs, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    # net
    if len(args.pruned_info) > 0:
        f_ = open(args.pruned_info)
        cfg_ = [int(x) for x in f_.read().split()]
        f_.close()
    else:
        cfg_ = None
    backbone = backbones.__dict__[args.network](cfg=cfg_)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone.dropout = torch.nn.Sequential()

    # quantization
    model = copy.deepcopy(backbone).cuda()
    model.eval()
    graph_module = torch.fx.symbolic_trace(model)
    qconfig = get_default_qconfig("fbgemm")
    qconfig_dict = {"": qconfig}
    model_prepared = prepare_fx(graph_module, qconfig_dict)
    calibrate(model_prepared, train_loader)  # 这一步是做后训练量化
    model_int8 = convert_fx(model_prepared)

    # save
    r_ = os.path.join(args.save_root, os.path.split(os.path.split(args.resume)[0])[-1]) + args.note_info
    if not os.path.exists(r_):
        os.makedirs(r_)
    torch.jit.save(torch.jit.script(model_int8), os.path.join(r_, 'backbone.tar'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--train_txt', type=str, default='/data/cve_data/glint360/glint360_data/')
    parser.add_argument('--bs', type=int, default=256)

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--pruned_info', type=str,
                        default='/home/xianfeng.chen/workspace/pruned_info-zoo/glint360k-se_iresnet100.txt')
    parser.add_argument('--resume', type=str,
                        default='/home/xianfeng.chen/workspace/model-zoo/glint360k-se_iresnet100-pruned/backbone.pth')

    parser.add_argument('--save_root', type=str,
                        default='/home/xianfeng.chen/workspace/model-zoo')
    parser.add_argument('--note_info', type=str, default='-PTQ')

    args = parser.parse_args()

    main(args)
