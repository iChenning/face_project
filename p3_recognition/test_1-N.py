import os
import math
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from thop import profile
import torch
import torch.nn.functional as F
from torchvision import transforms

import sys

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal


test_trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


test_trans2 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def extract_feats(backbone, txt_dir, bs=64):
    # read path
    f_ = open(txt_dir, 'r')
    f_paths = []
    for line in f_.readlines():
        line = line.replace('\n', '')
        f_paths.append(line)
    f_.close()

    # feats
    feats = []
    backbone.eval()
    for i in tqdm(range(math.ceil(len(f_paths) / bs))):
        paths = f_paths[i * bs: min(i * bs + bs, len(f_paths))]
        imgs1 = []
        imgs2 = []
        for f_path in paths:
            img = Image.open(f_path).convert("RGB")
            img1 = test_trans(img)
            img1 = torch.unsqueeze(img1, 0)
            imgs1.append(img1)
            img2 = test_trans2(img)
            img2 = torch.unsqueeze(img2, 0)
            imgs2.append(img2)

        imgs1 = torch.cat(imgs1, dim=0)
        imgs2 = torch.cat(imgs2, dim=0)
        feat1 = backbone(imgs1.cuda())
        feat2 = backbone(imgs2.cuda())
        feat = feat1 + feat2
        feats.append(feat.cpu().data)
    feats = torch.cat(feats, 0)
    feats = F.normalize(feats)
    return feats, f_paths


def main(args):
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
    backbone = backbone.cuda()

    # macs-params
    macs, params = profile(backbone, inputs=(torch.rand(1, 3, 112, 112).cuda(),))
    print('macs:', round(macs / 1e9, 2), 'G, params:', round(params / 1e6, 2), 'M')

    # key feats
    key_feats, key_paths = extract_feats(backbone, args.key_dir, args.bs)

    # query feats
    query_feats, query_paths = extract_feats(backbone, args.query_dir, args.bs)

    # similarity
    s = torch.mm(query_feats, key_feats.T)
    s = s.cpu().data.numpy()
    print(s.shape)

    # similarity analysis and save
    s_argmax = s.argmax(axis=1)
    print(s_argmax.shape)
    s_max = s.max(axis=1)
    s_new = np.concatenate((s_max[np.newaxis, :], s_argmax[np.newaxis, :]), axis=0)
    r_ = os.path.join(args.save_root, os.path.split(os.path.split(args.resume)[0])[-1]) + args.note_info
    if not os.path.exists(r_):
        os.makedirs(r_)
    for i in range(s.shape[0]):
        if not os.path.exists(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)))):
            os.makedirs(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100))))
        old_p = query_paths[i]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
        shutil.copy(old_p, new_p)

        old_p = key_paths[s_argmax[i]]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
        shutil.copy(old_p, new_p)

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--pruned_info', type=str, default=r'E:\pruned_info\glint360k-se_iresnet100.txt')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100-new\backbone.pth')
    parser.add_argument('--query_dir', type=str, default=r'E:\data_list\san_results-single-alig.txt')
    parser.add_argument('--key_dir', type=str, default=r'E:\data_list\san_3W.txt')

    parser.add_argument('--save_root', type=str, default=r'E:\results-1_N')
    parser.add_argument('--note_info', type=str, default='-3W-new')
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--bs', type=int, default=12)

    args_ = parser.parse_args()
    main(args_)
