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


def main(args):
    # net
    if len(args.pruned_info) > 0:
        f_ = open(args.pruned_info)
        cfg_ = [int(x) for x in f_.read().split()]
        f_.close()
    else:
        cfg_ = None
    backbone = backbones.__dict__[args.network](cfg=cfg_, embedding_size=args.embedding_size)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()

    # macs-params
    macs, params = profile(backbone, inputs=(torch.rand(1, 3, 112, 112).cuda(),))
    print('macs:', round(macs / 1e9, 2), 'G, params:', round(params / 1e6, 2), 'M')

    # read path
    f_ = open(args.txt_dir, 'r')
    f_paths = []
    for line in f_.readlines():
        line = line.replace('\n', '')
        f_paths.append(line)
    f_.close()
    f_paths.sort()

    # feat
    feats = []
    backbone.eval()
    for f_path in f_paths:
        img = Image.open(f_path).convert("RGB")

        img1 = test_trans(img)
        img1 = torch.unsqueeze(img1, 0)
        img2 = test_trans2(img)
        img2 = torch.unsqueeze(img2, 0)

        feat1 = backbone(img1.cuda())
        feat2 = backbone(img2.cuda())
        feat = feat1 + feat2
        feats.append(feat.cpu().data)
    feats = torch.cat(feats, 0)
    feats = F.normalize(feats)

    # similarity
    s = torch.mm(feats, feats.T)
    s = s.cpu().data.numpy()
    s[list(range(s.shape[0])), list(range(s.shape[0]))] = 0

    # similarity analysis and save
    s_argmax = s.argmax(axis=1)
    s_max = s.max(axis=1)
    s_new = np.concatenate((s_max[np.newaxis, :], s_argmax[np.newaxis, :]), axis=0)
    r_ = os.path.join(args.save_root, os.path.split(os.path.split(args.resume)[0])[-1]) + args.note_info
    if not os.path.exists(r_):
        os.makedirs(r_)
    r_2 = args.san_1920_dir
    for i in range(s.shape[0]):
        if s_max[i] > args.threshold:
            if not os.path.exists(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)))):
                os.makedirs(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100))))
            old_p = f_paths[i]
            name_ = os.path.split(old_p)[-1]
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
            shutil.copy(old_p, new_p)
            id_fre = name_.split('_')[-1]
            old_p = os.path.join(r_2, id_fre)
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), id_fre)
            shutil.copy(old_p, new_p)

            old_p = f_paths[s_argmax[i]]
            name_ = os.path.split(old_p)[-1]
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
            shutil.copy(old_p, new_p)
            id_fre = name_.split('_')[-1]
            old_p = os.path.join(r_2, id_fre)
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), id_fre)
            shutil.copy(old_p, new_p)

    print('done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100-pruned\backbone.pth')
    parser.add_argument('--pruned_info', type=str, default=r'E:\pruned_info\glint360k-se_iresnet100-0.3.txt')
    parser.add_argument('--txt_dir', type=str, default=r'E:\data_list\san_results-single-alig.txt')

    parser.add_argument('--save_root', type=str, default=r'E:\results_dup')
    parser.add_argument('--note_info', type=str, default='')
    parser.add_argument('--san_1920_dir', type=str, default=r'E:\datasets\san_1920')
    parser.add_argument('--threshold', type=float, default=0.55)

    args_ = parser.parse_args()

    main(args_)
