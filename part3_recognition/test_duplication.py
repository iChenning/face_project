import argparse
import torch
import numpy as np
import utils.backbones as backbones
from utils.load_model import load_normal
from config import config as cfg
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from thop import profile
import shutil
import os


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
    dropout = 0.4 if cfg.dataset is "webface" else 0
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=dropout, fp16=cfg.fp16)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()

    # macs-params
    macs, params = profile(backbone, inputs=(torch.rand(1, 3, 112, 112).cuda(),))
    print('macs:', macs, 'params:', params)

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

    parser.add_argument('--network', type=str, default='iresnet50', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-iresnet50\backbone.pth')
    parser.add_argument('--txt_dir', type=str, default=r'E:\data_list\san_results-single-alig.txt')

    parser.add_argument('--save_root', type=str, default=r'E:\dup')
    parser.add_argument('--note_info', type=str, default='')
    parser.add_argument('--san_1920_dir', type=str, default=r'E:\datasets\san_1920')
    parser.add_argument('--threshold', type=float, default=0.55)

    args_ = parser.parse_args()
    main(args_)
