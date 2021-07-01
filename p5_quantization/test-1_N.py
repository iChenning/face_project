import os
import math
import shutil
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms

import sys

sys.path.append('.')

device = torch.device('cpu')


test_trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def extract_feats(backbone, txt_dir, bs=64, is_withflip=True, isQ=False):
    # read path
    if isQ:
        f_ = open(txt_dir, 'r')
        f_paths = []
        IDs = []
        for line in f_.readlines():
            words = line.replace('\n', '').split()
            f_paths.append(words[0])
            IDs.append(words[1])
        f_.close()
    else:
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
        imgs = []
        for f_path in paths:
            img = Image.open(f_path).convert("RGB")
            img = test_trans(img)
            img = torch.unsqueeze(img, 0)
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)
        feat = backbone(imgs.to(device))
        if is_withflip:
            imgs_f = torch.flip(imgs, (3,))
            feat_f = backbone(imgs_f.to(device))
            feat = feat + feat_f
            feats.append(feat.cpu().data)
        else:
            feats.append(feat.cpu().data)
    feats = torch.cat(feats, 0)
    feats = F.normalize(feats)
    if isQ:
        return feats, f_paths, IDs
    else:
        return feats, f_paths


def main(args):
    # net
    backbone = torch.jit.load(args.quantized_dir).to(device)

    # key feats
    key_feats, key_paths = extract_feats(backbone, args.key_dir, args.bs, args.is_withflip, isQ=False)

    # query feats
    query_feats, query_paths, IDs = extract_feats(backbone, args.query_dir, args.bs, args.is_withflip, isQ=True)

    # similarity
    s = torch.mm(query_feats, key_feats.T)
    s = s.cpu().data.numpy()
    print(s.shape)

    # similarity analysis and save
    s_argmax = s.argmax(axis=1)
    print(s_argmax.shape)
    s_max = s.max(axis=1)
    s_new = np.concatenate((s_max[np.newaxis, :], s_argmax[np.newaxis, :]), axis=0)
    r_ = os.path.join(args.save_root, os.path.split(os.path.split(args.quantized_dir)[0])[-1]) + args.note_info
    if args.is_withflip:
        r_ += '-withflip'
    else:
        r_ += '-withoutflip'
    if not os.path.exists(r_):
        os.makedirs(r_)
    if args.is_savefeat:
        np.save(os.path.join(r_, 'key_feats'), key_feats.cpu().numpy())
        np.save(os.path.join(r_, 'query_feats'), query_feats.cpu().numpy())
    for i in range(s.shape[0]):
        id = IDs[i]
        if id in key_paths[s_argmax[i]] and s_max[i] > args.threshold:
            r_new = os.path.join(r_, 'isAccept')
        else:
            if id in key_paths[s_argmax[i]]:
                r_new = os.path.join(r_, 'isSame')
            else:
                r_new = os.path.join(r_, 'isDiff')

        folder_ = os.path.join(r_new, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)))
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        old_p = query_paths[i]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(folder_, name_)
        shutil.copy(old_p, new_p)

        old_p = key_paths[s_argmax[i]]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(folder_, name_)
        shutil.copy(old_p, new_p)

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--quantized_dir', type=str,
                        default=r'E:\model-zoo\glint360k-se_iresnet100-pruned-QAT\backbone-QAT.tar')
    parser.add_argument('--query_dir', type=str, default=r'E:\list-zoo\san_results-single-alig-ID.txt')
    parser.add_argument('--key_dir', type=str, default=r'E:\list-zoo\san_3W.txt')

    parser.add_argument('--is_withflip', type=bool, default=True)
    parser.add_argument('--is_savefeat', type=bool, default=False)

    parser.add_argument('--save_root', type=str, default=r'E:\results-1_N')
    parser.add_argument('--note_info', type=str, default='-3W')
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--bs', type=int, default=12)

    args_ = parser.parse_args()
    main(args_)
