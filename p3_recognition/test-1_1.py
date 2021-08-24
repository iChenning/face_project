import os
import math
import argparse
from PIL import Image
from tqdm import tqdm
from thop import profile
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

import s_backbones as backbones
from s_utils.load_model import load_normal

from s_data.MaskTheFace.utils.aux_functions import mask_image2
import dlib
import glob
import cv2
import numpy as np
import random


test_trans = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def extract_feats(backbone, txt_dir, bs=64, is_withflip=True):
    # read path
    f_ = open(txt_dir, 'r')
    f_paths = []
    for line in f_.readlines():
        line = line.replace('\n', '')
        f_paths.append(line)
    f_.close()

    # path1 path2 label
    paths_labels = []
    for i in range(0, len(f_paths), 2):
        path1 = f_paths[i]
        path2 = f_paths[i + 1]
        # check
        tmp1 = os.path.split(os.path.split(path1)[0])[-1]
        tmp2 = os.path.split(os.path.split(path2)[0])[-1]
        if tmp1 == tmp2:
            label = eval(tmp1.split('_')[-1])
            paths_labels.append((path1, path2, label))

    # feats
    fs1 = []
    fs2 = []
    labels = []
    backbone.eval()

    mask_types = ['N95', 'cloth', 'KN95', 'surgical', 'gas']
    pattern = []
    for dir_ in os.listdir(os.path.join(os.getcwd(), 's_data/MaskTheFace/masks/textures')):
        pattern.extend(glob.glob(os.path.join(os.getcwd(), 's_data/MaskTheFace/masks/textures', dir_) + '/*'))
    dlib_detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "s_data/MaskTheFace/dlib_models/shape_predictor_68_face_landmarks.dat"
    dlib_predictor = dlib.shape_predictor(path_to_dlib_model)
    for i in tqdm(range(math.ceil(len(paths_labels) / bs))):
        sub_pl = paths_labels[i * bs: min(i * bs + bs, len(f_paths))]
        imgs1 = []
        imgs2 = []
        for path1, path2, label in sub_pl:
            labels.append(label)

            img = Image.open(path1).convert("RGB")
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = mask_image2(dlib_detector, dlib_predictor, img, mask_type=random.choice(mask_types), pattern=random.choice(pattern))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img1 = test_trans(img)
            img1 = torch.unsqueeze(img1, 0)
            imgs1.append(img1)

            img = Image.open(path2).convert("RGB")
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = mask_image2(dlib_detector, dlib_predictor, img, mask_type=random.choice(mask_types), pattern=random.choice(pattern))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img2 = test_trans(img)
            img2 = torch.unsqueeze(img2, 0)
            imgs2.append(img2)

        imgs1 = torch.cat(imgs1, dim=0)
        feat1 = backbone(imgs1.cuda())
        if is_withflip:
            imgs1_f = torch.flip(imgs1, (3,))
            feat1_f = backbone(imgs1_f.cuda())
            f1 = feat1 + feat1_f
            fs1.append(f1.cpu().data)
        else:
            fs1.append(feat1.cpu().data)

        imgs2 = torch.cat(imgs2, dim=0)
        feat2 = backbone(imgs2.cuda())
        if is_withflip:
            imgs2_f = torch.flip(imgs2, (3,))
            feat2_f = backbone(imgs2_f.cuda())
            f2 = feat2 + feat2_f
            fs2.append(f2.cpu().data)
        else:
            fs2.append(feat2.cpu().data)
    fs1 = torch.cat(fs1, 0)
    fs1 = F.normalize(fs1)
    fs2 = torch.cat(fs2, 0)
    fs2 = F.normalize(fs2)

    return fs1, fs2, torch.tensor(labels)


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

    # feats
    fs1, fs2, labels = extract_feats(backbone, args.txt_dir, args.bs, args.is_withflip)
    s = torch.sum(fs1 * fs2, dim=1)

    # acc save
    r_ = os.path.join(args.save_root, os.path.split(os.path.split(args.resume)[0])[-1]) + args.note_info
    if args.is_withflip:
        r_ += '-withflip'
    else:
        r_ += '-withoutflip'
    if not os.path.exists(r_):
        os.makedirs(r_)
    txt_name = args.txt_dir.split('-')[-1]
    f_ = open(os.path.join(r_, txt_name), 'w')
    thres = torch.arange(-1, 1, 0.001)
    accs = []
    fprs = []
    tprs = []
    for thre in thres:
        # TP FP TN FN
        predicts = s.gt(thre)
        tp = torch.sum(torch.logical_and(predicts, labels)).item()
        fp = torch.sum(torch.logical_and(predicts, torch.logical_not(labels))).item()
        tn = torch.sum(torch.logical_and(torch.logical_not(predicts), torch.logical_not(labels))).item()
        fn = torch.sum(torch.logical_and(torch.logical_not(predicts), labels)).item()

        acc = (tp + tn) / (tp + fp + tn + fn)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        accs.append(acc)
        tprs.append(tpr)
        fprs.append(fpr)

        line = 'threshold:{:.4f}, acc:{:.6f}, fpr:{:.6f}, tpr:{:.6f}'.format(thre.item(), acc, fpr, tpr) + '\n'
        f_.write(line)
    accs = torch.tensor(accs)
    best_acc = torch.max(accs)
    best_idx = torch.argmax(accs)
    line = 'best-acc:{:.6f}, threshold:{:.4f}'.format(best_acc.item(), thres[best_idx]) + '\n'
    f_.write(line)
    f_.close()

    # plot roc
    plt.plot(fprs, tprs)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc curve')
    name_ = txt_name.split('.')[0]
    plt.savefig(os.path.join(r_, 'roc-' + name_ + '.png'), dpi=300)
    plt.show()

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='iresnet200', help='backbone network')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--pruned_info', type=str, default='')
    parser.add_argument('--resume', type=str, default=r'E:\model-zoo\glint360k-iresnet200-cosloss-mask\backbone.pth')
    parser.add_argument('--txt_dir', type=str, default=r'E:\list-zoo\test-1_1-lfw.txt')

    parser.add_argument('--is_withflip', type=bool, default=False)

    parser.add_argument('--save_root', type=str, default=r'E:\results-1_1')
    parser.add_argument('--note_info', type=str, default='-mask')
    parser.add_argument('--bs', type=int, default=6)

    args_ = parser.parse_args()

    main(args_)
