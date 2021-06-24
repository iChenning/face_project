import os
import argparse
import torch
import torch.nn as nn
import sys

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal
from thop import profile



def main(args):
    # net
    backbone = backbones.__dict__[args.network](embedding_size=args.embedding_size)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()
    print(backbone)

    #
    n_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    n_bns = ['bn1']

    # pruned sta
    total = 0
    bn = []
    for n_layer in n_layers:
        layer = getattr(backbone, n_layer)
        for d in range(len(layer)):
            block = layer[d]
            for n_bn in n_bns:
                m = getattr(block, n_bn)
                if isinstance(m, nn.BatchNorm2d):
                    total += m.weight.data.shape[0]
                    bn.extend(m.weight.data.abs().clone())
    bn = torch.tensor(bn)

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for n_layer in n_layers:
        layer = getattr(backbone, n_layer)
        for d in range(len(layer)):
            block = layer[d]
            for n_bn in n_bns:
                m = getattr(block, n_bn)
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())

    print('pruned_ratio =', pruned / total)
    print('Pruning done!')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    f_name = os.path.split(os.path.split(args.resume)[0])[-1] + '-' + str(args.percent) + '.txt'
    f_dir = os.path.join(args.save_dir, f_name)
    f_ = open(f_dir, 'w')
    for c in cfg:
        f_.write(str(c) + ' ')
    f_.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='iresnet100', help='backbone network')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--resume', type=str, default=r'E:\model-zoo\glint360k-iresnet100\backbone.pth')
    parser.add_argument('--percent', type=float, default=0.5,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--save_dir', type=str, default=r'E:\pruned_info-zoo')

    args_ = parser.parse_args()
    main(args_)
