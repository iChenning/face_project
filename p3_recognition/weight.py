import os
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
import sys

sys.path.append('.')

import s_backbones as backbones
import s_fc.losses as losses
from s_fc.partial_fc import PartialFC
from s_data.augment import train_trans_list_hard
from s_data.dataset_mx import MXFaceDataset
from s_data.dataset import MyDataset
from s_utils.seed_init import rand_seed
from s_utils.log import init_logging


def main(args):
    # dist
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = args.local_rank
    rank = int(os.environ['RANK'])
    dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # logging
    if not os.path.exists(args.save_dir) and rank == 0:
        os.makedirs(args.save_dir)
    if rank == 0:
        log_root = logging.getLogger()
        init_logging(log_root, args.save_dir)
        logging.info(args)

    # data
    trans_list = train_trans_list_hard() if args.augment_hard else None
    # trainset = MXFaceDataset(root_dir=args.train_txt, transform_list=trans_list)
    trainset = MyDataset(args.train_txt, transform_list=trans_list)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, args.bs, shuffle=False, num_workers=16, prefetch_factor=4,
                              pin_memory=True, sampler=train_sampler, drop_last=True)

    # backbone and DDP
    backbone = backbones.__dict__[args.network](dropout=args.dropout, embedding_size=args.embedding_size)
    backbone_pth = os.path.join(args.save_dir, "backbone.pth")
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
    if rank is 0:
        logging.info("backbone resume successfully!")
    backbone = backbone.cuda()
    backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    backbone = DDP(module=backbone, device_ids=[local_rank])


    # train and test
    log_fre = len(train_loader) if args.log_fre is None else args.log_fre
    start_epoch = 0
    save_iter = 0
    for i_epoch in range(start_epoch, args.max_epoch):
        backbone.eval()
        train_sampler.set_epoch(i_epoch)  # essential
        start_time = time.time()
        save_features = []
        save_label = []
        for i_iter, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()

            # forward and backward
            features = F.normalize(backbone(img))

            if (i_iter + 1) % 5000 == 0 and rank == 0:
                save_features = torch.cat(save_features, dim=0)
                torch.save(save_features, os.path.join(args.save_dir, str(save_iter) + '_feature.pth'))
                save_label = torch.cat(save_label, dim=0)
                torch.save(save_label, os.path.join(args.save_dir, str(save_iter) + '_label.pth'))
                save_features = []
                save_label = []
                save_iter += 1
            else:
                save_features.append(features.cpu().data)
                save_label.append(label.cpu().data)

            # logging info
            if (i_iter + 1) % log_fre == 0 and rank == 0:
                eta = (time.time() - start_time) / (i_iter + 1) * \
                      ((args.max_epoch - i_epoch) * len(train_loader) - i_iter - 1) / 3600
                logging.info("Training: Epoch[{:0>2}/{:0>2}] "
                             "Iter[{:0>5}/{:0>5}] "
                             "ETA:{:.2f}h".format(
                    i_epoch + 1, args.max_epoch,
                    i_iter + 1, len(train_loader),
                    eta))

    # release dist
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    # parser.add_argument('--train_txt', type=str, default='/data/cve_data/glint360/glint360_data/')
    parser.add_argument('--train_txt', type=str, default='/home/xianfeng.chen/workspace/list-zoo/train_webface260m_list.txt')
    parser.add_argument('--test_txt', type=str, default='')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--augment_hard', type=bool, default=False)

    parser.add_argument('--network', type=str, default='iresnet100', help='backbone network')
    parser.add_argument('--loss', type=str, default='cosloss', help='loss function')
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--resume', type=int, default=0, help='model resuming')

    parser.add_argument('--max_epoch', type=int, default=1, help='9 for webface260m, 20 for glint360k, 100 for webface')
    parser.add_argument('--lr', type=float, default=0.001)  # 0.1
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--milestones', type=list, default=[2, 3, 4],
                        help='[3, 5, 7, 8] for webface260m, [6, 11, 15, 18] for glint360k, [40, 70, 90] for webface')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0, help='0.0 for glint360k, 0.4 for webface')

    parser.add_argument('--embedding_size', type=int, default=512)

    parser.add_argument('--model_zoo', type=str, default='/data/cve_data/results/xianfeng.chen/model-zoo')
    parser.add_argument('--set_name', type=str, default='webface260m')
    parser.add_argument('--num_classes', type=int, default=2057290, help='2057290 for webface260m, 360232 for glink360k, 10572 for webface')
    parser.add_argument('--node', type=str, default='-mask')
    parser.add_argument('--log_fre', type=int, default=1)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.model_zoo, args.set_name + '-' + args.network + '-' + args.loss + args.node)

    rand_seed()
    main(args)
