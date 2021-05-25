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
from s_data.dataset_mx import MXFaceDataset
from s_utils.seed_init import rand_seed
from s_utils.log import init_logging

torch.backends.cudnn.benchmark = True


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
    trainset = MXFaceDataset(root_dir=args.train_txt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, args.bs, shuffle=False, num_workers=8,
                              pin_memory=True, sampler=train_sampler, drop_last=True)

    # backbone and DDP
    f_ = open(args.pruned_info)
    cfg_ = [int(x) for x in f_.read().split()]
    f_.close()
    backbone = backbones.__dict__[args.network](dropout=args.dropout, cfg=cfg_)
    if args.resume:
        try:
            backbone_pth = os.path.join(args.save_dir, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank is 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")
    backbone = backbone.cuda()
    backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)
    backbone = DDP(module=backbone, device_ids=[local_rank])

    # fc and loss
    margin_softmax = losses.__dict__[args.loss]()
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=args.bs, margin_softmax=margin_softmax, num_classes=360232,
        sample_rate=args.sample_rate, embedding_size=args.embedding_size, prefix=args.save_dir)

    # optimizer
    opt_backbone = torch.optim.SGD(params=[{'params': backbone.parameters()}],
                                   lr=args.lr / 512 * args.bs * world_size,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    opt_pfc = torch.optim.SGD(params=[{'params': module_partial_fc.parameters()}],
                              lr=args.lr / 512 * args.bs * world_size,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(opt_backbone, args.milestones, args.gamma)
    scheduler_pfc = torch.optim.lr_scheduler.MultiStepLR(opt_pfc, args.milestones, args.gamma)

    # train and test
    log_fre = len(train_loader) if args.log_fre is None else args.log_fre
    start_epoch = 0
    for i_epoch in range(start_epoch, args.max_epoch):
        backbone.train()
        module_partial_fc.train()
        train_sampler.set_epoch(i_epoch)  # essential
        start_time = time.time()
        for i_iter, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()

            # forward and backward
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            features = F.normalize(backbone(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            features.backward(x_grad)
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            opt_backbone.step()
            opt_pfc.step()
            module_partial_fc.update()

            # logging info
            if (i_iter + 1) % log_fre == 0 and rank == 0:
                eta = (time.time() - start_time) / (i_iter + 1) * \
                      ((args.max_epoch - i_epoch) * len(train_loader) - i_iter - 1) / 3600
                logging.info("Training: Epoch[{:0>2}/{:0>2}] "
                             "Iter[{:0>5}/{:0>5}] "
                             "lr: {:.5f} "
                             "Loss: {:.4f} "
                             "ETA:{:.2f}h".format(
                    i_epoch + 1, args.max_epoch,
                    i_iter + 1, len(train_loader),
                    opt_backbone.state_dict()['param_groups'][0]['lr'],
                    loss_v.item(),
                    eta))

        scheduler_backbone.step()
        scheduler_pfc.step()

        # save
        if rank is 0:
            torch.save(backbone.module.state_dict(), os.path.join(args.save_dir, "backbone.pth"))
        module_partial_fc.save_params()

    # release dist
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--train_txt', type=str, default='/data/cve_data/glint360/glint360_data/')
    parser.add_argument('--test_txt', type=str, default='')
    parser.add_argument('--bs', type=int, default=128)

    parser.add_argument('--network', type=str, default='se_iresnet18', help='backbone network')
    parser.add_argument('--pruned_info', type=str,
                        default='/home/xianfeng.chen/workspace/pruned_info-zoo/glint360k-se_iresnet100.txt')
    parser.add_argument('--loss', type=str, default='arcloss', help='loss function')
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--resume', type=int, default=0, help='model resuming')

    parser.add_argument('--max_epoch', type=int, default=20, help='20 for glint360k, 100 for webface')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--milestones', type=list, default=[6, 11, 15, 18],
                        help='[6, 11, 15, 18] for glint360k, [40, 70, 90] for webface')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0, help='0 for glint360k, 0.4 for webface')

    parser.add_argument('--embedding_size', type=int, default=512)

    parser.add_argument('--model_zoo', type=str, default='/home/xianfeng.chen/workspace/model-zoo')
    parser.add_argument('--set_name', type=str, default='glint360k')
    parser.add_argument('--node', type=str, default='-pruned')
    parser.add_argument('--log_fre', type=int, default=100)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.model_zoo, args.set_name + '-' + args.network + '-' + args.loss + args.node)

    rand_seed()
    main(args)
