import os
import time
import copy
import logging
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
import sys

sys.path.append('.')

import s_backbones as backbones
from s_fc.partial_fc import PartialFC
from s_utils.load_model import load_normal
import s_fc.losses as losses
from s_data.dataset_mx import MXFaceDataset
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
    trainset = MXFaceDataset(root_dir=args.train_txt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, args.bs, shuffle=False, num_workers=8,
                              pin_memory=True, sampler=train_sampler, drop_last=True)

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
    prefix_ = os.path.split(args.resume)[0]
    model_to_quantize = copy.deepcopy(backbone)
    del backbone
    qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('fbgemm')}
    model_to_quantize.train()
    model_prepared = prepare_qat_fx(model_to_quantize, qconfig_dict)
    if os.path.exists(os.path.join(prefix_, 'model_prepared.pth')):
        model_prepared.load_state_dict(load_normal(os.path.join(prefix_, 'model_prepared.pth')))
        if rank == 0:
            logging.info('load model_perpared.pth')
    model_prepared = model_prepared.cuda()
    model_prepared = DDP(module=model_prepared, device_ids=[local_rank])
    # fc and loss
    margin_softmax = losses.__dict__[args.loss]()
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=args.bs, margin_softmax=margin_softmax, num_classes=360232,
        sample_rate=args.sample_rate, embedding_size=args.embedding_size, prefix=prefix_)
    # optimizer
    opt_backbone = torch.optim.SGD(params=[{'params': model_prepared.parameters()}],
                                   lr=args.lr / 512 * args.bs * world_size,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    opt_pfc = torch.optim.SGD(params=[{'params': module_partial_fc.parameters()}],
                              lr=args.lr / 512 * args.bs * world_size,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    scheduler_backbone = torch.optim.lr_scheduler.MultiStepLR(opt_backbone, args.milestones, args.gamma)
    scheduler_pfc = torch.optim.lr_scheduler.MultiStepLR(opt_pfc, args.milestones, args.gamma)
    start_epoch = 0
    log_fre = len(train_loader) if args.log_fre is None else args.log_fre
    for i_epoch in range(start_epoch, args.max_epoch):
        model_prepared.train()
        module_partial_fc.train()
        train_sampler.set_epoch(i_epoch)  # essential
        start_time = time.time()
        for i_iter, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()

            # forward and backward
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            features = F.normalize(model_prepared(img))
            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            features.backward(x_grad)
            clip_grad_norm_(model_prepared.parameters(), max_norm=5, norm_type=2)
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
            torch.save(model_prepared.module.state_dict(), os.path.join(args.save_dir, "model_prepared.pth"))
        module_partial_fc.save_params()

    model_int8 = convert_fx(model_prepared.module)
    # save
    if rank is 0:
        torch.jit.save(torch.jit.script(model_int8), os.path.join(args.save_dir, 'backbone-QAT.tar'))

    # release dist
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--train_txt', type=str, default='/data/cve_data/glint360/glint360_data/')
    parser.add_argument('--test_txt', type=str, default='')
    parser.add_argument('--bs', type=int, default=96)

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--pruned_info', type=str,
                        default='/home/xianfeng.chen/workspace/pruned_info-zoo/glint360k-se_iresnet100.txt')
    parser.add_argument('--resume', type=str,
                        default='/home/xianfeng.chen/workspace/model-zoo/glint360k-se_iresnet100-pruned-QAT/backbone.pth')
    parser.add_argument('--loss', type=str, default='cosloss', help='loss function')
    parser.add_argument('--sample_rate', type=float, default=1.0)

    parser.add_argument('--max_epoch', type=int, default=10, help='20 for glint360k, 100 for webface')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--milestones', type=list, default=[5, 8],
                        help='[6, 11, 15, 18] for glint360k, [40, 70, 90] for webface')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0, help='0.0 for glint360k, 0.4 for webface')

    parser.add_argument('--embedding_size', type=int, default=512)

    parser.add_argument('--model_zoo', type=str, default='/home/xianfeng.chen/workspace/model-zoo')
    parser.add_argument('--set_name', type=str, default='glint360k')
    parser.add_argument('--node', type=str, default='-pruned-QAT')
    parser.add_argument('--log_fre', type=int, default=100)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.model_zoo, args.set_name + '-' + args.network + args.node)

    rand_seed()
    main(args)
