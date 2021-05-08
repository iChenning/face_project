import argparse
import logging
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import utils.backbones as backbones
import utils.fc as losses
from utils.fc.partial_fc import PartialFC
from config import config as cfg
from utils.data.dataset_mx import MXFaceDataset
from torch.utils.data import DataLoader
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler

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
    if not os.path.exists(cfg.output) and rank is 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    # data
    trainset = MXFaceDataset(root_dir=cfg.rec)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = DataLoader(trainset, cfg.batch_size, shuffle=False, num_workers=8,
                              pin_memory=True, sampler=train_sampler, drop_last=True)

    # backbone and DDP
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=cfg.dropout, fp16=cfg.fp16)
    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
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
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    # optimizer
    opt_backbone = torch.optim.SGD(params=[{'params': backbone.parameters()}],
                                   lr=cfg.lr / 512 * cfg.batch_size * world_size,
                                   momentum=0.9,
                                   weight_decay=cfg.weight_decay)
    opt_pfc = torch.optim.SGD(params=[{'params': module_partial_fc.parameters()}],
                              lr=cfg.lr / 512 * cfg.batch_size * world_size,
                              momentum=0.9,
                              weight_decay=cfg.weight_decay)
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    # train and valid
    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank is 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(2000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            global_step += 1
            features = F.normalize(backbone(img))

            x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            if cfg.fp16:
                features.backward(grad_scaler.scale(x_grad))
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                features.backward(x_grad)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()

            opt_pfc.step()
            module_partial_fc.update()
            opt_backbone.zero_grad()
            opt_pfc.zero_grad()
            loss.update(loss_v, 1)

            lr = opt_backbone.state_dict()['param_groups'][0]['lr']
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler, lr)
            callback_verification(global_step, backbone)
        callback_checkpoint(global_step, backbone, module_partial_fc)

        scheduler_backbone.step()
        scheduler_pfc.step()

    # release dist
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--loss', type=str, default='cosloss', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)
