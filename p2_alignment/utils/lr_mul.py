import math
from torch.optim.lr_scheduler import LambdaLR


__all__ = ['warm_cos']


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_epoch, t_total, len_trainloader, cycles=.5, last_epoch=-1):
        self.warmup_epoch = warmup_epoch * len_trainloader
        self.t_total = t_total * len_trainloader
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_epoch:
            return float(step) / float(max(1.0, self.warmup_epoch))
        # progress after warmup
        progress = float(step - self.warmup_epoch) / float(max(1, self.t_total - self.warmup_epoch))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def warm_cos(optimizer, warmup_epoch, t_total, len_trainloader, cycles=.5, last_epoch=-1):
    return WarmupCosineSchedule(optimizer, warmup_epoch, t_total, len_trainloader, cycles=cycles, last_epoch=last_epoch)