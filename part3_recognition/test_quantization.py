import argparse
import logging
import os
from tqdm import  tqdm
import torch

import torch.utils.data.distributed

from config import config as cfg


torch.backends.cudnn.benchmark = True


from eval import verification
from typing import List

class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            print('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            print(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank is 0 and num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


def main(args):
    callback_verification = CallBackVerification(1, 0, ["lfw"], r'E:\datasets\faces_webface_112x112')
    loaded_quantized_model = torch.jit.load('int8.pth').to('cpu')
    callback_verification(2, loaded_quantized_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--loss', type=str, default='cosloss', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)
