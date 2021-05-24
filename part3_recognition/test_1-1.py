import argparse
import os
import torch
import utils.backbones as backbones
from utils.load_model import load_normal


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
    callback_verification = CallBackVerification(1, 0, ["lfw", "cfp_fp", "agedb_30"], r'E:\datasets\faces_webface_112x112')
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=0.0, fp16=False)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    # backbone = backbone.cuda()
    callback_verification(2, backbone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100\backbone.pth')

    parser.add_argument('--bs', type=int, default=12)

    args_ = parser.parse_args()
    main(args_)
