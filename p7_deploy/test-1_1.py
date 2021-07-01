import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_txt(folder_dir):
    files = os.listdir(folder_dir)
    files.sort()

    feats1 = []
    feats2 = []
    labels = []
    for i, file in tqdm(enumerate(files)):
        f_ = open(os.path.join(folder_dir, file))
        feat = [float(x) for x in f_.read().split()]
        if i % 2 == 0:
            feats1.append(feat)
            label = eval(file.split('_')[1])
            labels.append(label)
        else:
            feats2.append(feat)
        f_.close()
    feats1 = np.array(feats1)
    feats1 = torch.from_numpy(feats1)
    feats1 = F.normalize(feats1, dim=1)

    feats2 = np.array(feats2)
    feats2 = torch.from_numpy(feats2)
    feats2 = F.normalize(feats2, dim=1)

    return feats1, feats2, torch.tensor(labels)


def main(args):
    # read path
    fs1, fs2, labels = read_txt(args.folder_dir)
    s = torch.sum(fs1 * fs2, dim=1)

    # acc fpr tpr save
    r_ = args.save_root
    if not os.path.exists(r_):
        os.makedirs(r_)
    name_ = args.folder_dir.split('-')[-1]
    f_ = open(os.path.join(r_, name_ + '.txt'), 'w')
    thres = torch.arange(0, 1, 0.001)
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
    plt.savefig(os.path.join(r_, 'roc-' + name_ + '.png'), dpi=300)
    plt.show()

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--folder_dir', default=r'E:\results-deploy\glint360k-iresnet100\results-agedb_30')
    parser.add_argument('--save_root', default=r'E:\results-deploy\glint360k-iresnet100\1_1-withoutflip')

    args_ = parser.parse_args()

    main(args_)
