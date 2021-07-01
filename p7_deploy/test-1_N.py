import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import copy
import shutil
from tqdm import tqdm


def read_txt(folder_dir, img_root, txt_dir=None):
    IDs = []
    if txt_dir is not None:
        files = []
        f_ = open(txt_dir, 'r')
        for line in f_.readlines():
            words = line.replace('\n', '').split()
            IDs.append(words[1])
            file = os.path.split(words[0])[-1].replace('.jpg', '.txt')
            files.append(file)
        f_.close()
    else:
        files = os.listdir(folder_dir)
        files.sort()

    feats = []
    paths = []
    for file in tqdm(files):
        f_ = open(os.path.join(folder_dir, file))
        feat = [float(x) for x in f_.read().split()]
        feats.append(feat)
        paths.append(os.path.join(img_root, file.replace('.txt', '.jpg')))
        f_.close()
    feats = np.array(feats)
    feats = torch.from_numpy(feats)
    feats = F.normalize(feats, dim=1)

    if txt_dir is not None:
        return feats, paths, IDs
    else:
        return feats, paths


def main(args):
    query_feats, query_paths, IDs = read_txt(args.san_query, args.san_query_img, args.txt_dir)
    key_san_feats, key_san_paths = read_txt(args.key_san, args.key_san_img)
    key_3W_feats, key_3W_paths = read_txt(args.key_3W, args.key_3W_img)

    key_feats = torch.cat((key_san_feats, key_3W_feats), dim=0)
    key_paths = copy.deepcopy(key_san_paths)
    key_paths.extend(key_3W_paths)

    # similarity
    s = torch.mm(query_feats, key_feats.T)
    s = s.cpu().data.numpy()
    print(s.shape)

    # similarity analysis and save
    s_argmax = s.argmax(axis=1)
    print(s_argmax.shape)
    s_max = s.max(axis=1)
    # s_new = np.concatenate((s_max[np.newaxis, :], s_argmax[np.newaxis, :]), axis=0)

    # save
    r_ = args.save_root
    if not os.path.exists(r_):
        os.makedirs(r_)
    for i in range(s.shape[0]):
        id = IDs[i]
        if id in key_paths[s_argmax[i]] and s_max[i] > args.threshold:
            r_new = os.path.join(r_, 'isAccept')
        else:
            if id in key_paths[s_argmax[i]]:
                r_new = os.path.join(r_, 'isSame')
            else:
                r_new = os.path.join(r_, 'isDiff')

        folder_ = os.path.join(r_new, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)))
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        old_p = query_paths[i]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(folder_, name_)
        shutil.copy(old_p, new_p)

        old_p = key_paths[s_argmax[i]]
        name_ = os.path.split(old_p)[-1]
        new_p = os.path.join(folder_, name_)
        shutil.copy(old_p, new_p)

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir', type=str, default=r'E:\list-zoo\san_results-single-alig-ID.txt')
    parser.add_argument('--san_query', type=str, default=r'E:\results-deploy\glint360k-iresnet100\results_san_query')
    parser.add_argument('--san_query_img', type=str, default=r'E:\datasets\san_results-single-alig')

    parser.add_argument('--key_san', type=str, default=r'E:\results-deploy\glint360k-iresnet100\results_san_FaceID-alig')
    parser.add_argument('--key_san_img', default=r'E:\datasets\san_FaceID-alig-new')

    parser.add_argument('--key_3W', type=str, default=r'E:\results-deploy\glint360k-iresnet100\results_3W-alig')
    parser.add_argument('--key_3W_img', default=r'E:\datasets\faces-recognition_test\1-N\3W-alig')


    parser.add_argument('--save_root', type=str, default=r'E:\results-deploy\glint360k-iresnet100\1_N-withoutflip')
    parser.add_argument('--threshold', type=float, default=0.55)

    args = parser.parse_args()
    main(args)