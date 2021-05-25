import os
import cv2
import pickle
import argparse
import mxnet as mx
from mxnet import ndarray as nd
from tqdm import tqdm
import torch


@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list


def main(args):
    data_set = load_bin(args.bin_dir, (112, 112))

    imgs_list = data_set[0]
    isname_list = data_set[1]

    for i, v in tqdm(enumerate(isname_list)):
        folder_ = os.path.join(args.save_dir, str(i).zfill(4) + '_' + str(v))
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        tmp_img1 = imgs_list[0][2 * i].numpy().transpose(1, 2, 0)
        tmp_img2 = imgs_list[0][2 * i + 1].numpy().transpose(1, 2, 0)

        cv2.imwrite(os.path.join(folder_, '1.jpg'), tmp_img1[..., ::-1])
        cv2.imwrite(os.path.join(folder_, '2.jpg'), tmp_img2[..., ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_dir', type=str, default=r'E:\datasets\vgg2_fp.bin')
    parser.add_argument('--save_dir', type=str, default=r'E:\datasets\vgg2_fp')

    args = parser.parse_args()
    main(args)
