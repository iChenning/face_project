import os
import cv2
import numpy as np
import argparse


imgSize = [112, 112]

coord5point = [[37.8517, 51.0274],  # calculate from lfw and agedb-30
               [73.2348, 50.7187],
               [55.1707, 71.2361],
               [41.5770, 88.149],
               [69.8179, 87.8327]]


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.asmatrix([0., 0., 1.])])


def main(args):
    f_ = open(args.txt_dir)
    files = f_.read().split()
    f_.close()
    files.sort()  # stay with landmark

    lands = []
    f_ = open(args.bbox_dir, 'r')
    for line in f_.readlines():
        words = line.split()
        if len(words) > 6:
            land = [[int(words[6]), int(words[7])],
                    [int(words[8]), int(words[9])],
                    [int(words[10]), int(words[11])],
                    [int(words[12]), int(words[13])],
                    [int(words[14]), int(words[15])]]
            flag = 1
        else:
            land = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            flag = 0
        lands.append((land, flag))

    for i, pic_path in enumerate(files):
        if lands[i][1] == 0:
            continue
        img_im = cv2.imread(pic_path)
        M = transformation_from_points(np.asmatrix(lands[i][0]), np.asmatrix(coord5point))
        dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
        crop_im = dst[0:imgSize[0], 0:imgSize[1]]
        if crop_im.shape[0] != 112 or crop_im.shape[1] != 112:
            crop_im = cv2.copyMakeBorder(crop_im, 0, 112 - crop_im.shape[0], 0, 112 - crop_im.shape[1],
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
        name_ = os.path.split(pic_path)[-1]
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_name = os.path.join(args.save_dir, name_)
        cv2.imwrite(save_name, crop_im)

        cv2.imshow('orig', img_im)
        cv2.imshow('affine', dst)
        cv2.imshow('crop', crop_im)
        cv2.waitKey(1)


if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--txt_dir', type=str, default=r'E:\list-zoo\san_results-single.txt')
    parse.add_argument('--bbox_dir', type=str, default=r'E:\results_save\part2_alignment\san_results-single\vis_bbox.txt')
    parse.add_argument('--save_dir', type=str, default=r'E:\datasets\san_results-single-alig')

    args = parse.parse_args()
    main(args)