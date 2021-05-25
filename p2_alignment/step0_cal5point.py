'''
this scrip is working for coord5point which are calculate from aligned images.
'''
import numpy as np
import argparse


def cal_points(txt_dir):
    f_ = open(txt_dir, 'r')
    points = []
    for line in f_.readlines():
        line = line.replace('\n', '')
        words = line.split()
        if len(words) >= 16:
            point = [float(x) for x in words[6:16]]
            points.append(point)
    f_.close()

    points = np.array(points)
    points_mean = points.mean(axis=0)
    return points_mean

def main(args):
    mean1 = cal_points(args.txt_dir1)
    mean2 = cal_points(args.txt_dir2)
    mean_ = (mean1 + mean2) / 2.
    print(mean1)
    print(mean2)
    print(mean_)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--txt_dir1', type=str, default='results_val-test/test_1-1_lfw/vis_bbox.txt')
    parse.add_argument('--txt_dir2', type=str, default='results_val-test/test_1-1_agedb-30/vis_bbox.txt')

    args = parse.parse_args()

    main(args)