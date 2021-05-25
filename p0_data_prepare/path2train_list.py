import argparse
import os
from tqdm import tqdm


def run(args):
    txt_dir = args.txt_dir
    dataset_dir = args.folder_dir

    if not os.path.exists(os.path.split(txt_dir)[0]):
        os.makedirs(os.path.split(txt_dir)[0])
    f_ = open(txt_dir, "w")
    label = 0
    for root, dirs, files in tqdm(os.walk(dataset_dir)):
        files.sort()
        for file in files:
            file_dir = os.path.join(root, file)
            if file_dir.endswith('.jpg') or file_dir.endswith('.JPG') or \
                file_dir.endswith('.bmp') or file_dir.endswith('.png'):
                f_.writelines(file_dir + ' ' + str(label - 1) + "\n")
        label += 1
    f_.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--folder_dir', type=str, default='/data/cve_data/CveTestResult/glint360k')
    parse.add_argument('--txt_dir', type=str, default='/home/xianfeng.chen/workspace/list-zoo/train_glint360k_list.txt')

    args = parse.parse_args()
    run(args)