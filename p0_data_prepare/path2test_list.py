import os
import argparse

def main(args):
    # to list.txt
    root_ = args.folder_dir

    files = []
    for root, dirs, file in os.walk(root_):
        for f in file:
            files.append(os.path.join(root, f))

    folder_ = os.path.split(args.txt_dir)[0]
    if not os.path.exists(folder_):
        os.makedirs(folder_)
    f_ = open(args.txt_dir, 'w')
    for file in files:
        line = file + '\n'
        f_.write(line)
    f_.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--folder_dir', type=str,
                       default='/home/xianfeng.chen/workspace/dataset-zoo/face_recognition-test/1_1/agedb_30')
    parse.add_argument('--txt_dir', type=str, default='/home/xianfeng.chen/workspace/list-zoo/test-agedb_30.txt')

    args = parse.parse_args()
    main(args)