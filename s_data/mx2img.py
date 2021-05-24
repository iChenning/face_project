import cv2
from eval import verification
import os
import argparse
from tqdm import tqdm

def main(args):
    data_set = verification.load_bin(args.bin_dir, (112, 112))

    imgs_list = data_set[0]
    isname_list = data_set[1]

    for i, v in tqdm(enumerate(isname_list)):
        folder_ = os.path.join(args.save_dir, str(i).zfill(4) + '_' + str(v))
        if not os.path.exists(folder_):
            os.makedirs(folder_)
        tmp_img1 = imgs_list[0][2*i].numpy().transpose(1,2,0)
        tmp_img2 = imgs_list[0][2*i+1].numpy().transpose(1,2,0)

        cv2.imwrite(os.path.join(folder_, '1.jpg'), tmp_img1[...,::-1])
        cv2.imwrite(os.path.join(folder_, '2.jpg'), tmp_img2[...,::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_dir', type=str, default=r'E:\datasets\faces_webface_112x112\agedb_30.bin')
    parser.add_argument('--save_dir', type=str, default=r'E:\datasets\agedb_30')

    args = parser.parse_args()
    main(args)