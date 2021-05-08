#!/bin/sh
# to txt list
cd ./part0_data_prepare
python path_file2txt.py \
  --folder_dir /data/cve_data/results/FaceID \
  --txt_dir ../data_list/san_FaceID.txt
cd ../

# landmarks
cd ./part2_alignment
python step1_landmark.py \
  --trained_model /home/xianfeng.chen/workspace/pre_models/widerface-mobile0.25/mobile0.25_150.pth \
  --txt_dir ../data_list/san_FaceID.txt \
  --save_folder detect_results/san_FaceID
cd ../