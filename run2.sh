#!/bin/sh
# to txt list
cd ./part0_data_prepare
python path_file2txt.py
cd ../

# landmarks
cd ./part2_alignment
python step1_landmark.py
cd ../