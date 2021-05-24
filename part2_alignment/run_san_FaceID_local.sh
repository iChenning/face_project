# landmarks
python step1_landmark.py \
  --trained_model E:\\pre-models\\widerface-mobile0.25\\mobile0.25_150.pth \
  --txt_dir E:\\data_list\\san_FaceID.txt \
  --save_folder E:\\results_save\\part2_alignment\\san_FaceID

# alignment
python step2_alignment.py \
  --txt_dir E:\\data_list\\san_FaceID.txt \
  --bbox_dir E:\\results_save\\part2_alignment\\san_FaceID\\vis_bbox.txt \
  --save_dir E:\\datasets\\san_FaceID-alig