import torch
import os
from tqdm import tqdm


save_dir = '/data/cve_data/results/xianfeng.chen/model-zoo/webface260m-iresnet100-cosloss-mask'
num_classes = 2057290
weight = torch.zeros(num_classes, 512)
weight_count = torch.zeros(num_classes)

for i in range(2):
    feature = torch.load(os.path.join(save_dir, str(i) + '_feature.pth'))
    label = torch.load(os.path.join(save_dir, str(i) + '_label.pth'))
    for idx in tqdm(range(feature.shape[0])):
        weight[label[idx]] += feature[idx]
        weight_count[label[idx]] += 1.0

weight_count = weight_count.unsqueeze(1)
weight /= weight_count

world_size = 6
for i_rank in range(world_size):
    num_local: int = num_classes // world_size + int(i_rank < num_classes % world_size)
    class_start: int = num_classes // world_size * i_rank + min(i_rank, num_classes % world_size)
    print(num_local, class_start)

    weight_name = os.path.join(save_dir, "rank:{}_softmax_weight.pt".format(i_rank))

    weight_cur = weight[class_start: class_start + num_local]
    torch.save(weight_cur, weight_name)