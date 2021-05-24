from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

import argparse
import torch
import utils.backbones as backbones
from utils.load_model import load_normal
from s_data import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import torchvision

device = torch.device('cuda')


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            model(image.to(device))


test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def main(args):
    # data
    test_dataset = MyDataset(r'E:\data_list\train_casia-webface_list.txt', test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=100,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=8)

    # net
    net = backbones.__dict__[args.network](pretrained=False, dropout=0.0, fp16=False)
    state_dict = load_normal(args.resume)
    net.load_state_dict(state_dict)

    model = copy.deepcopy(net).to(device)
    model.eval()
    graph_module = torch.fx.symbolic_trace(model)
    qconfig = get_default_qconfig("qnnpack")
    qconfig_dict = {"": qconfig}
    model_prepared = prepare_fx(graph_module, qconfig_dict)
    calibrate(model_prepared, test_loader)  # 这一步是做后训练量化
    model_int8 = convert_fx(model_prepared)
    torch.jit.save(torch.jit.script(model_int8), 'int8.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100\backbone.pth')

    args = parser.parse_args()
    main(args)
