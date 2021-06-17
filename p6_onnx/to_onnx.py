import torch
import sys
import torchvision
import time

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal
device = torch.device('cpu')


def main():
    # net
    print('begin')
    if len(r'') > 0:
        f_ = open(r'E:\pruned_info-zoo\glint360k-se_iresnet100.txt')
        cfg_ = [int(x) for x in f_.read().split()]
        f_.close()
    else:
        cfg_ = None
    backbone = backbones.__dict__['iresnet100'](cfg=cfg_)
    state_dict = load_normal(r'E:\model-zoo\glint360k-iresnet100\backbone.pth')
    backbone.load_state_dict(state_dict)
    backbone.dropout = torch.nn.Sequential()
    backbone = backbone.to(device)
    backbone.eval()
    dummy_input = torch.randn(10, 3, 112, 112).to(device)
    print('load')

    input_names = ["actual_input_1"]
    output_names = ["output1"]
    torch.onnx.export(backbone, dummy_input, r'E:\model-zoo\glint360k-iresnet100\backbone.onnx',
                      verbose=False, input_names=input_names,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=11)

    # #########
    # print('begin')
    # torch.quantization.get_default_qat_qconfig('fbgemm')
    # backbone = torch.jit.load(r'E:\model-zoo\glint360k-se_iresnet100-pruned-QAT\backbone-QAT.tar')
    # dummy_input = torch.randn(10, 3, 112, 112)
    # print('load')
    #
    # input_names = ["actual_input_1"]
    # output_names = ["output1"]
    # backbone.eval()
    # output = backbone(dummy_input)
    # torch.onnx.export(backbone, dummy_input, r'E:\model-zoo\glint360k-se_iresnet100-pruned-QAT\backbone-QAT.onnx',
    #                   verbose=False, input_names=input_names,
    #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=11)


if __name__ =='__main__':
    main()