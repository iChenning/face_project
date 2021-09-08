import os
import sys
import argparse
import torch

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal
device = torch.device('cuda')


def main(args):
    # net
    if len(args.pruned_info) > 0:
        f_ = open(args.pruned_info)
        cfg_ = [int(x) for x in f_.read().split()]
        f_.close()
    else:
        cfg_ = None
    backbone = backbones.__dict__[args.network](cfg=cfg_, embedding_size=args.embedding_size)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.to(device)
    print('load')

    # to onnx
    backbone.eval()
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    input_names = ["actual_input_1"]
    save_dir = os.path.join(os.path.split(args.resume)[0], 'backbone.onnx')
    torch.onnx.export(backbone, dummy_input, save_dir,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='se_iresnet200', help='backbone network')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--pruned_info', type=str, default='')
    parser.add_argument('--resume', type=str, default=r'E:\model-zoo\glint360k-se_iresnet200-arcloss-mask\backbone.pth')

    args = parser.parse_args()
    main(args)