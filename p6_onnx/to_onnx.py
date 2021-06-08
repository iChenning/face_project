import torch
import sys
import torchvision
import time

sys.path.append('.')

import s_backbones as backbones
from s_utils.load_model import load_normal
device = torch.device('cpu')


def main():
    # # net
    # if len(r'E:\pruned_info-zoo\glint360k-se_iresnet100.txt') > 0:
    #     f_ = open(r'E:\pruned_info-zoo\glint360k-se_iresnet100.txt')
    #     cfg_ = [int(x) for x in f_.read().split()]
    #     f_.close()
    # else:
    #     cfg_ = None
    # backbone = backbones.__dict__['se_iresnet100'](cfg=cfg_)
    # state_dict = load_normal(r'E:\model-zoo\glint360k-se_iresnet100-pruned\backbone.pth')
    # backbone.load_state_dict(state_dict)
    # backbone.dropout = torch.nn.Sequential()
    # backbone = backbone.to(device)
    #
    # dummy_input = torch.randn(10, 3, 112, 112).to(device)
    #
    # input_names = ["actual_input_1"]
    # output_names = ["output1"]
    #
    # torch.onnx.export(backbone, dummy_input, "se_iresnet100.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names)

    # #########
    # print('begin')
    # # torch.backends.quantized.engine = "qnnpack"
    # torch.quantization.get_default_qat_qconfig('fbgemm')
    #
    # # backbone = torch.jit.load(r'E:\model-zoo\glint360k-se_iresnet100-pruned-PTQ2\backbone.tar')
    # backbone = torch.jit.load(r'E:\model-zoo\glint360k-se_iresnet100-pruned-QAT\backbone-QAT.tar')
    # # backbone = torch.jit.load('/home/xianfeng.chen/workspace/model-zoo/glint360k-se_iresnet100-pruned-QAT/backbone-QAT.tar')
    # dummy_input = torch.randn(10, 3, 112, 112)
    # print('load')
    #
    # input_names = ["actual_input_1"]
    # output_names = ["output1"]
    #
    # backbone.eval()
    # output = backbone(dummy_input)
    #
    # # torch.onnx.export(backbone, dummy_input, "se_iresnet100-quantized.onnx", verbose=False, example_outputs=output,
    # #                   input_names=input_names,
    # #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=9)
    # torch.onnx.export(backbone, dummy_input, "se_iresnet100-quantized.onnx", verbose=True, input_names=input_names,
    #                   output_names=output_names)

    x = torch.randn(100, 3, 112, 112)
    net_quantized = torchvision.models.resnet18(pretrained=True)
    time_s = time.time()
    net_quantized.eval()
    output = net_quantized(x)
    print('net_quantized time is :{:.4f}'.format(time.time() - time_s))
    input_names = ["actual_input_1"]
    import copy
    torch.onnx.export(copy.deepcopy(net_quantized), x, "net.onnx", verbose=False,
                      input_names=input_names,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=11)

    # # quantization
    # from torch.quantization import get_default_qconfig
    # from torch.quantization.quantize_fx import prepare_fx, convert_fx
    # model = copy.deepcopy(net_quantized)
    # model.eval()
    # graph_module = torch.fx.symbolic_trace(model)
    # qconfig = get_default_qconfig("fbgemm")
    # qconfig_dict = {"": qconfig}
    # model_prepared = prepare_fx(graph_module, qconfig_dict)
    # model_int8 = convert_fx(model_prepared)
    # torch.onnx.export(copy.deepcopy(model_prepared), x, "net-quantized.onnx", verbose=False,
    #                   input_names=input_names,
    #                   operator_export_type=torch.onnx.OperatorExportTypes.ONNX, opset_version=9)



if __name__ =='__main__':
    main()