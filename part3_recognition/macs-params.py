from utils.backbones import iresnet18, iresnet34, mobilev3, mobilev2
import torch
from thop import profile
import torchvision.models as models

x = torch.rand(1,3,112,112)

net1 = iresnet18()
net2 = iresnet34()
net3 = mobilev3()
net4 = mobilev2()

net1.eval()
net2.eval()
net3.eval()
net4.eval()

mbv2 = models.mobilenet_v2(pretrained=True, width_mult=0.5)

Macs1, params1 = profile(net1, inputs=(x,))
Macs2, params2 = profile(net2, inputs=(x,))
Macs3, params3 = profile(net3, inputs=(x,))
Macs4, params4 = profile(net4, inputs=(x,))
Macs5, params5 = profile(mbv2, inputs=(x,))

print('iresnet18-Macs:', int(Macs1 / 1e6), 'M; params:', round(params1 / 1e6, 2), 'M')
print('iresnet34-Macs:', int(Macs2 / 1e6), 'M; params:', round(params2 / 1e6, 2), 'M')
print('mobilev3-Macs:', int(Macs3 / 1e6), 'M; params:', round(params3 / 1e6, 2), 'M')
print('mobilev2-Macs:', int(Macs4 / 1e6), 'M; params:', round(params4 / 1e6, 2), 'M')
print('mbv2-Macs:', int(Macs5 / 1e6), 'M; params:', round(params5 / 1e6, 2), 'M')


