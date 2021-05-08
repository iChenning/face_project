import torch
from torch import nn

__all__ = ['cosloss', 'arcloss']


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


def cosloss(s=64., m=0.4):
    return CosFace(s, m)


import numpy as np


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        cosine.acos_()
        # cosine_np = cosine.data.cpu().numpy()
        # cosine_np = np.arccos(cosine_np)
        # cosine.data = torch.from_numpy(cosine_np).to(device=cosine.device)
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        if torch.isnan(cosine).sum() > 0:
            index = torch.where(torch.isnan(cosine))
            print(3, index)
            print(3, cosine[index])
            assert False, torch.isnan(cosine).sum()
        return cosine


def arcloss(s=64., m=0.5):
    return ArcFace(s, m)
