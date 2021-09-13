import torch
from torch import nn
import math

__all__ = ['cosloss', 'arcloss', 'arccosloss']


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


# class ArcFace(nn.Module):
#     def __init__(self, s=64.0, m=0.5):
#         super(ArcFace, self).__init__()
#         self.s = s
#         self.m = m
#
#     def forward(self, cosine: torch.Tensor, label):
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
#         m_hot.scatter_(1, label[index, None], self.m)
#         cosine.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
#         cosine.acos_()
#         # cosine = 2*torch.arctan(torch.sqrt((1. - cosine*cosine)/(1. + cosine)))
#         cosine[index] += m_hot
#         cosine.cos_().mul_(self.s)
#         if torch.isnan(cosine).sum() > 0:
#             index = torch.where(torch.isnan(cosine))
#             print(3, index)
#             print(3, cosine[index])
#             assert False, torch.isnan(cosine).sum()
#         return cosine


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.50):
        """ArcFace formula:
            cos(m + theta) = cos(m)cos(theta) - sin(m)sin(theta)
        Note that:
            0 <= m + theta <= Pi
        So if (m + theta) >= Pi, then theta >= Pi - m. In [0, Pi]
        we have:
            cos(theta) < cos(Pi - m)
        So we can use cos(Pi - m) as threshold to check whether
        (m + theta) go out of [0, Pi]
        Args:
            embedding_size: usually 128, 256, 512 ...
            class_num: num of people when training
            s: scale, see normface https://arxiv.org/abs/1704.06369
            m: margin, see SphereFace, CosFace, and ArcFace paper
        https://github.com/siriusdemon/Build-Your-Own-Face-Model/blob/master/recognition/model/metric.py
        """
        super().__init__()
        self.s = s
        self.m = m

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, label):
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        # update y_i by phi in cosine
        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s


def arcloss(s=64., m=0.5):
    return ArcFace(s, m)


class ArcCosFace(nn.Module):
    def __init__(self, s=64.0, m=0.5, m_cos=0.4):
        super(ArcCosFace, self).__init__()
        self.s = s
        self.m = m
        self.m_cos = m_cos

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        # cosine.clamp_(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        
        m_cos_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_cos_hot.scatter_(1, label[index, None], self.m_cos)
        cosine[index] -= m_cos_hot

        if torch.isnan(cosine).sum() > 0:
            index = torch.where(torch.isnan(cosine))
            print(3, index)
            print(3, cosine[index])
            assert False, torch.isnan(cosine).sum()
        return cosine


def arccosloss(s=64., m=0.5, m_cos=0.4):
    return ArcCosFace(s, m, m_cos)