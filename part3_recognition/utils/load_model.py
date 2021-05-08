import torch
from collections import OrderedDict


def load_normal(load_path):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if '.module.' in k:
            k = k.replace('.module.', '.')
        new_state_dict[k] = v
    return new_state_dict