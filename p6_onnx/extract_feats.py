import onnxruntime
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

test_trans = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    # load list
    txt_dir = r'E:\list-zoo\san_results-single-alig-ID.txt'
    f_ = open(txt_dir, 'r')
    f_paths = []
    IDs = []
    for line in f_.readlines():
        words = line.replace('\n', '').split()
        f_paths.append(words[0])
        IDs.append(words[1])
    f_.close()

    # onnx测试
    resnet_session = onnxruntime.InferenceSession(r'E:\model-zoo\glint360k-iresnet100\backbone.onnx')
    onnx_feats = []
    for f_path in tqdm(f_paths):
        img = Image.open(f_path).convert("RGB")
        img1 = test_trans(img)
        img1 = torch.unsqueeze(img1, 0)

        # compute ONNX Runtime output prediction
        inputs = {resnet_session.get_inputs()[0].name: to_numpy(img1)}
        outs = resnet_session.run(None, inputs)[0]
        onnx_feats.append(outs)
    onnx_feats = np.concatenate(onnx_feats, axis=0)
    np.save('onnx_feats', onnx_feats)
