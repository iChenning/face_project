import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F


def main(args):
    # load pc feats
    feats_pc = np.load(args.pth_dir)
    feats_pc = torch.from_numpy(feats_pc)
    if args.is_select:
        feats_pc = feats_pc[:args.nums]

    # load onnx feats
    feats_onnx = np.load(args.onnx_dir)
    feats_onnx = torch.from_numpy(feats_onnx)
    feats_onnx = F.normalize(feats_onnx)
    if args.is_select:
        feats_onnx = feats_onnx[:args.nums]

    # load deploy step1 float feats -- float
    feats_step1 = np.loadtxt(args.convert_step1_dir)
    feats_step1 = feats_step1.reshape(feats_step1.shape[0] // args.embedding_size, args.embedding_size)
    feats_step1 = np.array(feats_step1)
    feats_step1 = torch.from_numpy(feats_step1)
    feats_step1 = F.normalize(feats_step1)
    if args.is_select:
        feats_step1 = feats_step1[:args.nums]

    # load deploy step2 float feats -- quantized
    feats_step2 = np.loadtxt(args.convert_step2_dir)
    feats_step2 = feats_step2.reshape(feats_step2.shape[0] // args.embedding_size, args.embedding_size)
    feats_step2 = np.array(feats_step2)
    feats_step2 = torch.from_numpy(feats_step2)
    feats_step2 = F.normalize(feats_step2)
    if args.is_select:
        feats_step2 = feats_step2[:args.nums]

    # load deploy feats -- deploy
    feats_deploy = []
    f_ = open(args.txt_dir)
    for line in f_.readlines():
        line = line.replace('\n', '')
        words = line.split()
        img_name = os.path.split(words[0])[-1].split('.')[0]
        txt_tmp = os.path.join(args.deploy_dir, img_name + '.txt')
        f_tmp = open(txt_tmp)
        f_deploy_tmp = [float(x) for x in f_tmp.read().split()]
        feats_deploy.append(f_deploy_tmp)
        f_tmp.close()
    f_.close()
    feats_deploy = np.array(feats_deploy)
    feats_deploy = torch.from_numpy(feats_deploy)
    feats_deploy = F.normalize(feats_deploy, dim=1)
    if args.is_select:
        feats_deploy = feats_deploy[:args.nums]

    print('{}{}{}{}'.
          format(''.rjust(40), 'max'.ljust(9), 'mean'.ljust(9), 'min'.ljust(6)))
    # pth&onnx
    sim = torch.sum(feats_pc * feats_onnx, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('pth'.rjust(18), 'onnx'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))

    # onnx&step1
    sim = torch.sum(feats_onnx * feats_step1, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('onnx'.rjust(18), '0_import_model'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))

    # step1&step2
    sim = torch.sum(feats_step1 * feats_step2, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('0_import_model'.rjust(18), '1_quantize_model'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))

    # pth &step2
    sim = torch.sum(feats_pc * feats_step2, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('onnx'.rjust(18), '1_quantize_model'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))

    # step2&deploy
    sim = torch.sum(feats_step2 * feats_deploy, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('1_quantize_model'.rjust(18), 'deploy'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))

    # pth &deploy
    sim = torch.sum(feats_pc * feats_deploy, dim=1)
    print('{} & {}:{:.4f} | {:.4f} | {:.4f}'.
          format('pth'.rjust(18), 'deploy'.ljust(18), sim.max().item(), sim.mean().item(), sim.min().item()))
    for i in range(feats_step2.shape[0]):
        if sim[i] < 0.9:
            print(i, sim[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_dir', default=r'E:\results-1_N\glint360k-iresnet100-3W-withoutflip\query_feats.npy')
    parser.add_argument('--onnx_dir', default=r'E:\results-1_N\glint360k-iresnet100-3W-withoutflip\onnx_feats.npy')
    parser.add_argument('--convert_step1_dir',
                        default=r'E:\results-deploy\glint360k-iresnet100\out_float\att.tensor')
    parser.add_argument('--convert_step2_dir',
                        default=r'E:\results-deploy\glint360k-iresnet100\out_quantized\att.tensor')

    parser.add_argument('--deploy_dir', default=r'E:\results-deploy\glint360k-iresnet100\results_san_query')
    parser.add_argument('--txt_dir', default=r'E:\list-zoo\san_results-single-alig-ID.txt')

    parser.add_argument('--is_select', type=bool, default=False)
    parser.add_argument('--nums', type=int, default=10)
    parser.add_argument('--embedding_size', type=int, default=512)

    args = parser.parse_args()
    main(args)
