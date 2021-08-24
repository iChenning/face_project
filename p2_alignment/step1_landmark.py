from __future__ import print_function
import cv2
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils.config import cfg_mnet
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
from utils.load_model import load_normal

cudnn.benchmark = True


def save_image(dets, vis_thres, img_raw, save_folder, img_name, save_all=False):
    count_ = 0
    line = ''
    for b in dets:
        if b[4] < vis_thres:
            continue
        count_ += 1
        line += ' ' + str(int(b[0])) + ' ' + str(int(b[1])) + ' ' + str(int(b[2])) + ' ' + str(int(b[3]))
        text = "{:.4f}".format(b[4])
        b4 = b[4]
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        line += ' ' + str(int(b[5])) + ' ' + str(int(b[6])) + \
                ' ' + str(int(b[7])) + ' ' + str(int(b[8])) + \
                ' ' + str(int(b[9])) + ' ' + str(int(b[10])) + \
                ' ' + str(int(b[11])) + ' ' + str(int(b[12])) + \
                ' ' + str(int(b[13])) + ' ' + str(int(b[14])) + \
                ' ' + str(int(b4 * 100))

    # save image
    name = os.path.join(os.path.join(save_folder, 'pictures'), img_name)
    dirname = os.path.dirname(name)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if save_all:
        cv2.imwrite(name, img_raw)
    else:
        if count_ > 0:
            cv2.imwrite(name, img_raw)
    line_write = os.path.split(img_name)[-1] + ' ' + str(count_) + line + '\n'
    return line_write


def main(args):
    # net and load
    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase='test')
    state_dict = load_normal(args.trained_model)
    net.load_state_dict(state_dict)
    net = net.cuda()

    # data
    with open(args.txt_dir, 'r') as f_:
        f_paths = f_.read().split()
    f_paths.sort()  # take care

    # run: detect and align
    _t = {'forward_pass': Timer(), 'misc': Timer()}
    net.eval()

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    f_ = open(os.path.join(args.save_folder, 'vis_bbox.txt'), 'w')

    for i, f_path in enumerate(f_paths):
        img_name = f_path[f_path.find('datasets') + 9:]
        # load image
        img_raw = cv2.imread(f_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)

        # scale
        im_height, im_width, _ = img.shape
        bbox_scale = torch.Tensor([im_width, im_height] * 2).cuda()
        landms_scale = torch.Tensor([im_width, im_height] * 5).cuda()

        # image to tensor
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()

        # forward
        _t['forward_pass'].tic()
        loc, conf, landms = net(img)
        _t['forward_pass'].toc()

        # landmarks, boxes, scores
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.cuda()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * bbox_scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        landms = landms * landms_scale
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]
        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        # --------------------------------------------------------------------
        save_name = os.path.join(args.save_folder, 'txt', img_name)[:-4] + '.txt'
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d}'
              ' forward_pass_time: {:.4f}s'
              ' misc: {:.4f}s'
              ' img_shape:{:}'.format
              (i + 1, len(f_paths),
               _t['forward_pass'].average_time,
               _t['misc'].average_time,
               img.shape))

        # save bbox-image
        line_write = save_image(dets, args.vis_thres, img_raw, args.save_folder, img_name, save_all=args.save_image_all)
        f_.write(line_write)
        f_.flush()
    f_.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--trained_model', type=str, default=r'E:\model-zoo\widerface-mobile0.25\mobile0.25_150.pth')

    parser.add_argument('--txt_dir', type=str, default=r'E:\list-zoo\real.txt')
    parser.add_argument('--save_folder', type=str, default=r'E:\results_save\part2_alignment\real')
    parser.add_argument('--save_image_all', default=True)

    parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=500, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.25, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=100, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.75, type=float, help='visualization_threshold')

    args = parser.parse_args()

    main(args)
