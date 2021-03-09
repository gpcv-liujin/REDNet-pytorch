
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from datasets import *
from datasets.preprocess import write_cam, write_pfm, save_pfm
from utils import *

import numpy as np
import argparse, os, time, gc, cv2
import matplotlib.pyplot as plt
from PIL import Image
import os.path as osp
from collections import *
import sys

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict WHU.')
parser.add_argument('--model', default='rednet', help='select model', choices=['mvsnet', 'rmvsnet', 'rednet'])
parser.add_argument('--dataset', default='whu_eval', help='select dataset')

parser.add_argument('--testpath', type=str, help='path to root directory.',
                    default="X:/liujin_densematching/MVS_traindata/meitan_RS/test_largeimage")
parser.add_argument('--savepath', type=str, help='path to save depth maps.',
                    default="X:/liujin_densematching/MVS_traindata/meitan_RS/test_largeimage/depths_rednet")
parser.add_argument('--ckpt', type=str, help='the path for pre-trained model.',
                    default='./checkpoints/whu_rednet/model_000005.ckpt')

#test parameters
parser.add_argument('--view_num', type=int, default=3, help='num of candidate views')
parser.add_argument('--numdepth', type=int, default=256, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=5376, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=5376, help='Maximum image height')
parser.add_argument('--fext', type=str, default='.png', help='Type of images.')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=0.5, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')

args = parser.parse_args()


def main(args):
    # dataset, dataloader
    MVSPredDataset = find_dataset_def(args.dataset)
    testset = MVSPredDataset(args.testpath, "val", args.view_num, args, args.numdepth, args.interval_scale)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=4, drop_last=False)

    # build model
    if args.model == 'mvsnet':
        from models.mvsnet import InferenceMVSNet
        model = InferenceMVSNet(refine=False)
    if args.model == 'rmvsnet':
        from models.rmvsnet import InferenceRMVSNet
        model = InferenceRMVSNet()
    if args.model == 'rednet':
        from models.rednet import InferenceREDNet
        model = InferenceREDNet()

    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.ckpt))
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict['model'])
    model.eval()

    with torch.no_grad():
        # create output folder
        output_folder = os.path.join(args.savepath)
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        step = 0
        list_index = []
        f = open(output_folder + '/index.txt', "w")

        first_start_time = time.time()

        for batch_idx, sample in enumerate(test_loader):
            print("=====MVS: {} / {}".format(batch_idx, len(test_loader)))

            start_time = time.time()
            sample_cuda = dict2cuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            depth_est = outputs["depth"].cpu().numpy()
            photometric_confidence = outputs["photometric_confidence"].cpu().numpy()
            duration = time.time() - start_time

            # save results
            depth_est = np.float32(np.squeeze(tensor2numpy(depth_est)))
            prob = np.float32(np.squeeze(tensor2numpy(photometric_confidence)))

            ref_image = np.squeeze(tensor2numpy(sample["outimage"]))
            ref_cam = np.squeeze(tensor2numpy(sample["outcam"]))
            out_location = np.squeeze(sample["outlocation"])
            out_index = out_location[0]
            vid = sample["out_view"][0]
            name = sample["out_name"][0]

            # paths
            if not os.path.exists(output_folder + '/color/'):
                os.mkdir(output_folder + '/color/')
            init_depth_map_path = output_folder + ('/%s_init.pfm' % name)
            prob_map_path = output_folder + ('/%s_prob.pfm' % name)
            out_ref_image_path = output_folder + ('/%s.jpg' % name)
            out_ref_cam_path = output_folder + ('/%s.txt' % name)

            if name not in list_index:
                # if (list_index.index(out_index)==-1):
                out_location[3] = str(args.max_w)
                out_location[4] = str(args.max_h)
                list_index.append(name)
                for word in out_location:
                    f.write(str(word) + ' ')
                f.write('\n')

            # save output
            save_pfm(init_depth_map_path, depth_est)
            save_pfm(prob_map_path, prob)
            plt.imsave(out_ref_image_path, ref_image, format='jpg')
            write_cam(out_ref_cam_path, ref_cam, out_location)

            # color output
            size1 = len(depth_est)
            size2 = len(depth_est[1])
            e = np.ones((size1, size2), dtype=np.float32)
            out_init_depth_image = e * 360000 - depth_est
            #out_init_depth_image = depth_est
            plt.imsave(output_folder + ('/color/%s_init.png' % name), out_init_depth_image, format='png')
            plt.imsave(output_folder + ('/color/%s_prob.png' % name), prob, format='png')

            del outputs, sample_cuda

            step = step + 1
            print('depth inference {} finished, image {} finished, ({:3f} sec/step)'.format(step, name, duration))

        f.close()
        print("final, total_cnt = {}, total_time = {:3f}".format(step, time.time() - first_start_time))
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    with torch.no_grad():
        main(args)