from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from datasets.data_io import *
from datasets.preprocess import *
from imageio import imread, imsave, imwrite


# the WHU dataset preprocessed by Jin Liu (only for predict)

class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, args, ndepths=192, interval_scale=1.06, **kwargs):

        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.args = args
        self.view_num = view_num
        self.counter = 0
        assert self.mode in ["train", "val", "test"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)
        self.ndepths = ndepths
        self.interval_scale = interval_scale

    def build_list(self):
        # Prepare all training samples
        sample_list = gen_predict_mvs_list(self.data_folder, self.view_num, self.args.fext)

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_cam_whu(self, file, ndepths, interval_scale=1):
        """ read camera txt file (XrightYup，Twc)"""
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Twc

        # if cam ori is XrightYup
        O = np.eye(3, dtype=np.float32)
        O[1, 1] = -1
        O[2, 2] = -1
        R = extrinsics[0:3, 0:3]
        R2 = np.matmul(R, O)
        extrinsics[0:3, 0:3] = R2

        extrinsics = np.linalg.inv(extrinsics)  # Tcw
        cam[0, :, :] = extrinsics

        for i in range(0, 13):
            pera[0][i] = words[17 + i]

        # K
        f = pera[0][0]
        x0 = pera[0][1]
        y0 = pera[0][2]
        cam[1][0][0] = f
        cam[1][1][1] = f
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        # depth range
        cam[1][3][0] = np.float32(pera[0][3])  # start
        cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
        cam[1][3][3] = np.float32(pera[0][4])  # end

        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

        if acturald > ndepths:
            scale = acturald / np.float32(ndepths)
            cam[1][3][1] = cam[1][3][1] * scale
            acturald = ndepths
        cam[1][3][2] = acturald
        # cam[1][3][2] = ndepths
        location = words[23:30]

        return cam, location

    def read_img(self, filename):
        img = Image.open(filename)

        return img

    def __getitem__(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######
        outimage = None
        outcam = None
        outlocation = None

        centered_images = []
        proj_matrices = []
        depth_values = None


        for view in range(self.view_num):
            # Images
            image = self.read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            cam, location = self.read_cam_whu(data[2 * view + 1], self.ndepths, self.interval_scale)
            location.append(str(self.args.resize_scale))

            # determine a proper scale to resize input
            scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
            # crop to fit network
            croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h,
                                                  max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            if view == 0:
                outimage = image
                outcam = cam
                outlocation = location
                depth_min = cam[1][3][0]
                depth_interval = cam[1][3][1]
                new_ndepths = cam[1][3][2]


            # scale cameras for building cost volume
            scaled_cam = scale_camera(cam, scale=self.args.sample_scale)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(center_image(image))

        # Depth
        # print(new_ndepths)
        depth_values = np.arange(np.float(depth_min), np.float(depth_interval * (new_ndepths - 0.5) + depth_min),
                                 np.float(depth_interval),
                                 dtype=np.float32)

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        name = os.path.splitext(os.path.basename(data[0]))[0]
        vid = os.path.dirname(data[0]).split("/")[-1]

        return {"imgs": centered_images,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outcam,
                "outlocation": outlocation,
                "out_name": name,
                "out_view": vid}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 128)
