import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *



class UNetDS2BN(nn.Module):
    def __init__(self):
        super(UNetDS2BN, self).__init__()
        # 2D Net with 32 channel output
        self.base_filter = 8
        self.conv1_0 = ConvBnReLU(3, 16, 3, 2, 1)
        self.conv2_0 = ConvBnReLU(16, 32, 3, 2, 1)
        self.conv3_0 = ConvBnReLU(32, 64, 3, 2, 1)
        self.conv4_0 = ConvBnReLU(64, 128, 3, 2, 1)

        self.conv0_1 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv0_2 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv1_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv1_2 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv2_1 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)

        self.conv3_1 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv3_2 = ConvBnReLU(64, 64, 3, 1, 1)

        self.conv4_1 = ConvBnReLU(128, 128, 3, 1, 1)
        self.conv4_2 = ConvBnReLU(128, 128, 3, 1, 1)

        self.conv5_0 = ConvTransBnReLU(128, 64, 3, 2, 1, 1)
        self.conv5_1 = ConvBnReLU(128, 64, 3, 1, 1)
        self.conv5_2 = ConvBnReLU(64, 64, 3, 1, 1)

        self.conv6_0 = ConvTransBnReLU(64, 32, 3, 2, 1, 1)
        self.conv6_1 = ConvBnReLU(64, 32, 3, 1, 1)
        self.conv6_2 = ConvBnReLU(32, 32, 3, 1, 1)

        self.conv7_0 = ConvTransBnReLU(32, 16, 3, 2, 1, 1)
        self.conv7_1 = ConvBnReLU(32, 16, 3, 1, 1)
        self.conv7_2 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv8_0 = ConvTransBnReLU(16, 8, 3, 2, 1, 1)
        self.conv8_1 = ConvBnReLU(16, 8, 3, 1, 1)
        self.conv8_2 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv9_0 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv9_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv9_2 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv10_0 = ConvBnReLU(16, 32, 5, 2, 2)  # pad = (kernel-1)/2
        self.conv10_1 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1_0(x)
        x2 = self.conv2_0(x1)
        x3 = self.conv3_0(x2)
        x4 = self.conv4_0(x3)

        y1 = self.conv0_2(self.conv0_1(x))
        y2 = self.conv1_2(self.conv1_1(x1))
        y3 = self.conv2_2(self.conv2_1(x2))
        y4 = self.conv3_2(self.conv3_1(x3))
        y5 = self.conv5_0(self.conv4_2(self.conv4_1(x4)))
        y6 = self.conv6_0(self.conv5_2(self.conv5_1(torch.cat((y5, y4), dim=1))))  # 特征维度(B F H W)
        y7 = self.conv7_0(self.conv6_2(self.conv6_1(torch.cat((y6, y3), dim=1))))
        y8 = self.conv8_0(self.conv7_2(self.conv7_1(torch.cat((y7, y2), dim=1))))
        y9 = self.conv9_0(self.conv8_2(self.conv8_1(torch.cat((y8, y1), dim=1))))
        y10 = self.conv10_0(self.conv9_2(self.conv9_1(y9)))
        y10 = self.conv10_2(self.conv10_1(y10))


        return y10



class RMVS_Regularization(nn.Module):
    def __init__(self, base_channels = 8):
        super(RMVS_Regularization, self).__init__()
        self.base_channels = base_channels
        self.conv_gru1 = ConvGRUCell(32, 16, 3)
        self.conv_gru2 = ConvGRUCell(16, 4, 3)
        self.conv_gru3 = ConvGRUCell(4, 2, 3)
        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, volume_variance):
        depth_costs = []

        b_num, f_num, d_num, img_h, img_w = volume_variance.shape
        state1 = torch.zeros((b_num, 16, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 4, img_h, img_w)).cuda()
        state3 = torch.zeros((b_num, 2, img_h, img_w)).cuda()

        cost_list = volume_variance.chunk(d_num, dim=2)
        cost_list = [cost.squeeze(2) for cost in cost_list]

        for cost in cost_list:
            # Recurrent Regularization
            reg_cost1, state1 = self.conv_gru1(-cost, state1)
            reg_cost2, state2 = self.conv_gru2(reg_cost1, state2)
            reg_cost3, state3 = self.conv_gru3(reg_cost2, state3)
            reg_cost = self.conv2d(reg_cost3)
            depth_costs.append(reg_cost)

        prob_volume = torch.stack(depth_costs, dim=1)
        prob_volume = prob_volume.squeeze(2)

        return prob_volume


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
    

class RMVSNet(nn.Module):
    def __init__(self):
        super(RMVSNet, self).__init__()
        self.feature = UNetDS2BN()
        self.base_channels = 8
        self.conv_gru1 = ConvGRUCell2(32, 16, 3)
        self.conv_gru2 = ConvGRUCell2(16, 4, 3)
        self.conv_gru3 = ConvGRUCell2(4, 2, 3)
        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)


    def compute_depth(self, prob_volume, depth_values=None):
        '''
        prob_volume: 1 x D x H x W
        '''
        B, M, H, W = prob_volume.shape[0], prob_volume.shape[1],prob_volume.shape[2],prob_volume.shape[3]
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        # depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        # depth_range = depth_values.to(prob_volume.device)
        depths = torch.index_select(depth_values, 1, indices.flatten())
        depth_image = depths.view(B, H, W)
        prob_image = probs.view(B, H, W)

        return depth_image, prob_image


    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        depth_costs = []
        b_num, f_num, img_h, img_w = features[0].shape

        state1 = torch.zeros((b_num, self.base_channels * 2, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 4,  img_h, img_w)).cuda()
        state3 = torch.zeros((b_num, 2, img_h, img_w)).cuda()

        for d in range(num_depth):
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
            depth_value = depth_values[:, d:d+1]
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_value)
                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    # TODO: this is only a temporal solution to save memory, better way?
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume

            # aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            volume_variance = volume_variance.squeeze(2)


            # Recurrent Regularization
            reg_cost1, state1 = self.conv_gru1(-volume_variance, state1)
            reg_cost2, state2 = self.conv_gru2(reg_cost1, state2)
            reg_cost3, state3 = self.conv_gru3(reg_cost2, state3)
            reg_cost = self.conv2d(reg_cost3)
            depth_costs.append(reg_cost)
        prob_volume = torch.stack(depth_costs, dim=1)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume = F.softmax(prob_volume, dim=1)
        prob_volume = prob_volume.squeeze(2)
        """
        # max
        depth, photometric_confidence = self.compute_depth(prob_volume, depth_values=depth_values)
        """
        # regression
        depth = depth_regression(prob_volume, depth_values=depth_values)
        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {"depth": depth, "photometric_confidence": photometric_confidence}

class InferenceRMVSNet(nn.Module):
    def __init__(self):
        super(InferenceRMVSNet, self).__init__()
        self.feature = UNetDS2BN()
        self.base_channels = 8
        self.conv_gru1 = ConvGRUCell2(32, 16, 3)
        self.conv_gru2 = ConvGRUCell2(16, 4, 3)
        self.conv_gru3 = ConvGRUCell2(4, 2, 3)
        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)


    def compute_depth(self, prob_volume, depth_values=None):
        '''
        prob_volume: 1 x D x H x W
        '''
        B, M, H, W = prob_volume.shape[0], prob_volume.shape[1],prob_volume.shape[2],prob_volume.shape[3]
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        # depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        # depth_range = depth_values.to(prob_volume.device)
        depths = torch.index_select(depth_values, 1, indices.flatten())
        depth_image = depths.view(B, H, W)
        prob_image = probs.view(B, H, W)

        return depth_image, prob_image


    def forward(self, imgs, proj_matrices, depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        depth_costs = []
        b_num, f_num, img_h, img_w = features[0].shape

        state1 = torch.zeros((b_num, self.base_channels * 2, img_h, img_w)).cuda()
        state2 = torch.zeros((b_num, 4,  img_h, img_w)).cuda()
        state3 = torch.zeros((b_num, 2, img_h, img_w)).cuda()

        # initialize variables
        exp_sum = torch.zeros((b_num, 1, img_h, img_w)).cuda()
        depth_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()
        max_prob_image = torch.zeros((b_num, 1, img_h, img_w)).cuda()

        for d in range(num_depth):
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, 1, 1, 1)
            depth_value = depth_values[:, d:d+1]
            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume
            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_value)
                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    # TODO: this is only a temporal solution to save memory, better way?
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume

            # aggregate multiple feature volumes by variance
            volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
            volume_variance = volume_variance.squeeze(2)


            # Recurrent Regularization
            reg_cost1, state1 = self.conv_gru1(-volume_variance, state1)
            reg_cost2, state2 = self.conv_gru2(reg_cost1, state2)
            reg_cost3, state3 = self.conv_gru3(reg_cost2, state3)
            reg_cost = self.conv2d(reg_cost3)

            prob = reg_cost.exp()

            update_flag_image = (max_prob_image < prob).float()
            new_max_prob_image = update_flag_image * prob + (1 - update_flag_image) * max_prob_image
            # update the best
            # new_depth_image = update_flag_image * depth_value + (1 - update_flag_image) * depth_image
            # update the sum_avg
            new_depth_image = depth_value * prob + depth_image

            max_prob_image = new_max_prob_image
            depth_image = new_depth_image
            exp_sum = exp_sum + prob

            # get output
            # update the best
            # forward_prob_map = (max_prob_image/(exp_sum + 1e-7)).squeeze(1)
            # forward_depth_map = depth_image.squeeze(1)

            # update the sum_avg
        forward_exp_sum = exp_sum + 1e-10
        forward_depth_map = (depth_image / forward_exp_sum).squeeze(1)
        forward_prob_map = (max_prob_image / forward_exp_sum).squeeze(1)

        # upsample depth
        depth_est = torch.unsqueeze(forward_depth_map, dim=1)  # 在第2个维度上扩展
        batch, img_height, img_width = depth_est.shape[0], depth_est.shape[2], depth_est.shape[3]
        depth_est = F.interpolate(depth_est, [img_height*4, img_width*4], mode='bilinear')
        forward_depth_map = torch.squeeze(depth_est, dim=1)

        return {"depth": forward_depth_map, "photometric_confidence": forward_prob_map}
