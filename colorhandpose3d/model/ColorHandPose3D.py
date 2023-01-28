import os

import torch

from .HandSegNet import HandSegNet
from .PoseNet import PoseNet
from .PosePrior import PosePrior
from .ViewPoint import ViewPoint
from ..utils.transforms import *
from main.config import cfg

class ColorHandPose3D(torch.nn.Module):
    """ColorHandPose3D predicts the 3D joint location of a hand given the
    cropped color image of a hand."""

    def __init__(self, crop_size=None, num_keypoints=None): #(self, weight_path=None, crop_size=None, num_keypoints=None):
        super(ColorHandPose3D, self).__init__()
        self.handsegnet = HandSegNet()
        self.posenet = PoseNet()
        self.poseprior = PosePrior()
        self.viewpoint = ViewPoint()

        if crop_size is None:
            self.crop_size = 256
        else:
            self.crop_size = crop_size

        if num_keypoints is None:
            self.num_keypoints = 21
        else:
            self.num_keypoints = num_keypoints

        # # Load weights
        # if weight_path is not None:
        #     self.handsegnet.load_state_dict(
        #             torch.load(os.path.join(weight_path, 'handsegnet.pth.tar')))
        #     self.posenet.load_state_dict(
        #             torch.load(os.path.join(weight_path, 'posenet.pth.tar')))
        #     self.poseprior.load_state_dict(
        #             torch.load(os.path.join(weight_path, 'poseprior.pth.tar')))
        #     self.viewpoint.load_state_dict(
        #             torch.load(os.path.join(weight_path, 'viewpoint.pth.tar')))

    def forward(self, x, hand_sides, mixed_clipf=None, nomal_aug_x=None, iter=None):
        """Forward pass through the network.

        Args:
            x - Tensor (B x C x H x W): Batch of images.
            hand_sides - Tensor (B x 2): One-hot vector indicating if the hand
                is left or right.

        Returns:
            coords_xyz_rel_normed (B x N_k x 3): Normalized 3D coordinates of
                the joints, where N_k is the number of keypoints.
        """
        '''
        # Segment the hand
        hand_scoremap = self.handsegnet.forward(x)

        # Calculate single highest scoring object
        hand_mask = single_obj_scoremap(hand_scoremap, self.num_keypoints)

        # crop and resize
        centers, _, crops = calc_center_bb(hand_mask)
        crops = crops.to(torch.float32)

        crops *= 1.25
        scale_crop = torch.min(
                torch.max(self.crop_size / crops,
                    torch.tensor(0.25, device=x.device)),
                torch.tensor(5.0, device=x.device))
        image_crop = crop_image_from_xy(x, centers, self.crop_size, scale_crop)
        '''
        if cfg.Contrastive and mixed_clipf!=None:
            if cfg.Ablation_normal_aug or cfg.Ablation_normal_aug_clip:
                peclr_aug_img = x
                x = [peclr_aug_img, nomal_aug_x]
            # detect 2d keypoints
            keypoints_scoremap = self.posenet(x, mixed_clipf, iter) #image_crop

            encoding = keypoints_scoremap[1]
            keypoints_scoremap = keypoints_scoremap[0]
            
        else:
            # detect 2d keypoints
            keypoints_scoremap = self.posenet(x, mixed_clipf) #image_crop

        if cfg.Contrastive and mixed_clipf!=None and cfg.Cont_add==True:
            hand_sides = torch.cat([hand_sides, hand_sides],dim=0)

        # estimate 3d pose
        coord_can = self.poseprior(keypoints_scoremap, hand_sides)

        rot_params = self.viewpoint(keypoints_scoremap, hand_sides)

        # get normalized 3d coordinates
        rot_matrix = get_rotation_matrix(rot_params)
        cond_right = torch.eq(torch.argmax(hand_sides, 1), 1)
        cond_right_all = torch.reshape(cond_right, [-1, 1, 1]).repeat(1, self.num_keypoints, 3)
        coords_xyz_can_flip = flip_right_hand(coord_can, cond_right_all)
        coords_xyz_rel_normed = coords_xyz_can_flip @ rot_matrix

        # flip left handed inputs wrt to the x-axis for Libhand compatibility.
        coords_xyz_rel_normed = flip_left_hand(coords_xyz_rel_normed, cond_right_all)

        # scale heatmaps
        keypoints_scoremap = F.interpolate(keypoints_scoremap,
                                           self.crop_size,
                                           mode='bilinear',
                                           align_corners=False)

        if cfg.Contrastive and mixed_clipf!=None:
            keypoints_scoremap = [keypoints_scoremap, encoding]
        return coords_xyz_rel_normed, keypoints_scoremap #, image_crop, centers, scale_crop

    