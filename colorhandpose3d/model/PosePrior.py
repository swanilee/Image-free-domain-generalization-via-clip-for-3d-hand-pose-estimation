# PoseNet.py
# Alex Dillhoff (ajdillhoff@gmail.com)
# Model definition for the hand segmentation network.

import torch
import torch.nn as nn
import torch.nn.functional as F
from main.config import cfg

from .TFConv2D import TFConv2D


class PosePrior(nn.Module):
    """Implements the PosePrior architecture.

    The purpose of this network is to estimate the 3D canonical coordinates of
    the hand using 2D heatmaps.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self):
        """Defines and initializes the network."""

        super(PosePrior, self).__init__()
        self.conv_pose_0_1 = nn.Conv2d(21, 32, 3, padding=1)
        self.conv_pose_0_2 = TFConv2D(32, 32, 3, stride=2)
        self.conv_pose_1_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_pose_1_2 = TFConv2D(64, 64, 3, stride=2)
        self.conv_pose_2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_pose_2_2 = TFConv2D(128, 128, 3, stride=2)

        self.fc_rel0 = nn.Linear(2050, 512)
        self.fc_rel1 = nn.Linear(512, 512)
        self.fc_xyz = nn.Linear(512, 63)

    def forward(self, x, hand_side):
        """Forward pass through PosePrior.

        Args:
            x - (batch x 21 x 256 x 256): 2D keypoint heatmaps.
            hand_side - (batch x 2): One-hot vector encoding if the image is
                showing the left or right hand.

        Returns:
            (batch x num_keypoints x 3): xyz coordinates of the hand in
                canonical 3D space.
        """

        s = x.shape
        x = F.leaky_relu(self.conv_pose_0_1(x)) #torch.Size([20, 21, 32, 32])
        x = F.leaky_relu(self.conv_pose_0_2(x)) #torch.Size([20, 32, 16, 16])
        x = F.leaky_relu(self.conv_pose_1_1(x)) #torch.Size([20, 64, 16, 16])
        x = F.leaky_relu(self.conv_pose_1_2(x)) #torch.Size([20, 64, 8, 8])
        x = F.leaky_relu(self.conv_pose_2_1(x)) #torch.Size([20, 128, 8, 8])
        x = F.leaky_relu(self.conv_pose_2_2(x)) #torch.Size([20, 128, 4, 4])

        # Permute before reshaping since these weights are loaded from a TF
        # model.
        x = torch.reshape(x.permute(0, 2, 3, 1), (s[0], -1))

        
        x = torch.cat((x, hand_side), dim=1)

        x = F.leaky_relu(self.fc_rel0(x))
        x = F.leaky_relu(self.fc_rel1(x))
        x = self.fc_xyz(x)

        return torch.reshape(x, (s[0], 21, 3))
