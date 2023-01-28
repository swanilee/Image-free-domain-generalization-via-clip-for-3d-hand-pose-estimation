# ViewPoint.py
# Alex Dillhoff (ajdillhoff@gmail.com)
# Model definition for the viewpoint prediction network.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .TFConv2D import TFConv2D


class ViewPoint(nn.Module):
    """Implements the ViewPoint architecture.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self):
        """Defines and initializes the network."""

        super(ViewPoint, self).__init__()

        self.conv_vp_0_1 = TFConv2D(21, 64, 3)
        self.conv_vp_0_2 = TFConv2D(64, 64, 3, stride=2)
        self.conv_vp_1_1 = TFConv2D(64, 128, 3)
        self.conv_vp_1_2 = TFConv2D(128, 128, 3, stride=2)
        self.conv_vp_2_1 = TFConv2D(128, 256, 3)
        self.conv_vp_2_2 = TFConv2D(256, 256, 3, stride=2)

        self.fc_vp0 = nn.Linear(4098, 256)
        self.fc_vp1 = nn.Linear(256, 128)

        self.fc_vp_ux = nn.Linear(128, 1)
        self.fc_vp_uy = nn.Linear(128, 1)
        self.fc_vp_uz = nn.Linear(128, 1)


    def forward(self, x, hand_side):
        """Forward pass through the ViewPoint network.

        Args:
            x - (batch x 21 x 256 x 256): 2D keypoint heatmaps.
            hand_side - (batch x 2): One-hot vector encoding if the image is
                showing the left or right hand.

        Returns:
            (batch x 3): axis-angle representation with encoded angle.
        """

        s = x.shape
        x = F.leaky_relu(self.conv_vp_0_1(x))
        x = F.leaky_relu(self.conv_vp_0_2(x))

        x = F.leaky_relu(self.conv_vp_1_1(x))
        x = F.leaky_relu(self.conv_vp_1_2(x))

        x = F.leaky_relu(self.conv_vp_2_1(x))
        x = F.leaky_relu(self.conv_vp_2_2(x))

        # Permute before reshaping since these weights are loaded from a TF
        # model.
        x = torch.reshape(x.permute(0, 2, 3, 1), (s[0], -1))
        x = torch.cat((x, hand_side), dim=1)

        x = F.leaky_relu(self.fc_vp0(x))
        x = F.leaky_relu(self.fc_vp1(x))

        ux = self.fc_vp_ux(x)
        uy = self.fc_vp_uy(x)
        uz = self.fc_vp_uz(x)

        return torch.cat((ux, uy, uz), 1)
