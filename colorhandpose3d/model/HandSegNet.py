# HandSegNet.py
# Alex Dillhoff (ajdillhoff@gmail.com)
# Model definition for the hand segmentation network.

import torch.nn as nn
import torch.nn.functional as F


class HandSegNet(nn.Module):
    """Implements a hand segmentation network.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self):
        """Defines and initializes the network."""

        super(HandSegNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 128, 3, padding=1)
        self.conv6_1 = nn.Conv2d(128, 512, 1)
        self.conv6_2 = nn.Conv2d(512, 2, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """Forward pass through the hand segementation network.

        Args:
            x - [batch x 256 x 256 x 3]: Color image containing a hand.

        Returns:
            [batch x 256 x 256] hand segmentation mask.
        """

        s = x.shape
        x = F.leaky_relu(self.conv1_1(x)) # 1
        x = F.leaky_relu(self.conv1_2(x)) # 2
        x = self.pool(x)          # 3
        x = F.leaky_relu(self.conv2_1(x)) # 4
        x = F.leaky_relu(self.conv2_2(x)) # 5
        x = self.pool(x)          # 6
        x = F.leaky_relu(self.conv3_1(x)) # 7
        x = F.leaky_relu(self.conv3_2(x)) # 8
        x = F.leaky_relu(self.conv3_3(x)) # 9
        x = F.leaky_relu(self.conv3_4(x)) # 10
        x = self.pool(x)          # 11
        x = F.leaky_relu(self.conv4_1(x)) # 12
        x = F.leaky_relu(self.conv4_2(x)) # 13
        x = F.leaky_relu(self.conv4_3(x)) # 14
        x = F.leaky_relu(self.conv4_4(x)) # 15
        x = F.leaky_relu(self.conv5_1(x)) # 16
        x = F.leaky_relu(self.conv5_2(x))
        x = F.leaky_relu(self.conv6_1(x))
        x = self.conv6_2(x)         # 17
        x = F.interpolate(x, s[2], mode='bilinear', align_corners=False) # 18

        return x
