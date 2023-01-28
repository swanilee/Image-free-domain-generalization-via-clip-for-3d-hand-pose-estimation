# PoseNet.py
# Alex Dillhoff (ajdillhoff@gmail.com)
# Model definition for the 2D heatmap prediction network.

import torch
import torch.nn as nn
import torch.nn.functional as F
from main.config import cfg


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out



class PoseNet(nn.Module):
    """Implements the PoseNet architecture.

    This architecture is defined in:
        Zimmermann, C., & Brox, T. (2017).
        Learning to Estimate 3D Hand Pose from Single RGB Images.
        Retrieved from http://arxiv.org/abs/1705.01389
    """

    def __init__(self):
        """Defines and initializes the network."""

        super(PoseNet, self).__init__()
        # Stage 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, 3, padding=1)
        if cfg.baseline == True or cfg.frontClipFeat==True or cfg.adain1==True or cfg.adain2==True or cfg.adain3==True:
            self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1) #256->512
        elif cfg.MultiplyFeat==True or cfg.AddFeat==True or cfg.AvgSum==True or cfg.Contrastive==True :
            self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1) #256->512
        else:
            self.conv4_1 = nn.Conv2d(512, 512, 3, padding=1) #512->512
            
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_5 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_7 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5_1 = nn.Conv2d(128, 512, 1)
        self.conv5_2 = nn.Conv2d(512, 21, 1)
        self.pool = nn.MaxPool2d(2, 2)

        # Stage 2
        self.conv6_1 = nn.Conv2d(149, 128, 7, padding=3)
        self.conv6_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv6_6 = nn.Conv2d(128, 128, 1)
        self.conv6_7 = nn.Conv2d(128, 21, 1)

        # Stage 3
        self.conv7_1 = nn.Conv2d(149, 128, 7, padding=3)
        self.conv7_2 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_3 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_4 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_5 = nn.Conv2d(128, 128, 7, padding=3)
        self.conv7_6 = nn.Conv2d(128, 128, 1)
        self.conv7_7 = nn.Conv2d(128, 21, 1)

        # Clip Concat
        self.dconv1 = nn.ConvTranspose2d(512,512,4, stride=2, padding=1, bias=False) #torch.Size([20, 512, 4, 4])
        self.bn1 = nn.BatchNorm2d(512)
        self.dconv2 = nn.ConvTranspose2d(512, 512,4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.dconv3 = nn.ConvTranspose2d(512,512, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.dconv4 = nn.ConvTranspose2d(512,512, 4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.dconv5 = nn.ConvTranspose2d(512,256, 4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.cconv6 = nn.Conv2d(512, 256, 1, stride=1, padding=1)

        #AdaIN
        self.norm1 = ADAIN(128, 512)
        self.norm2 = ADAIN(256, 512)
        self.norm3 = ADAIN(128, 512)


        if cfg.Contrastive:
            #Linear layers for below projection
            self.proj_linear1 = nn.Linear(21504, 10752)
            self.proj_linear2 = nn.Linear(10752, 5376)
            self.proj_linear3 = nn.Linear(5376, 2048)

            #ProjectionHead with ContrastiveLearning    
            self.projection_head = nn.Sequential(
                nn.Linear(
                    512, #self.config.projection_head_input_dim,
                    512, #self.config.projection_head_hidden_dim,
                    bias=True,
                ),
                nn.BatchNorm1d(512), #self.config.projection_head_hidden_dim
                nn.ReLU(),
                nn.Linear(
                    512, #self.config.projection_head_hidden_dim
                    128, #self.config.output_dim,
                    bias=False,
                ),
            )

            #cat version
            self.cat_cnn1 =  nn.Conv2d(256, 512, 1, stride=2)
            self.cat_cnn2 =  nn.Conv2d(512, 1024, 1, stride=2)
            self.cat_maxpool = nn.MaxPool2d(8)

            self.cat_fc1 = nn.Linear(1024, 512)
            self.cat_fc2 = nn.Linear(1024+512, 512)

            self.ablation_add_fc1 =  nn.Linear(1024, 512)
        '''
        # Mediate feature
        self.convc_1 = nn.Conv2d(128, 64, 1)
        self.bnc_1 = nn.BatchNorm2d(64)
        self.convc_2 = nn.Conv2d(64, 21, 3, padding=1)
        self.bnc_2 = nn.BatchNorm2d(21)
        '''

    def forward(self, x, mixed_clipf=None, iter=None):
        """Forward pass through PoseNet.

        Args:
            x - [batch x 3 x 256 x 256]: Color image containing a cropped
                image of the hand.

        Returns:
            [batch x 21 x 32 x 32] hand keypoint heatmaps.
        """

        if cfg.Ablation_normal_aug or cfg.Ablation_normal_aug_clip:
            x = torch.cat([x[0], x[1]], dim=0) #[peclr_aug_img, nomal_aug_x]

        if cfg.frontClipFeat == False:  
            # Stage 1
            x = F.leaky_relu(self.conv1_1(x)) # 1 
            x = F.leaky_relu(self.conv1_2(x)) # 2 torch.Size([20, 64, 256, 256])
            x = self.pool(x)          # 3         torch.Size([20, 64, 128, 128])
            x = F.leaky_relu(self.conv2_1(x)) # 4
            x = F.leaky_relu(self.conv2_2(x)) # 5 torch.Size([20, 128, 128, 128])
            x = self.pool(x)          # 6         torch.Size([20, 128, 64, 64])
            if cfg.adain1 == True: ##torch.Size([20, 128, 64, 64])
                x = self.norm1(x, mixed_clipf)
            x = F.leaky_relu(self.conv3_1(x)) # 7
            x = F.leaky_relu(self.conv3_2(x)) # 8
            x = F.leaky_relu(self.conv3_3(x)) # 9
            x = F.leaky_relu(self.conv3_4(x)) # 10 torch.Size([20, 256, 64, 64])
            x = self.pool(x)          # 11         torch.Size([20, 256, 32, 32]) Layer 11
            if cfg.adain2 == True: #torch.Size([20, 256, 32, 32])
                x = self.norm2(x, mixed_clipf)
            
            # if cfg.Ablation_normal_aug:
            #     nax = x[x.shape[0]//2:]
            #     x = x[:x.shape[0]//2]
                

        if cfg.baseline == False and cfg.adain1 ==False and cfg.adain2 ==False and cfg.adain3 ==False and mixed_clipf!=None:
            '''
            mixed_clipf shape: [B, 512]
            x shape: [B, 256, 32, 32] -> [B, 262144]
            '''
            # if cfg.Ablation_normal_aug or cfg.Ablation_normal_aug_clip:
            #     mnax = self.cat_cnn1(nax.clone()) #nn.Conv2d(256, 512, 1, stride=2)
            #     mnax = self.cat_cnn2(mnax)
            #     mnax = self.cat_maxpool(mnax)
            #     mnax = torch.flatten(mnax, start_dim=1)
            #     only_feat_mnax = self.cat_fc1(mnax)
                            
            mx = self.cat_cnn1(x.clone())  #nn.Conv2d(256, 512, 1, stride=2)
            mx = self.cat_cnn2(mx) #nn.Conv2d(512, 1024, 1, stride=2)
            mx = self.cat_maxpool(mx) #MaxPool(8) -> torch.Size([B, 1024, 8, 8])
            mx = torch.flatten(mx, start_dim=1) #torch.Size([B, 1024, 1, 1]) -> #torch.Size([B, 1024])
            only_feat_mx = self.cat_fc1(mx)  #nn.Linear(1024, 512)

            if cfg.Ablation_normal_aug or cfg.Ablation_normal_aug_clip:
                mx = mx[x.shape[0]//2:].clone()

                peclr_aug_feat = only_feat_mx[:x.shape[0]//2]
                only_feat_mx = only_feat_mx[x.shape[0]//2:]
                x = x[x.shape[0]//2:]
                

            if cfg.Ablation_add:
                mx = self.ablation_add_fc1(mx)
                clipCatFeat_mx = mx + mixed_clipf
            else:
                clipCatFeat_mx = torch.cat([mx, mixed_clipf],dim=1)
                clipCatFeat_mx = self.cat_fc2(clipCatFeat_mx) #nn.Linear(1024+512, 512)
            

            if cfg.Ablation_normal_aug:
                mx = torch.cat([only_feat_mx, peclr_aug_feat],dim=0)
            elif cfg.Ablation_normal_aug_clip:
                if iter % 2 == 0:
                    mx = torch.cat([only_feat_mx, peclr_aug_feat],dim=0)
                else:
                    mx = torch.cat([only_feat_mx, clipCatFeat_mx],dim=0)
            else:
                mx = torch.cat([only_feat_mx, clipCatFeat_mx],dim=0)


            '''
            ####============================= Clip feat +concat
            mx = mixed_clipf.unsqueeze(-1)
            mx = mx.unsqueeze(-1)
            mx = F.relu(self.bn1(self.dconv1(mx)))
            mx = F.relu(self.bn2(self.dconv2(mx)))
            mx = F.relu(self.bn3(self.dconv3(mx)))
            mx = F.relu(self.bn4(self.dconv4(mx)))
            mx = F.relu(self.bn5(self.dconv5(mx))) #torch.Size([64, 512, 16, 16])

            if cfg.frontClipFeat == False and cfg.AddFeat==False and cfg.MultiplyFeat==False and cfg.AvgSum==False: #x => torch.Size([64, 256, 32, 32])/ mx => torch.Size([64, 256, 32, 32])
                mxc = mx.clone()

                if cfg.Cont_add:
                    mxc = x + mxc
                    mx = torch.cat([x, mxc],dim=0) #torch.Size([20, 512, 32, 32])
                else:
                    print(mxc.shape)
                # mx = torch.cat([x, mx],dim=1) #torch.Size([20, 512, 32, 32])

            elif cfg.MultiplyFeat == True: #x => torch.Size([64, 256, 32, 32])/ mx => torch.Size([64, 256, 32, 32])
                mx = x * mx #torch.Size([20, 256, 32, 32])
            elif cfg.AddFeat == True: #x => torch.Size([64, 256, 32, 32])/ mx => torch.Size([64, 256, 32, 32])
                mx = x + mx #torch.Size([20, 256, 32, 32])
            elif cfg.AvgSum == True: #x => torch.Size([64, 256, 32, 32])/ mx => torch.Size([64, 256, 32, 32])
                mx = (x + mx) /2 #torch.Size([20, 256, 32, 32])
            x = mx
            '''
            
            ####=============================


        x = F.leaky_relu(self.conv4_1(x)) # 12 torch.Size([20, 512, 32, 32])    x or mx
        x = F.leaky_relu(self.conv4_2(x)) # 13
        x = F.leaky_relu(self.conv4_3(x)) # 14 torch.Size([20, 256, 32, 32])
        x = F.leaky_relu(self.conv4_4(x)) # 15
        x = F.leaky_relu(self.conv4_5(x)) # 16
        x = F.leaky_relu(self.conv4_6(x)) # 17
        encoding = F.leaky_relu(self.conv4_7(x)) # 18 torch.Size([20, 128, 32, 32])
        if cfg.adain3 == True: #torch.Size([20, 256, 32, 32])
            encoding = self.norm3(encoding, mixed_clipf)

        x = F.leaky_relu(self.conv5_1(encoding))
        scoremap = self.conv5_2(x) # torch.Size([20, 21, 32, 32])

        # Stage 2
        x = torch.cat([scoremap, encoding], dim=1) #torch.Size([20, 149, 32, 32])
        x = F.leaky_relu(self.conv6_1(x)) #torch.Size([20, 128, 32, 32])
        x = F.leaky_relu(self.conv6_2(x))
        x = F.leaky_relu(self.conv6_3(x))
        x = F.leaky_relu(self.conv6_4(x))
        x = F.leaky_relu(self.conv6_5(x))
        x = F.leaky_relu(self.conv6_6(x))
        scoremap = self.conv6_7(x) #torch.Size([20, 21, 32, 32])

        # Stage 3
        x = torch.cat([scoremap, encoding], dim=1) #torch.Size([20, 149, 32, 32])
        x = F.leaky_relu(self.conv7_1(x)) #torch.Size([20, 128, 32, 32])
        x = F.leaky_relu(self.conv7_2(x))
        x = F.leaky_relu(self.conv7_3(x))
        x = F.leaky_relu(self.conv7_4(x))
        x = F.leaky_relu(self.conv7_5(x))
        x = F.leaky_relu(self.conv7_6(x))
        x = self.conv7_7(x) #torch.Size([20, 21, 32, 32])

        if cfg.Contrastive and mixed_clipf!=None:
            # cx = torch.flatten(mx.clone(), start_dim=1)
            # cx = self.proj_linear1(cx)
            # cx = self.proj_linear2(cx)
            # cx = self.proj_linear3(cx)
            encoding = self.projection_head(mx)
            x = [x, encoding]


        # if cfg.Contrastive and mixed_clipf!=None:
            # cx = self.convc_1(encoding.clone())
            # cx = self.convc_2(cx)
            # cx = torch.flatten(cx, start_dim=1)
            # cx = self.proj_linear1(cx)
            # cx = self.proj_linear2(cx)
            # cx = self.proj_linear3(cx)
            # cx = self.projection_head(cx)
            # x = [x, cx]

        return x
