import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from colorhandpose3d.utils.param import * 
from config import cfg
import pickle

from colorhandpose3d.model.ColorHandPose3D import ColorHandPose3D
from colorhandpose3d.utils.transforms import *
from main.preprocessing import heatmap2skeleton
import clip


img_size = 128
device = torch.cuda.set_device(0)


class Model(nn.Module):
    def __init__(self, pose_model, clip_model, clip_process): 
        super(Model, self).__init__()

        self.clip_model = clip_model.to(device) 
        self.clip_process = clip_process

        self.pose_model = pose_model
        self.batch_size = cfg.train_batch_size

    def forward(self, inputs, targets, meta_info, mode):

        input_img = inputs['img']         

        coords_xyz_rel_normed, keypoint_scoremap = self.pose_model(input_img, meta_info['hand_side'])
            
        out = {}

        out['intrinsic'] = meta_info['test_intrinsic']
        out['img2bb_trans'] = meta_info['img2bb_trans']
        out['bb2img_trans'] = meta_info['bb2img_trans']
        out['2d_coord'] = heatmap2skeleton(keypoint_scoremap) 
        out['depth'] = torch.unsqueeze(coords_xyz_rel_normed[:,:,2], 2)
        out['3d_coord'] = torch.cat([out['2d_coord'], out['depth'].detach().cpu()],dim=2)
        out['3d_related_coord'] = coords_xyz_rel_normed 
        out['img'] = input_img

        if 'fit_mesh_coord_cam' in targets:
            out['mesh_coord_cam_target'] = targets['fit_mesh_coord_cam']
        
        return out

def init_weights(m):
	if type(m) == nn.ConvTranspose2d:
		nn.init.kaiming_normal_(m.weight)
	elif type(m) == nn.Conv2d:
		nn.init.kaiming_normal_(m.weight)
		nn.init.constant_(m.bias, 0)
	elif type(m) == nn.BatchNorm2d:
		nn.init.constant_(m.weight,1)
		nn.init.constant_(m.bias,0)
	elif type(m) == nn.Linear and cfg.Contrastive == False:
		nn.init.kaiming_normal_(m.weight)
		nn.init.constant_(m.bias,0)

def get_model(mode):
    pose_net = ColorHandPose3D()
    clip_model, clip_process = clip.load("ViT-B/32")
    clip_process.transforms.pop(1)
    clip_process.transforms.pop(1)
    clip_process.transforms.pop(1)
    if mode == 'train':
        pose_net.apply(init_weights)
        freeze_param(clip_model)

    model = Model(pose_net, clip_model.to(device), clip_process) 

    return model

