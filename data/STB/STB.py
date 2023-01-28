import os
import os.path as osp
import numpy as np

import torch
import torchvision.transforms as transforms

import cv2
import pickle
import copy
import torch.nn.functional as F
from main.config import cfg

from main.mano import MANO
from main.preprocessing import load_img, test_augmentation, create_multiple_gaussian_map
from main.transforms import cam2pixel, pixel2cam
from main.metrics import Evaluator
import os

class STB(torch.utils.data.Dataset):

    def __init__(self, transform, data_split):
        self.transform = transform
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.clip_normalize_img = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.data_split = data_split

        with open("/workspace/HandMesh/template/MANO_RIGHT.pkl", 'rb') as f:
            mano = pickle.load(f, encoding='latin1')
        self.j_regressor = np.zeros([21, 778])
        self.j_regressor[:16] = mano['J_regressor'].toarray()
        for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
            self.j_regressor[k, v] = 1
        
        self.std = torch.tensor(0.20)
        # MANO joint set
        self.mano = MANO()
        self.face = self.mano.face
        self.joint_regressor = self.mano.joint_regressor
        self.vertex_num = self.mano.vertex_num
        self.joint_num = self.mano.joint_num
        self.joints_name = self.mano.joints_name
        self.skeleton = self.mano.skeleton
        self.root_joint_idx = self.mano.root_joint_idx

        self.datalist = self.load_data()
        self.metric = Evaluator(0)
    
        self.bbox = np.load('../data/STB/STB_bbox.npy')
        
    def load_data(self):

        img_root_path = '/data/sglee/handAR/data/STB/STB_data/all_image'

        test_targets = ['B1Counting', 'B1Random']

        datalist = []

        with open('/data/sglee/handAR/data/STB/STB_data/keypoint_train.pickle', 'rb') as f:
            data = pickle.load(f)


        for idx in range(len(data['annotations'])):
            ann = data['annotations'][idx]
            img = data['images'][idx]

            if img['file_name'].split('/')[0] not in test_targets:
                continue

            img_path = osp.join(img_root_path, img['file_name'])
            img_shape = (img['height'], img['width'])

            cam_param = np.array(img['param'])
            focal = np.array([cam_param[0,0], cam_param[1,1]])
            princpt = np.array([cam_param[0,2], cam_param[1,2]])
            uvd = cam2pixel(np.array(ann['xyz']), focal, princpt)
            
            joint_cam, joint_img = np.array(ann['xyz']), uvd

            root_joint_depth = joint_cam[0,2]

            side = 'left'
            _type = 'STB'
            
            datalist.append({
                    'img_path': img_path,
                    'img_shape': img_shape,
                    'joint_cam': joint_cam,
                    'joint_img': joint_img,
                    'cam_param': cam_param,
                    'root_joint_depth': root_joint_depth,
                    'side': side,
                    'type': _type
                    }
            )

        return datalist
    
    def __len__(self):
        return len(self.datalist)


    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, joint_cam, joint_img, cam_param, root_joint_depth, side, _type  = data['img_path'], data['img_shape'], data['joint_cam'], data['joint_img'], data['cam_param'], data['root_joint_depth'], data['side'], data['type']        

        # img, mask laod
        img = load_img(img_path)
        origin_img = copy.deepcopy(img)

        if side == 'left':
            joint_img[:, 0] = img_shape[1] - joint_img[:, 0]
            img = cv2.flip(img, 1)
        
        bbox = self.bbox[idx] 
        bbox[0] = int(bbox[0]); bbox[1] = int(bbox[1]); bbox[2] = int(bbox[2]); bbox[3] = int(bbox[3])
    

        # dummy mask / tex_mask
        img, clip_img, img2bb_trans, bb2img_trans, rot, _ = test_augmentation(img, bbox, self.data_split, exclude_flip=True) # FreiHAND dataset only contains right hands. do not perform flip aug.
        origin_img = self.transform(origin_img.astype(np.float32))/255.
        img = self.transform(img.astype(np.float32))/255.

        # affine transform x,y coordinates. root-relative depth
        mano_coord_img_xy1 = np.concatenate((joint_img[:,:2], np.ones_like(joint_img[:,:1])),1)
        joint_img[:,:2] = np.dot(img2bb_trans, mano_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
        
        inputs = {'img': img, 'clip_img':clip_img}

        meta_info = {}
        meta_info['img'] = img
        meta_info['joints_3d'] = joint_cam
        meta_info['uvd'] = joint_img

        meta_info['K'] = torch.from_numpy(cam_param).float()
        
        meta_info['idx'] = idx
        hand_side = torch.tensor([1.0, 0.0])
        meta_info['hand_side'] = hand_side
        
        joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[0]  
        joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0]  

        joint_img[:,2] = joint_img[:,2] - root_joint_depth
        joint_img[:,2] = joint_img[:,2] / 1000
        joint_img[:,2] = (joint_img[:,2] / (0.3 / 2) + 1)/2.

        targets = {}
        targets['fit_joint_img'] = joint_img
        targets['gau_map'] = create_multiple_gaussian_map(joint_img * cfg.input_img_shape[0] , inputs['img'].shape[1:])
        meta_info['test_intrinsic'] = cam_param
        meta_info['img2bb_trans'] = img2bb_trans
        meta_info['bb2img_trans'] = bb2img_trans
        meta_info['bb2img_trans'] = bb2img_trans

        return inputs, targets, meta_info
    
    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'3d_coord': [], '3d_related_coord': [], 'img2bb_trans': [], 'bb2img_trans':[]}
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs[n]

            if '3d_coord' in  out.keys():
                coord_3d = out['3d_coord']
                relative_coord = out['3d_related_coord']

            eval_result['3d_related_coord'].append(relative_coord.tolist())
            eval_result['3d_coord'].append(coord_3d.tolist())
            eval_result['img2bb_trans'].append(out['img2bb_trans'].tolist())
            eval_result['bb2img_trans'].append(out['bb2img_trans'].tolist())

        return eval_result
        
    def evaluate_detailed(self, outs, cur_sample_idx):
        
        annots = self.datalist
        sample_num = len(outs['3d_coord'])
        eval_result = {'joint_out': [], 'mesh_out': [], 'x_epe' : 0.0, 'y_epe' : 0.0, 'z_epe' : 0.0, 'epe' : 0.0}
        
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs['3d_coord'][cur_sample_idx + n]
            out = np.array(out)

            annot['focal'] = np.array([annot['cam_param'][0,0], annot['cam_param'][1,1]])
            annot['princpt'] = np.array([annot['cam_param'][0,2], annot['cam_param'][1,2]])

            out[:,2] = (out[:,2] * 2 -1) * (0.13 / 2) 
            out[:,2] = out[:,2] * 1000
            out[:,2] += annot['root_joint_depth'] - out[0,2]
            
            mano_coord_img_xy1 = np.concatenate((out[:,:2], np.ones_like(out[:,:1])),1)
            out[:,:2] = np.dot(outs['bb2img_trans'][cur_sample_idx + n], mano_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            
            out = pixel2cam(out, annot['focal'], annot['princpt'])

            gt = annot['joint_cam']
            gt[:,0] *= -1

            out[:,2] += gt[0,2] - out[0,2]
            xyz_epe, epe = self.metric.epe_detailed(gt,out)


            eval_result['epe'] += epe
            eval_result['x_epe'] += xyz_epe[0]
            eval_result['y_epe'] += xyz_epe[1]
            eval_result['z_epe'] += xyz_epe[2]

        eval_result['epe'] = eval_result['epe'] / sample_num
        eval_result['x_epe'] = eval_result['x_epe'] / sample_num
        eval_result['y_epe'] = eval_result['y_epe'] / sample_num
        eval_result['z_epe'] = eval_result['z_epe'] / sample_num

        print('mean_x_epe EPE : ', eval_result['x_epe'])
        print('mean_y_epe EPE : ', eval_result['y_epe'])
        print('mean_z_epe EPE : ', eval_result['z_epe'])
        print('Mean EPE : ', eval_result['epe'],'\n')

        return
