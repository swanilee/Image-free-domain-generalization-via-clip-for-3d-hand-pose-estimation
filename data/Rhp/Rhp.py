import os
import os.path as osp
import numpy as np
from main.config import cfg

import torch
import torchvision.transforms as transforms
import cv2

import pickle
import copy
import torch.nn.functional as F

from main.mano import MANO
from main.preprocessing import load_img, test_augmentation, create_multiple_gaussian_map
from main.transforms import pixel2cam
from main.metrics import Evaluator


class Rhp(torch.utils.data.Dataset):

    def __init__(self, transform, data_split):
        self.transform = transform
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.clip_normalize_img = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.data_split = data_split
        self.data_path = "/data/sglee/handAR/data/Rhp/Rhp_data"
        
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

        self.bbox = np.load('../data/Rhp/Rhp_bbox.npy')

        
    def load_data(self):
        if self.data_split == 'train':
            with open(osp.join(self.data_path, 'training_n.pickle'), 'rb') as f:
                data = pickle.load(f)
                self.data_path = osp.join(self.data_path, 'training', 'color')
                
        else:
            with open(osp.join(self.data_path, 'testing_n.pickle'), 'rb') as f:
                data = pickle.load(f)
                self.data_path = osp.join(self.data_path, 'evaluation', 'color')
        
        filelist = ["00359.png", "02819.png", "03931.png", "07731.png", "08401.png", "08919.png", "09208.png", "10608.png",
                    "11842.png", "12662.png", "13902.png", "14157.png", "14328.png", "15064.png", "15211.png", "15644.png",
                    "16326.png", "21832.png", "22118.png", "24453.png", "24456.png", "24751.png", "26259.png", "26771.png",
                    "27900.png", "28508.png", "29103.png", "00162.png", "05372.png", "29394.png", "27585.png", "36496.png",
                    '07854.png', '29456.png', '24144.png', '36126.png', '38512.png', '29295.png', '24636.png', '29222.png',
                    '21607.png', '29860.png', '24883.png', '24556.png','10775.png', '01488.png', '14285.png', '33648.png',
                    '22117.png', '40115.png', '40057.png', '08061.png', '32002.png'] 
        
        datalist = []
        for idx in range(len(data['annotations'])):
            ann = data['annotations'][idx]
            img = data['images'][idx]

            img_path = osp.join(self.data_path, img['file_name'])
            img_shape = (img['height'], img['width'])

            if img['file_name'] in filelist:
                continue

            cam_param, joint_cam, joint_img = img['param'], ann['xyz'], ann['uvd']
            
            # if bbox is None: continue
            root_joint_depth = joint_cam[0,2]

            side = ann['side']
            _type = 'Rhp'
             
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
        img_path, img_shape, joint_cam, joint_img, cam_param, side, root_joint_depth, _type = data['img_path'], data['img_shape'], data['joint_cam'], data['joint_img'], data['cam_param'], data['side'], data['root_joint_depth'], data['type']        

        # img, mask laod
        img = load_img(img_path)
        origin_img = copy.deepcopy(img)

        if side == 'left':
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

        inputs = {'img': img, 'clip_img': clip_img}

        meta_info = {}
        meta_info['img'] = img
        meta_info['joints_3d'] = joint_cam
        meta_info['uvd'] = joint_img

        meta_info['K'] = torch.from_numpy(cam_param).float()
        meta_info['idx'] = idx
        hand_side = torch.tensor([1.0, 0.0])
        meta_info['hand_side'] = hand_side
        meta_info['side'] = side

        joint_img[:,0] = joint_img[:,0] / cfg.input_img_shape[0]  
        joint_img[:,1] = joint_img[:,1] / cfg.input_img_shape[0]  

        joint_img[:,2] = joint_img[:,2] - root_joint_depth
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
        out_scale = 0
        gt_scale = 0
        out_z = 0
        gt_z = 0
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            out = outs['3d_related_coord'][cur_sample_idx + n] 
            out = np.array(out)
            out[:,:2] = out[:,:2] * 256

            annot['focal'] = np.array([annot['cam_param'][0,0], annot['cam_param'][1,1]])
            annot['princpt'] = np.array([annot['cam_param'][0,2], annot['cam_param'][1,2]])

            out[:,2] = (out[:,2] * 2 -1) * (0.3 / 2) 
            out[:,2] = out[:,2] + annot['root_joint_depth']

            mano_coord_img_xy1 = np.concatenate((out[:,:2], np.ones_like(out[:,:1])),1)
            out[:,:2] = np.dot(outs['bb2img_trans'][cur_sample_idx + n], mano_coord_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]

            out = pixel2cam(out, annot['focal'], annot['princpt'])
            gt = annot['joint_cam']
            out[:,2] += gt[0,2] - out[0,2]

            out *= 1000
            gt *= 1000

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