import numpy as np
import cv2
import random
from config import cfg
from PIL import Image, ImageEnhance
import torch
import copy

def pil2opencv(img):
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def opencv2pil(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil_img


def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()

    img = img.astype(np.float32)
    return img

def heatmap2skeleton(heatmapsPoseNet):
    skeletons = torch.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2)) #(B,21,2)
    for m in range(heatmapsPoseNet.shape[0]):
        for i in range(heatmapsPoseNet.shape[1]):
            u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i].cpu().detach().numpy()), (256, 256))
            skeletons[m, i, 0] = v
            skeletons[m, i, 1] = u

    return skeletons

def create_multiple_gaussian_map(coords_uv, output_size, valid_vec=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    coords_uv = torch.from_numpy(coords_uv)   
    sigma = torch.Tensor([10]) 
    sigma = sigma.type(torch.FloatTensor)
    assert len(output_size) == 2
    s = coords_uv.shape
    coords_uv = coords_uv.type(torch.IntTensor)
    if valid_vec is not None:
        valid_vec = valid_vec.type(torch.FloatTensor)
        valid_vec = torch.squeeze(valid_vec)
        cond_val = torch.greater(valid_vec, 0.5)
    else:
        cond_val = torch.ones_like(coords_uv[:, 0], dtype=torch.float)
        cond_val = torch.greater(cond_val, 0.5)

    cond_1_in = torch.logical_and(torch.less(coords_uv[:, 0], output_size[0]-1), torch.greater(coords_uv[:, 0], 0))
    cond_2_in = torch.logical_and(torch.less(coords_uv[:, 1], output_size[1]-1), torch.greater(coords_uv[:, 1], 0))
    cond_in = torch.logical_and(cond_1_in, cond_2_in)
    cond = torch.logical_and(cond_val, cond_in)

    coords_uv = coords_uv.type(torch.FloatTensor) 

    # create meshgrid
    x_range = torch.unsqueeze(torch.arange(output_size[0]), 1)
    y_range = torch.unsqueeze(torch.arange(output_size[1]), 0) 

    X = x_range.repeat([1, output_size[1]]) 
    Y = y_range.repeat([output_size[1], 1 ]) 
    X = X.type(torch.FloatTensor) 
    Y = Y.type(torch.FloatTensor) 
    
    X.view((output_size[0], output_size[1]))
    Y.view((output_size[0], output_size[1]))

    X = torch.unsqueeze(X, -1)
    Y = torch.unsqueeze(Y, -1)

    X_b = X.repeat([1, 1, s[0]])
    Y_b = Y.repeat([1, 1, s[0]])

    X_b -= coords_uv[:, 0]
    Y_b -= coords_uv[:, 1]

    dist = torch.square(X_b) + torch.square(Y_b)

    cond = cond.type(torch.FloatTensor)
    scoremap = torch.exp(-dist / torch.square(sigma)) * cond

    return scoremap



def get_aug_config(exclude_flip):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2

    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.random.uniform(-3.0, 3.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip


def augmentation(img, bbox, data_split, exclude_flip=False): #tex_mask
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
    else:
        print(cfg.scale, cfg.rot)
        scale, rot, color_scale, do_flip = cfg.scale, cfg.rot, np.array([1, 1, 1]), False
        brightness = 1
        contrast = 1

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    clip_img = copy.deepcopy(img)

    pil_img = opencv2pil(img.astype(np.uint8))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    img = pil2opencv(pil_img)
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, clip_img, trans, inv_trans, rot, do_flip #tex_mask


def test_augmentation(img,bbox, data_split, exclude_flip=False):
    scale, rot, color_scale, do_flip = cfg.scale, cfg.rot, np.array([1, 1, 1]), False
    brightness = 1
    contrast = 1

    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)
    clip_img = copy.deepcopy(img)

    pil_img = opencv2pil(img.astype(np.uint8))

    img = pil2opencv(pil_img)
    img = np.clip(img * color_scale[None, None, :], 0, 255)
    return img, clip_img, trans, inv_trans, rot, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

