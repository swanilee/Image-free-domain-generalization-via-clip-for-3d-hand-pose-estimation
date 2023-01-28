import torch
import numpy as np
from config import cfg

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    import warnings
    warnings.filterwarnings("error")
    try:
        x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
        y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
        z = pixel_coord[:, 2]
    except RuntimeWarning:
        print(pixel_coord)
        print(f)
        print(c)
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord
