from __future__ import print_function, unicode_literals
import numpy as np
import json
import os
import time
import skimage.io as io

""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


""" Dataset related functions. """
def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


def load_db_annotation(base_path, writer=None, set_name=None):
    if set_name == 'training':
        if writer is not None:
            writer.print_str('Loading FreiHAND training set index ...')
        t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % set_name)

        # assumed paths to data containers
        mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

        # load if exist
        K_list = json_load(k_path)
        mano_list = json_load(mano_path)
        xyz_list = json_load(xyz_path)

        # should have all the same length
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
        if writer is not None:
            writer.print_str('Loading of %d %s samples done in %.2f seconds' % (len(K_list), set_name, time.time()-t))
        return zip(K_list, mano_list, xyz_list)
    elif set_name == 'evaluation':
        if writer is not None:
            writer.print_str('Loading FreiHAND eval set index ...')
        t = time.time()
        k_path = os.path.join(base_path, '%s_K.json' % set_name)
        scale_path = os.path.join(base_path, '%s_scale.json' % set_name)
        K_list = json_load(k_path)
        scale_list = json_load(scale_path)

        assert len(K_list) == len(scale_list), 'Size mismatch.'
        if writer is not None:
            writer.print_str('Loading of %d eval samples done in %.2f seconds' % (len(K_list), time.time() - t))
        return zip(K_list, scale_list)
    else:
        raise Exception('set_name error')


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]


    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size*cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb', '%08d.jpg' % sample_version.map_id(idx, version))
    if not os.path.exists(img_rgb_path):
        img_rgb_path = os.path.join(base_path, set_name, 'rgb2', '%08d.jpg' % idx)

    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def read_img_abs(idx, base_path, set_name):
    img_rgb_path = os.path.join(base_path, set_name, 'rgb', '%08d.jpg' % idx)
    if not os.path.exists(img_rgb_path):
        img_rgb_path = os.path.join(base_path, set_name, 'rgb2', '%08d.jpg' % idx)

    _assert_exist(img_rgb_path)
    return io.imread(img_rgb_path)


def read_msk(idx, base_path, set_name):
    mask_path = os.path.join(base_path, set_name, 'mask',
                             '%08d.jpg' % idx)
    _assert_exist(mask_path)
    return (io.imread(mask_path)[:, :, 0] > 240).astype(np.uint8)