import math

import numpy as np
import torch
import torch.nn.functional as F



def get_rotation_matrix(rot_params):
    """Converts axis-angle parameters to a rotation matrix.

    The axis-angle parameters have an encoded angle.

    Args:
        rot_params - Tensor (batch x 3): ux, uy, uz axis-angle parameters.

    Returns:
        rot_matrix - Tensor (batch x 3 x 3): Rotation matrices.
    """

    theta = rot_params.norm(dim=1)

    st = torch.sin(theta)
    ct = torch.cos(theta)
    one_ct = 1.0 - torch.cos(theta)

    norm_fac = 1.0 / theta
    ux = rot_params[:, 0] * norm_fac
    uy = rot_params[:, 1] * norm_fac
    uz = rot_params[:, 2] * norm_fac

    top = torch.stack((ct + ux * ux * one_ct, ux * uy * one_ct - uz * st, ux * uz * one_ct + uy * st), dim=1)
    mid = torch.stack((uy * ux * one_ct + uz * st, ct + uy * uy * one_ct, uy * uz * one_ct - ux * st), dim=1)
    bot = torch.stack((uz * ux * one_ct - uy * st, uz * uy * one_ct + ux * st, ct + uz * uz * one_ct), dim=1)

    rot_matrix = torch.stack((top, mid, bot), dim=1)
    return rot_matrix


def flip_right_hand(coords_xyz_canonical, cond_right):
    """Flips the given canonical coordinates, when cond_right is true.

    The returned coordinates represent those of the left hand.

    Args:
        coords_xyz_canonical - Tensor (batch x num_keypoints x 3): Coordinates
            for each keypoint.
        cond_right - Tensor (batch x num_keypoints x 3): Values are 0 or 1
            depending on if the keypoints represent the right hand.

    Returns:
        coords_xyz_canonical_left - Tensor (batch x num_keypoints x 3):
            Resulting coordinates. Remain unchanged if cond_right is False.
    """

    s = coords_xyz_canonical.shape
    assert len(s) == 3, "coords_xyz_canonical must be (batch x 3 x num_keypoints)."
    assert len(cond_right.shape) == 3, "cond_right must be (batch x 3 x num_keypoints)."

    coords_xyz_canonical_mirrored = coords_xyz_canonical.clone()
    coords_xyz_canonical_mirrored[:, :, 2] *= -1.

    coords_xyz_canonical_left = torch.where(cond_right, coords_xyz_canonical_mirrored, coords_xyz_canonical)

    return coords_xyz_canonical_left


def flip_left_hand(coords_xyz_canonical, cond_right):
    """Flips the given canonical coordinates, when cond_right is false.

    The returned coordinates represent those of the right hand.

    Args:
        coords_xyz_canonical - Tensor (batch x num_keypoints x 3): Coordinates
            for each keypoint.
        cond_right - Tensor (batch x num_keypoints x 3): Values are 0 or 1
            depending on if the keypoints represent the left hand.

    Returns:
        coords_xyz_canonical_right - Tensor (batch x num_keypoints x 3):
            Resulting coordinates. Remain unchanged if cond_right is True.
    """

    s = coords_xyz_canonical.shape
    assert len(s) == 3, "coords_xyz_canonical must be (batch x 3 x num_keypoints)."
    assert len(cond_right.shape) == 3, "cond_right must be (batch x 3 x num_keypoints)."

    coords_xyz_canonical_mirrored = coords_xyz_canonical.clone()
    coords_xyz_canonical_mirrored[:, :, 0] *= -1.

    coords_xyz_canonical_left = torch.where(cond_right, coords_xyz_canonical, coords_xyz_canonical_mirrored)

    return coords_xyz_canonical_left


def transform_cropped_coords(cropped_coords, centers, scale, crop_size):
    """Transforms the cropped coordinates to the original image space.

    Args:
        cropped_coords - Tensor (batch x num_keypoints x 3): Estimated hand
            coordinates in the cropped space.
        centers - Tensor (batch x 1): Repeated coordinates of the
            center of the hand in global image space.
        scale - Tensor (batch x 1): Scaling factor between the original image
            and the cropped image.
        crop_size - int: Size of the crop.

    Returns:
        coords - Tensor (batch x num_keypoints x 3): Transformed coordinates.
    """

    coords = np.copy(cropped_coords)
    coords -= crop_size // 2
    coords /= scale
    coords += centers
    return coords
