import numpy as np
import math

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def epe(self, gt_joint, out_joint):

        assert gt_joint.shape == out_joint.shape
        result = np.mean((np.sum((gt_joint - out_joint) ** 2, axis = 1)) ** 0.5)

        xyz_epe = np.mean(np.abs(gt_joint - out_joint), axis = 0)
        print(f"x_epe: {xyz_epe[0]}, y_epe: {xyz_epe[1]}, z_epe: {xyz_epe[2]}, epe: {result}")
        return xyz_epe, result


    def epe_detailed(self, gt_joint, out_joint):

        x_sum = 0
        y_sum = 0 
        z_sum = 0 
        error = 0 
        
        for per_joints in range(21):
            x = (gt_joint[per_joints][0] - out_joint[per_joints][0]) ** 2
            y = (gt_joint[per_joints][1] - out_joint[per_joints][1]) ** 2
            z = (gt_joint[per_joints][2] - out_joint[per_joints][2]) ** 2

            xy_sum = x + y + z
            xy_sum = math.sqrt(xy_sum)
            
            x_sum += (math.sqrt(x) / 21); y_sum += (math.sqrt(y) / 21); z_sum += (math.sqrt(z) / 21); 
            error += (xy_sum / 21)
            
        return [x_sum , y_sum , z_sum ], error
