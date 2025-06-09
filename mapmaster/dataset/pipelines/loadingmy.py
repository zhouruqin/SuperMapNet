import mmcv
import numpy as np
import pandas as pd
import open3d as o3d
import torch
from skimage import io as skimage_io
from skimage.transform import resize



class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, re_size , to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.re_size = re_size

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        imgs = []
        post_intrinsics = []
        i = 0
        for filename,  cam_intrinsic in zip( results['img_filenames'], results['cam_intrinsics']):
            img = skimage_io.imread(filename)
            #print(img.shape)
            if i == 0:
                img = img[896:2048, 0:1550, :]    # crop changes cam_intrinsic
                crop_matrix = np.array([
                [0,      0,      0,    0],
                [0,      0,      -896,    0],
                [0,      0,      0,    0],
                [0,      0,      0,    0]])
                cam_intrinsic =  cam_intrinsic + crop_matrix[:3, :3]
                     
            img, scaleW, scaleH = mmcv.imresize(img,
                                                # NOTE: mmcv.imresize expect (w, h) shape
                                                self.re_size,  #, w, H
                                                return_scale=True)
            rot_resize_matrix = np.array([     #resize changes cam_intrinsic
            [scaleW, 0,      0,    0],
            [0,      scaleH, 0,    0],
            [0,      0,      1,    0],
            [0,      0,      0,    1]])
            post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
            post_intrinsics.append(post_intrinsic)
            imgs.append(img)
            i = i + 1
        
        
        results['img'] = imgs
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        #results['img_norm_cfg'] = dict(
        #    mean=np.zeros(num_channels, dtype=np.float32),
        #    std=np.ones(num_channels, dtype=np.float32),
        #    to_rgb=True)
        results['img_fields'] = ['img']
        results.update({
            'cam_intrinsics': post_intrinsics,
        })

        return results
        
        
    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, 'f"color_type='{self.color_type}')"


