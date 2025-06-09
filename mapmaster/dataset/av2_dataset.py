import os
import numpy as np
import os.path as osp
import torch
import mmcv
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from time import time
from functools import partial
from multiprocessing import Pool
import multiprocessing
from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from tqdm import tqdm
import icecream as ic
import warnings
import pandas as pd
import open3d as o3d
from .rasterize_av2 import preprocess_map, preprocess_osm_map
from .image import normalize_tensor_img
from .utils import label_onehot_encoding
from .av2map_extractor import  VectorizedAV2LocalMap
from .pipelines import VectorizeMap, LoadMultiViewImagesFromFiles, FormatBundleMap#,LoadLiDARPointCloudFromFiles
from .pipelines import PhotoMetricDistortionMultiViewImage, ResizeMultiViewImages, PadMultiViewImages
from .rasterize import RasterizedLocalMap
from .generate_pivots import GenPivots
from .visual import visual_map_gt,visual_map_gt_pivot
from PIL import Image
from .av2_converter_copy import create_av2_infos_mp#, extract_local_map
from av2.map.map_api import ArgoverseStaticMap
from shapely.geometry import Polygon, LineString, box, MultiPolygon, MultiLineString

warnings.filterwarnings("ignore")


NUM_CLASSES = 3

FAIL_LOGS = [
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275',
]

CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]

def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]
    
    
MAPCLASSES = ('divider' , 'ped_crossing', 'boundary')


class AV2PMapNetSemanticDataset(Dataset):
    def __init__(self,  img_key_list, map_conf, point_conf,  dataset_setup, transforms, data_split="train"):
        # import pdb; pdb.set_trace()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.map_region =map_conf['map_region']
        self.av2_root = map_conf["av2_root"]
        self.mask_key = map_conf["mask_key"]
        self.ego_size = map_conf["ego_size"]

        self.thickness = map_conf['line_width']   #
        self.angle_class = map_conf['angle_class']     #
     
        patch_h = map_conf['map_region'][3] - map_conf['map_region'][2]   #120   30 
        patch_w = map_conf['map_region'][1] - map_conf['map_region'][0]   
        canvas_h = int(patch_h / map_conf['map_resolution']) #800， 200
        canvas_w = int(patch_w / map_conf['map_resolution'])      
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
         
         
        self.split_mode= data_split
        
        img_size = (map_conf['image_size'][0], map_conf['image_size'][1])
        # category configs
        
        self.interval = 1
        
        # bev configs
        roi_size =  self.ego_size
        # vectorize params
        coords_dim = 2
        sample_dist = -1
        sample_num = -1
        # meta info for submission pkl
       

        # model configs
        num_points = -1
        sample_dist = 1
        permute = False
        # data processing pipelines
        #self.vectorize_map = VectorizeMap(coords_dim=coords_dim,  roi_size=roi_size,  sample_num=num_points, sample_dist=sample_dist, normalize=False, permute=permute,)
        self.MAPCLASSES = MAPCLASSES
        self.NUM_VECTORCLASSES = len(self.MAPCLASSES)
        self.samples = self.load_annotations()
      
        aux_seg=dict(
                use_aux_seg=False,
                bev_seg=False,
                pv_seg=False,
                seg_classes=1,
                feat_down_sample=32)
        self.code_size=2
        padding_value=-10000
        fixed_ptsnum_per_line= -1

        self.vector_map = VectorizedAV2LocalMap(self.av2_root,
                                                patch_size=self.patch_size, test_mode=self.split_mode,
                                                map_classes=self.MAPCLASSES,
                                                fixed_ptsnum_per_line=num_points,
                                                padding_value=-10000)
        #self.vector_map = AV2MapExtractor(roi_size = roi_size, patch_size=self.patch_size, canvas_size=self.canvas_size, id2map = self.id2map)
        self.load_images = LoadMultiViewImagesFromFiles(dataset_setup["input_size"], to_float32=True)
        #self.load_lidar = LoadLiDARPointCloudFromFiles(self.map_conf, to_float32=True)
        #self.aug_images = PhotoMetricDistortionMultiViewImage()
        #self.resize_images = ResizeMultiViewImages(size=img_size,  change_intrinsics=True)
        #self.pad_images = PadMultiViewImages(size= (2048, 2700), change_intrinsics = True)
        #self.format = FormatBundleMap()
        self.raster_map = RasterizedLocalMap(self.patch_size, self.canvas_size, num_degrees=[2, 1, 3], max_channel=3, thickness=[1, 8], bezier=False) #[pixel]
        self.pivot_gen = GenPivots(max_pts=point_conf['max_pieces'], map_region = map_conf['map_region'], resolution=map_conf['map_resolution'])#
        self.transforms = transforms
    

        super(AV2PMapNetSemanticDataset, self).__init__()
      
    def load_annotations(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # import pdb; pdb.set_trace()
        start_time = time()
     
        #split = os.listdir(osp.join(self.av2_root, self.split_mode))   #train /val/test
        
        data = create_av2_infos_mp(root_path=self.av2_root, info_prefix=None,  dest_path=None, split= self.split_mode, num_multithread=64)
        data_infos = list(sorted(data['samples'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.interval]
        #id2map = data['id2map']
        
        print(f'collected {len(data_infos)} samples in {(time() - start_time):.2f}s')
        return data_infos#, id2map
      
    
    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        log_id = sample['log_id']
        map_elements = sample['annotation']
       
        
        vectors = self.vector_map.gen_vectorized_samples(log_id, map_elements, sample['e2g_translation'],  sample['e2g_rotation'])
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
        
        input_dict = {
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            'lidar_filenames':  sample['lidar_fpath'],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, NOTE: **ego2cam**
            # 'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'cam_extrinsics': [np.linalg.inv(c['extrinsics']) for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'vectors': vectors,
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': sample['e2g_rotation'],
            #'scene_name': sample['scene_name'],
            'timestamp': sample['timestamp'],
            'log_id':log_id,
        }
       
        return input_dict
    
    
    
    def get_lidar_data(self, results, idx):
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
        filename = results['lidar_filenames']
        lidar_data = []
        #print(len(filename))
        #for i in range(len(filename)):
        lidar_data_cur = np.array(pd.read_feather(filename))#LidarPointCloud.from_file(filename
        #print(lidar_data_cur.shape)
        lidar_data.append(lidar_data_cur)
        
        lidar_data = np.concatenate(lidar_data, axis = 0)
        
        #print('lidar', lidar_data.shape)
        #np.savetxt('./data/av2/lidar/' +  str(idx) + '.txt', lidar_data, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
        #lidar_data.remove_close(min_distance)
        lidar_data = self.sampled(lidar_data)
        lidar_data_original = torch.Tensor(lidar_data)[:3]#将原始数据转换为PyTorch张量#ic| lidar_data_original.size(): torch.Size([3, 80996])
        #print('2lidar', lidar_data.shape)
        lidar_data = lidar_data.transpose(1, 0)#对雷达数据进行转置, lidar_data.shape: (76311, 5)
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [50000, 5]).astype('float32')
        
        
        lidar_mask = np.ones(50000).astype('float32')
        lidar_mask[num_points:] *= 0.0

        results['lidar'] = torch.Tensor(lidar_data)
        results['lidar_mask'] = lidar_mask
        results['lidar_data_original'] = lidar_data_original
        return results


        
    def sampled(self, pc):   #[5, N]
        mask_x = np.fabs(pc[ :, 0])<np.fabs(self.map_conf['map_region'][0])  
        mask_y = np.fabs(pc[ :, 1])<np.fabs(self.map_conf['map_region'][2])
        mask_z = np.fabs(pc[ :, 2])< 3
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        selected_pc = pc[mask, : ]
    
        point = selected_pc[ :,:3]
        intensity = np.zeros(point.shape)
        #print(selected_pc.shape, intensity.shape)
        intensity[ :, 0:2] = selected_pc[ :, 3:5]
        #print(intensity)
        pcd=o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(point)#转换格式    
        pcd.colors = o3d.utility.Vector3dVector(intensity)
        pcd_down =  pcd.voxel_down_sample(voxel_size=0.15)
        #print(np.asarray(pcd_down.points).shape)
        #model, idx = pcd_down.segment_plane(distance_threshold = 0.3, ransac_n = 3, num_iterations = 100)
        #ground = pcd_down.select_by_index(idx)
        
        return np.concatenate((np.asarray(pcd_down.points), np.asarray(pcd_down.colors)), axis = -1).transpose(1,0)
        
    
            
    def pipeline(self, input_dict,idx):

        # VectorizeMap
        semantic_masks, instance_masks, instance_vec_points, instance_ctr_points =  self.raster_map.convert_vec_to_mask(input_dict['vectors'])
        pivots, pivot_lengths = self.pivot_gen.pivots_generate(instance_vec_points)
        
        targets = dict(masks=instance_masks[1], points=pivots, valid_len=pivot_lengths) 
        input_dict['targets'] = targets

        # LoadMultiViewImagesFromFiles
        input_dict = self.load_images(input_dict)

        input_dict = self.get_lidar_data(input_dict, idx)
        # PhotoMetricDistortionMultiViewImage
        # only train
        if self.split_mode == 'train':
            input_dict = self.aug_images(input_dict)
        # ResizeMultiViewImages
       
        # PadMultiViewImages
        #input_dict = self.pad_images(input_dict) 
        #input_dict = self.resize_images(input_dict)
        # format
        #input_dict = self.format(input_dict)
        
        visual_map_gt_pivot(self.map_region, targets, str(idx)+str('+')+input_dict["log_id"], self.av2_root)
        visual_map_gt(self.map_region, instance_vec_points, str(idx)+str('+')+input_dict["log_id"] , self.av2_root)
        return input_dict


    def __getitem__(self, idx):
        input_dict = self.get_sample(idx)
        data = self.pipeline(input_dict,idx)

        imgs = data['img'] # N,3,H,W
        #imgs = normalize_tensor_img(imgs/255)
        ego2imgs = data['cam_extrinsics'] # list of 7 cameras, 4x4 array
        ego2global_tran = data['ego2global_translation'] # 3
        ego2global_rot = data['ego2global_rotation'] # 3x3
        timestamp = torch.tensor(data['timestamp'] / 1e9) # TODO: verify ns?
        #scene_id = data['scene_name']
        rots = torch.stack([torch.Tensor(ego2img[:3, :3]) for ego2img in ego2imgs]) # 7x3x3
        trans = torch.stack([torch.Tensor(ego2img[:3, 3]) for ego2img in ego2imgs]) # 7x3
        post_rots = rots
        post_trans = trans
        intrins = torch.stack([torch.Tensor(intri) for intri in data['cam_intrinsics']])
        car_trans = torch.tensor(ego2global_tran) # [3]
        pos_rotation = Quaternion(matrix=ego2global_rot)
        yaw_pitch_roll = torch.tensor(pos_rotation.yaw_pitch_roll)
        
        # for uniform interface
        lidar_data = data['lidar']
        lidar_mask = data['lidar_mask']
        #np.savetxt('./data/av2/lidar/' +  str(idx) + '.txt', lidar_data, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
        #print(imgs[0].permute(1, 2, 0).shape, lidar_data.shape)#[3, 928, 1600])
        #exit()
        item = dict(images= imgs, lidars=lidar_data,  lidar_mask= lidar_mask, targets=data['targets'],
                    extra_infos=dict(token=data['token'], map_size=self.ego_size),
                    extrinsic=np.stack(ego2imgs, axis=0), intrinsic=np.stack(intrins, axis=0))
        
        
        
        if self.transforms is not None:
            item = self.transforms(item)
        
        '''if(idx%200==0):
            cam_filename = self.av2_root + '/cam/'
            if not os.path.exists(cam_filename):
                os.makedirs(cam_filename)
            for i in range(7):
                #print(imgs[i].shape, item['images'][i].size()) #[1500, 2048, 3])   item['images'][i]   #[3, 768, 1024]
                im = Image.fromarray(imgs[i])
                print(cam_filename +  str(idx) + str(img_key_list[i])+ '.png')
                im.save(cam_filename +  str(idx) + str(img_key_list[i])+ '.png')
        #print(item['images'].shape)  #  [7, 3,768, 1024, ])'''
        return   item
    
    
    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':




    map_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
        'dataset_name': "Argoverse2",
        'av2_root':'./data/av2/',
        'split_dir':"assets/splits/nuscenes",
        'num_classes':3,
        'ego_size':(120, 30),
        'map_region':(-60, 60, -15, 15),
        'map_resolution':0.15,
        'map_size':(800, 200),
        'mask_key':"instance_mask8",
        'line_width':8,
        'save_thickness':1,
    }

    dataset = AV2PMapNetSemanticDataset (map_conf=map_conf, data_split = "training")
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)