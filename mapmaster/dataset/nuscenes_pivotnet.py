import os
import numpy as np
import pickle as pkl
from PIL import Image
from torch.utils.data import Dataset
from .rasterize import RasterizedLocalMap
from .vectorize import VectorizedLocalMap
from .generate_pivots import GenPivots
from nuscenes import NuScenes
from pyquaternion import Quaternion
from skimage import io as skimage_io
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from icecream import ic
import open3d as o3d
from functools import reduce
import torch
import random
from .voxel import pad_or_trim_to_np
from .visual import visual_map_gt
from PIL import Image

 
class NuScenesMapDataset(Dataset):
    def __init__(self, img_key_list, map_conf, point_conf,  transforms,  data_split="training"):
        super().__init__()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        
        self.ego_size = map_conf["ego_size"]
        self.mask_key = map_conf["mask_key"]
        self.nusc_root = map_conf["nusc_root"]
        self.split_dir = map_conf["split_dir"]        # instance_mask/instance_mask8
        self.map_region =map_conf['map_region']
        
        patch_h = map_conf['map_region'][3] - map_conf['map_region'][2]   #120   30 
        patch_w = map_conf['map_region'][1] - map_conf['map_region'][0]    
        canvas_h = int(patch_h / map_conf['map_resolution']) #800， 200
        canvas_w = int(patch_w / map_conf['map_resolution'])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=map_conf['version'], dataroot= map_conf['nusc_root'], verbose=False)
        self.vector_map = VectorizedLocalMap( map_conf['nusc_root'], patch_size=self.patch_size, canvas_size=self.canvas_size)    #[m]
        self.raster_map = RasterizedLocalMap(self.patch_size, self.canvas_size, num_degrees=[2, 1, 3], max_channel=3, thickness=[1, 8], bezier=False) #[pixel]
        self.pivot_gen = GenPivots(max_pts=point_conf['max_pieces'], map_region = map_conf['map_region'], resolution=map_conf['map_resolution'])#
        
        self.split_mode= data_split
        
        split_path = os.path.join(self.split_dir, f'{self.split_mode}.txt')
        self.tokens = [token.strip() for token in open(split_path).readlines()]
        self.transforms = transforms
        
    def get_data_info(self, record):
        imgs, extrins, intrins = [], [], []
        for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
            samp = self.nusc.get('sample_data', record['data'][cam])

            img = skimage_io.imread( os.path.join(self.nusc.dataroot, samp['filename']))   #(900, 1600, 3)
            imgs.append(img)
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            extrinsic =  np.eye(4) 
            extrinsic[ :3, :3] = Quaternion(sens['rotation']).rotation_matrix
            extrinsic[  :3, 3] = sens['translation']
            extrins.append(extrinsic)
            intrins.append(sens['camera_intrinsic'])
        return imgs, extrins, intrins
    
    def get_lidar_data(self, nusc, sample_rec, nsweeps, min_distance):
        """
        Returns at most nsweeps of lidar in the ego frame.
        Returned tensor is 5(x, y, z, reflectance, dt) x N"""
        points = np.zeros((5, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data']['LIDAR_TOP']
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        #print(len(current_sd_rec), len( current_sd_rec['filename']))
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)
            #print(current_pc.points.shape, len(current_sd_rec), len(sample_data_token), len(current_sd_rec['filename']), current_sd_rec['filename'])
            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)


            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)#new_points.shape: (5, 25555)
            points = np.concatenate((points, new_points), 1)# points.shape: (5, 254921)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return points
    
    def sampled(self, pc):   #[5, N]
        mask_x = np.fabs(pc[0, :])<np.fabs(self.map_conf['map_region'][0])  
        mask_y = np.fabs(pc[1, :])<np.fabs(self.map_conf['map_region'][2])
        mask_z = np.fabs(pc[2, :])< 3
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        selected_pc = pc[:, mask]
    
        point = selected_pc[:3, :]
        intensity = np.zeros(point.shape)
        #print(selected_pc.shape, intensity.shape)
        intensity[0:2, :] = selected_pc[3:5, :]
        #print(intensity)
        pcd=o3d.geometry.PointCloud()  # 传入3d点云格式
        pcd.points = o3d.utility.Vector3dVector(point.transpose(1, 0))#转换格式    
        pcd.colors = o3d.utility.Vector3dVector(intensity.transpose(1, 0))
        pcd_down =  pcd.voxel_down_sample(voxel_size=0.15)
        #model, idx = pcd_down.segment_plane(distance_threshold = 0.3, ransac_n = 3, num_iterations = 100)
        #ground = pcd_down.select_by_index(idx)
        
        return np.concatenate((np.asarray(pcd_down.points), np.asarray(pcd_down.colors)), axis = -1).transpose(1,0)
        
    
    def get_lidar(self, rec, data_aug, flip_sign):
        #print('1')
        lidar_data = self.get_lidar_data(self.nusc, rec, nsweeps=10, min_distance=2.2)#lidar_data.shape: (5, 76311)
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(lidar_data)
        # open3d.visualization.draw_geometries([point_cloud])
        #print('1lidar', lidar_data.shape)
        lidar_data = self.sampled(lidar_data)
        lidar_data_original = torch.Tensor(lidar_data)[:3]#将原始数据转换为PyTorch张量#ic| lidar_data_original.size(): torch.Size([3, 80996])
        #print('2lidar', lidar_data.shape)
        lidar_data = lidar_data.transpose(1, 0)#对雷达数据进行转置, lidar_data.shape: (76311, 5)
        num_points = lidar_data.shape[0]
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(lidar_data)
        # open3d.visualization.draw_geometries([point_cloud])
        # exit(0)
        '''DA = False
        drop_points = False
        if data_aug:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                drop_points = random.uniform(0, 0.2)

        

        if drop_points is not False:
            points_to_drop = np.random.randint(
                0, num_points-1, int(num_points*drop_points))
            lidar_data = np.delete(
                lidar_data, points_to_drop, axis=0)

        if flip_sign:
            lidar_data[:, 1] = -lidar_data[:, 1]

        # 根据条件DA的取值，对lidar_data张量中的数据进行随机扰动操作，用于数据增强。
        if DA:
            jitter_x = random.uniform(-0.05, 0.05)
            jitter_y = random.uniform(-0.05, 0.05)
            jitter_z = random.uniform(-1.0, 1.0)
            lidar_data[:, 0] += jitter_x
            lidar_data[:, 1] += jitter_y
            lidar_data[:, 2] += jitter_z'''

        lidar_data = pad_or_trim_to_np(lidar_data, [50000, 5]).astype('float32')
        
        lidar_mask = np.ones(50000).astype('float32')
        lidar_mask[num_points:] *= 0.0

        return torch.Tensor(lidar_data) , lidar_mask, lidar_data_original# lidar_data.size: 409600,lidar_mask.size: 81920

    def get_lidar_10(self, rec, nsweeps=10):
        lidar_data = self.get_lidar_data( self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        lidar_data_original = torch.Tensor(lidar_data)[:3]
        return lidar_data_original
      
    def __getitem__(self, idx: int):
        from av2.geometry.se3 import SE3
        token = self.tokens[idx]
        
        record = self.nusc.sample[idx]
        location = self.nusc.get('log', self.nusc.get('scene', record['scene_token'])['log_token'])['location']
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', record['data']['LIDAR_TOP'])['ego_pose_token'])

        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        imgs, extrins, intrins = self.get_data_info(record)
        semantic_masks, instance_masks, instance_vec_points, instance_ctr_points = self.raster_map.convert_vec_to_mask(vectors)
        pivots, pivot_lengths = self.pivot_gen.pivots_generate(instance_vec_points)
        
        lidar_data, lidar_mask, lidar_data_original = self.get_lidar(record, data_aug=False, flip_sign=False)

        #np.savetxt('./data/nuscenes/lidar/' +  str(idx) + '.txt', lidar_data, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
        targets = dict(masks=instance_masks[1], points=pivots, valid_len=pivot_lengths) 
        #print(imgs[0].shape)#(900, 1600, 3)
      
        item = dict(images=imgs, targets=targets, lidars=lidar_data,  lidar_mask= lidar_mask, 
                    extra_infos=dict(token=token, map_size=self.ego_size),
                    extrinsic=np.stack(extrins, axis=0), intrinsic=np.stack(intrins, axis=0))
        #print(lidar_data.size())
        if self.transforms is not None:
            item = self.transforms(item)
        
        #print(item['images'].size(), item['lidars'].size())#torch.Size([6, 3, 512, 896]) torch.Size([50000, 5])
        #visual_map_gt(self.map_region, instance_vec_points, idx, self.data_root)

        #print(idx, token, record)
        
        '''cam = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i in range(6):
            print(imgs[i].shape, item['images'][i].size())  #(900, 1600, 3)  3, 512, 896
            im = Image.fromarray(imgs[i])
            print(item['images'][i])
            im.save('./data/nuscenes/cam/' +  str(idx) + str(cam[i])+ '.png')'''
          
        return item

    def __len__(self):
        return len(self.tokens)
