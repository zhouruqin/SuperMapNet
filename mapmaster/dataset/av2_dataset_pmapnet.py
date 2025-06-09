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
import warnings
warnings.filterwarnings("ignore")

#from data_osm.rasterize import preprocess_map, preprocess_osm_map
#from .const import NUM_CLASSES
from .image import normalize_tensor_img
from .utils import label_onehot_encoding
from .av2map_extractor_pmapnet import AV2MapExtractor
from .pipelines import VectorizeMap, LoadMultiViewImagesFromFiles#, FormatBundleMap, Normalize3D
#from .pipelines import PhotoMetricDistortionMultiViewImage, ResizeMultiViewImages, PadMultiViewImages
from .rasterize import RasterizedLocalMap
from .generate_pivots import GenPivots
from .visual import visual_map_gt,visual_map_gt_pivot
import pandas as pd
import open3d as o3d
from torchvision.utils import save_image
from PIL import Image
CAM_NAMES = ['ring_front_center', 'ring_front_right', 'ring_front_left',
    'ring_rear_right','ring_rear_left', 'ring_side_right', 'ring_side_left',
    # 'stereo_front_left', 'stereo_front_right',
    ]

NUM_CLASSES = 3

FAIL_LOGS = [
    '01bb304d-7bd8-35f8-bbef-7086b688e35e',
    '453e5558-6363-38e3-bf9b-42b5ba0a6f1d',
    '75e8adad-50a6-3245-8726-5e612db3d165',
    '54bc6dbc-ebfb-3fba-b5b3-57f88b4b79ca',
    'af170aac-8465-3d7b-82c5-64147e94af7d',
    '6e106cf8-f6dd-38f6-89c8-9be7a71e7275', 
    '3b2994cb-5f82-4835-9212-0cac8fb3d164',
    '06e5ac08-f4cb-34ae-9406-3496f7cadc62',   #8
    '0749e9e0-ca52-3546-b324-d704138b11b5', #10
    '0a8a4cfa-4902-3a76-8301-08698d6290a2',  #19
    '1d950a38-5c2f-39ce-9cd3-61249bc85194',   #60
    '4977e8a8-4e1f-3ca2-a44e-454cd3756a5f',  #150
    '8223c3d0-3b08-3889-9cdc-a88592c4bd4a',
    '84bb4b17-e7f2-3a1b-8c2b-6d6ec9a23e31',
    '95a47a36-1041-3924-bbd0-4dcad52c323a',
    '9f6d282e-f573-31d5-80e6-9a193e80cd7d',
    'a7a2236e-8f8e-34aa-9343-722f9b3bb829',
    '2e95b33b-8ea1-3b48-875b-2f35f3092059',#13  test
    '2ee0eda7-151a-3957-bab5-1e5370192122', #14 
    '45488531-3648-3e2d-8f9c-3c287032112d', #23
    '557dd6a4-2b80-3264-9c13-f70094526174',#36   
    '9282db22-c361-3456-a7b5-414959f5f25e',#59
    'a89557fc-1268-36e5-9cce-335f2da27bc8',#72
    #'b0116f1c-f88f-3c09-b4bf-fc3c8ebeda56',#75
    'e4221cc6-a19d-31ca-bf94-031adb0ea390',#102   
    #'fee0f78c-cf00-35c5-975b-72724f53fd64',  #117
    '386c34fc-ff56-371c-9288-6ba42620f23a',  #17  keyi
    #'1c8648f9-e7a1-3056-a2c0-19c8827a6a50',#6  keyi '''
]


def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]
    
    
def get_data_from_logid(log_id, loader: AV2SensorDataLoader, data_root):
    samples = []
    discarded = 0
    # import pdb;pdb.set_trace()
    
    # We use lidar timestamps to query all sensors.
    # The frequency is 10Hz

    cam_timestamps = loader._sdb.per_log_lidar_timestamps_index[log_id]
    for ts in cam_timestamps[::4]:#::4
        cam_ring_fpath = [loader.get_closest_img_fpath(
                log_id, cam_name, ts
            ) for cam_name in CAM_NAMES]
        lidar_fpath = loader.get_closest_lidar_fpath(log_id, ts)

        # If bad sensor synchronization, discard the sample
        if None in cam_ring_fpath or lidar_fpath is None:
            discarded += 1
            continue

        cams = {}
        for i, cam_name in enumerate(CAM_NAMES):
            pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)
            cam_timestamp_ns = int(cam_ring_fpath[i].stem)
            cam_city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            cams[cam_name] = dict(
                img_fpath=str(cam_ring_fpath[i]),
                intrinsics=pinhole_cam.intrinsics.K,
                extrinsics=pinhole_cam.extrinsics,
                e2g_translation=cam_city_SE3_ego.translation,
                e2g_rotation=cam_city_SE3_ego.rotation,
            )
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, int(ts))
        e2g_translation = city_SE3_ego.translation
        e2g_rotation = city_SE3_ego.rotation
        
    
        samples.append(dict(
            e2g_translation=e2g_translation,
            e2g_rotation=e2g_rotation,
            cams=cams, 
            lidar_fpath=str(lidar_fpath),
            # map_fpath=map_fname,
            timestamp=ts,
            log_id=log_id,
            token=str(log_id+'_'+str(ts)),
            scene_name=log_id,
            ))

        
    return samples, discarded
    

def create_av2_infos_mp(  root_path,
                            dest_path=None,  
                            split='train',
                            num_multithread=64):
    
    print(split)
    root_path = osp.join(root_path, split)
    if dest_path is None:
        dest_path = root_path
    
    loader = AV2SensorDataLoader(Path(root_path), Path(root_path))
    log_ids = list(loader.get_log_ids())
    for i in FAIL_LOGS:
        if i in log_ids:
            log_ids.remove(i)
    # dataloader by original split
    # import pdb; pdb.set_trace()

    print('collecting samples...')
    start_time = time()
    print('num cpu:', multiprocessing.cpu_count())
    print(f'using {num_multithread} threads')

    # ignore warning from av2.utils.synchronization_database
    #sdb_logger = logging.getLogger('av2.utils.synchronization_database')
    #prev_level = sdb_logger.level
    #sdb_logger.setLevel(logging.CRITICAL)

    pool = Pool(num_multithread)
    fn = partial(get_data_from_logid, loader=loader, data_root=root_path)
    
    rt = pool.map_async(fn, log_ids)
    pool.close()
    pool.join()
    results = rt.get()

    samples = []
    discarded = 0
    sample_idx = 0
    for _samples, _discarded in tqdm(results):
        for i in range(len(_samples)):
            _samples[i]['sample_idx'] = sample_idx
            sample_idx += 1
        samples.extend(_samples)
        discarded += _discarded
    
    # sdb_logger.setLevel(prev_level)
    print(f'{len(samples)} available samples, {discarded} samples discarded')
    
    id2map = {}
    for log_id in tqdm(log_ids):
        if log_id in loader._sdb.get_valid_logs():
            loader = loader
        
        map_path_dir = osp.join(loader._data_dir, log_id, 'map')
        map_fname = str(list(Path(map_path_dir).glob("log_map_archive_*.json"))[0])
        #print(map_path_dir, map_fname)
        #map_fname
        id2map[log_id] = map_fname

    print('collected in {:.1f}s'.format(time() - start_time))
    infos = dict(samples=samples, id2map=id2map)
    return infos


class AV2PMapNetSemanticDataset(Dataset):
    def __init__(self, img_key_list,  map_conf, point_conf,  dataset_setup, transforms, data_split="train"):
        # import pdb; pdb.set_trace()
        self.img_key_list = img_key_list
        self.map_conf = map_conf
        self.thickness = map_conf['line_width'] 
        self.angle_class = map_conf['angle_class']
        self.av2_root = map_conf["av2_root"]
        self.map_region =map_conf['map_region']
        self.mask_key = map_conf["mask_key"]
        
        patch_h =  map_conf['map_region'][3] - map_conf['map_region'][2]
        patch_w = map_conf['map_region'][1] - map_conf['map_region'][0]
        canvas_h =  int(patch_h / map_conf['map_resolution'])
        canvas_w =  int(patch_w / map_conf['map_resolution'])    
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.ego_size = map_conf["ego_size"]
        self.split_mode= data_split
        
        img_size =  (map_conf['image_size'][0], map_conf['image_size'][1])
  
        self.interval = 1
         
        # bev configs
        roi_size =  self.ego_size
        # vectorize params
        coords_dim = 2
        sample_dist = -1
        sample_num = -1
        # meta info for submission pkl
        meta = dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False,
            output_format='vector')

        # model configs
        num_points = -1
        sample_dist = 1
        permute = False
        img_norm_cfg = map_conf['img_norm_cfg']
        # data processing pipelines
        self.load_images = LoadMultiViewImagesFromFiles(re_size = img_size, to_float32=True)
        #self.aug_images = PhotoMetricDistortionMultiViewImage()
        #self.norm_images = Normalize3D(mean=img_norm_cfg['mean'], std=img_norm_cfg['std'], to_rgb=img_norm_cfg['to_rgb'])
        #self.resize_images = ResizeMultiViewImages(size=img_size, change_intrinsics=True)
        #self.pad_images = PadMultiViewImages(size=(2048,2048), change_intrinsics=True)
        #self.format = FormatBundleMap()
        
        self.samples, self.id2map = self.load_annotations()
        self.map_extractor = AV2MapExtractor(roi_size, self.id2map)
        
        self.raster_map = RasterizedLocalMap(self.patch_size, self.canvas_size, num_degrees=[2, 1, 3], max_channel=3, thickness=[1, 8], bezier=False) #[pixel]
        self.pivot_gen = GenPivots(max_pts=point_conf['max_pieces'], map_region = map_conf['map_region'], resolution=map_conf['map_resolution'])#
        self.transforms = transforms
 
        super(AV2PMapNetSemanticDataset, self).__init__()
    
    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.samples)
    
    def load_annotations(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        # import pdb; pdb.set_trace()
        start_time = time()
        ann = create_av2_infos_mp(root_path=self.av2_root,  dest_path=None, split= self.split_mode, num_multithread=64)
        samples = ann['samples']
        samples = samples[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
       
        return samples, ann['id2map']
    
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
        
        map_geoms = self.map_extractor.get_map_geom(log_id, sample['e2g_translation'], 
                sample['e2g_rotation'])

 
        
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
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_geoms, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': sample['e2g_rotation'],
            'scene_name': sample['scene_name'],
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
        
    
        
    '''def pipeline(self, data):
        # VectorizeMap
        #data = self.vectorize_map(data)
        # LoadMultiViewImagesFromFiles
        data = self.load_images(data)
        #print(data['img'][0].shape,
        # data['img'][1].shape, data['img'][2].shape, data['img'][3].shape, data['img'][4].shape, data['img'][5].shape, data['img'][6].shape)
        
        # PhotoMetricDistortionMultiViewImage
        # only train
        #if self.split_mode == 'train':
        #    data = self.aug_images(data)
        
        #imgs = data['img'] # N,3,H,W
        
        data = self.norm_images(data)
        # ResizeMultiViewImages

        #data = self.pad_images(data)
        #print(data['img'][0].shape,
        # data['img'][1].shape, data['img'][2].shape, data['img'][3].shape, data['img'][4].shape, data['img'][5].shape, data['img'][6].shape)
        

        #data = self.resize_images(data)
       
        #print(data['img'][0].shape,
        # data['img'][1].shape, data['img'][2].shape, data['img'][3].shape, data['img'][4].shape, data['img'][5].shape, data['img'][6].shape)
        

       
       
        # PadMultiViewImages
        
        
        # format
        #data = self.format(data)
        

        return data
    '''

    def __getitem__(self, idx):
        data = self.get_sample(idx)
        data = self.load_images(data)
        data = self.get_lidar_data(data, idx)
        
        ego2imgs = data['ego2img'] # list of 7 cameras, 4x4 array
        ego2global_tran = data['ego2global_translation'] # 3
        ego2global_rot = data['ego2global_rotation'] # 3x3
        timestamp = torch.tensor(data['timestamp'] / 1e9) # TODO: verify ns?
        scene_id = data['scene_name']
        rots = torch.stack([torch.Tensor(ego2img[:3, :3]) for ego2img in ego2imgs]) # 7x3x3
        trans = torch.stack([torch.Tensor(ego2img[:3, 3]) for ego2img in ego2imgs]) # 7x3
        post_rots = rots
        post_trans = trans
        intrins = torch.stack([torch.Tensor(intri) for intri in data['cam_intrinsics']])
        extrins = torch.stack([torch.Tensor(extri) for extri in data['cam_extrinsics']])
        car_trans = torch.tensor(ego2global_tran) # [3]
        pos_rotation = Quaternion(matrix=ego2global_rot)
        yaw_pitch_roll = torch.tensor(pos_rotation.yaw_pitch_roll)
        
        lidar_data = data['lidar']
        lidar_mask = data['lidar_mask']
        
        targets = self.get_semantic_map(data, idx)
        # for uniform interface
      
        #print(len(data['img']), data['img'][0].shape)   # [768, 1024, 3]
        item = dict(images= data['img'], lidars=lidar_data,  lidar_mask= lidar_mask, targets=targets,
                    extra_infos=dict(token=data['token'], map_size=self.ego_size),
                    extrinsic=np.stack(extrins, axis=0), intrinsic=np.stack(intrins, axis=0))
        
        '''lidar_filename = self.av2_root + '/lidar/'
        if not os.path.exists(lidar_filename):
            os.makedirs(lidar_filename)
        np.savetxt(lidar_filename +  str(idx)+str('+')+data["log_id"] + '.txt', lidar_data, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)
         
        #if(idx%200==0):
        cam_filename = self.av2_root + '/cam/'
        if not os.path.exists(cam_filename):
            os.makedirs(cam_filename)
        for i in range(7):
           # print( item['images'][i].size(), item['images'][i]) #3, 640, 640])   
            #print(type(imgs))
            #im = item['images'][i]
            #im = Image.fromarray(np.uint8(item['images'][i].permute(1, 2, 0).numpy()))
            #print(im )
            #print(cam_filename +  str(idx)+str('+')+data["log_id"] + str(self.img_key_list[i])+ '.png')
            im = Image.fromarray(item['images'][i])
            im.save(cam_filename +  str(idx)+str('+')+data["log_id"] + str(self.img_key_list[i])+ '.png')
        '''
            
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def get_semantic_map(self, data, idx):

        vectors = data['map_geoms']
       
                    
        semantic_masks, instance_masks, instance_vec_points, instance_ctr_points =  self.raster_map.convert_vec_to_mask(vectors)
        pivots, pivot_lengths = self.pivot_gen.pivots_generate(instance_vec_points)
    
        targets = dict(masks=instance_masks[1], points=pivots, valid_len=pivot_lengths) 
        
        #visual_map_gt_pivot(self.map_region, targets, str(idx)+str('+')+data["log_id"], self.av2_root)
        #visual_map_gt(self.map_region, instance_vec_points, str(idx)+str('+')+data["log_id"] , self.av2_root)
    
        return targets

