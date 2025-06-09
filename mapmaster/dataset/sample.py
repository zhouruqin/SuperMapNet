import open3d as o3d
import numpy as np




def sampled(path):
    pc = np.frombuffer(open(path, "rb").read(), dtype=np.float32)
    pc = pc.reshape(-1, 5)[:, :4]
    print(pc.shape)
    point = pc[:, :3]
    intensity = np.zeros(point.shape)
    intensity[:, 0] = pc[:, 3]
    #print(intensity)
    pcd=o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(point)#转换格式    
    pcd.colors = o3d.utility.Vector3dVector(intensity)
    pcd_down =  pcd.voxel_down_sample(voxel_size=0.1)
    path_new = path + ".pcd"
    o3d.io.write_point_cloud(path_new, pcd_down)
    
    
if __name__ == '__main__':
    sampled('/zrq/PivotNet/data/nuscenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin')