import torch
import torch.nn as nn
import torch_scatter

from mapmaster.dataset.voxel import points_to_voxels
from icecream import ic
#import open3d as o3d
#from open3d import*
import numpy as np

class PillarBlock(nn.Module):
    def __init__(self,  idims=64, dims=64, num_layers=1,
                 stride=1):
        super(PillarBlock, self).__init__()
        layers = []
        self.idims = idims
        self.stride = stride
        for i in range(num_layers):
            layers.append(nn.Conv2d(self.idims, dims, 3, stride=self.stride,
                                    padding=1, bias=False))
            layers.append(nn.BatchNorm2d(dims))
            layers.append(nn.ReLU(inplace=True))
            #layers.append(nn.Dropout(dropout))
            self.idims = dims
            self.stride = 1
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PointNet(nn.Module):
    def __init__(self, idims=64, odims=64):
        super(PointNet, self).__init__()
        self.pointnet = nn.Sequential(
            nn.Conv1d(idims, odims, kernel_size=1, bias=False),
            nn.BatchNorm1d(odims),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
        )

    def forward(self, points_feature, points_mask):
        batch_size, num_points, num_dims = points_feature.shape
        points_feature = points_feature.permute(0, 2, 1)
        mask = points_mask.view(batch_size, 1, num_points)
        return self.pointnet(points_feature) * mask


class PointPillar(nn.Module):
    def __init__(self, C, xbound, ybound, zbound, embedded_dim=16, direction_dim=37, ppdim=15):
        super(PointPillar, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.embedded_dim = embedded_dim
        self.pn = PointNet(dropout, ppdim, 64)
        self.block1 = PillarBlock(dropout=dropout, idims = 64, dims=64, num_layers=2, stride=1)
        self.block2 = PillarBlock(dropout=dropout, idims =64, dims=128, num_layers=3, stride=2)
        self.block3 = PillarBlock(dropout=dropout, idims = 128, dims=256, num_layers=3, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(128, C, 1),
        )
        self.instance_conv_out = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(128, embedded_dim, 1),
        )
        self.direction_conv_out = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(dropout),
            nn.Conv2d(128, direction_dim, 1),
        )

    def forward(self, points, points_mask ):
        points_xyz = points[:, :, :3]
        points_feature = points[:, :, 3:]
        voxels = points_to_voxels(
            points_xyz, points_mask, self.xbound, self.ybound, self.zbound
        )
        points_feature = torch.cat(
            [points,  # 5
             torch.unsqueeze(voxels['voxel_point_count'], dim=-1),  # 1
             voxels['local_points_xyz'],  # 3
             voxels['point_centroids'],  # 3
             points_xyz - voxels['voxel_centers'],  # 3
             ], dim=-1
        )
        points_feature = self.pn(points_feature, voxels['points_mask'])
        voxel_feature = torch_scatter.scatter_mean(
            points_feature,
            torch.unsqueeze(voxels['voxel_indices'], dim=1),
            dim=2,
            dim_size=voxels['num_voxels'])
        batch_size = points.size(0)
        voxel_feature = voxel_feature.view(
            batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])
        voxel_feature1 = self.block1(voxel_feature)
        voxel_feature2 = self.block2(voxel_feature1)
        voxel_feature3 = self.block3(voxel_feature2)
        voxel_feature1 = self.up1(voxel_feature1)
        voxel_feature2 = self.up2(voxel_feature2)
        voxel_feature3 = self.up3(voxel_feature3)
        voxel_feature = torch.cat( [voxel_feature1, voxel_feature2, voxel_feature3], dim=1)
        return self.conv_out(voxel_feature)#.transpose(3, 2), self.instance_conv_out(voxel_feature).transpose(3, 2), self.direction_conv_out(voxel_feature).transpose(3, 2)


class PointPillarEncoder(nn.Module):
    def __init__(self, C, xbound, ybound, zbound, ppdim=4):
        super(PointPillarEncoder, self).__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.pn = PointNet( ppdim, 64)
        self.block1 = PillarBlock(idims =64, dims=64, num_layers=2, stride=2)
        self.block2 = PillarBlock(idims =64, dims=128, num_layers=3, stride=2)
        self.block3 = PillarBlock(idims =128, dims=C, num_layers=3, stride=2)
        '''self.up1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self. up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(C, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(448, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, C, 1),
        )'''

    def forward(self, points, points_mask):
        points_xyz = points[:, :, :3]# points_xyz.shape: torch.Size([1, 81920, 3])
    
        points_feature = points[:, :, 3:]
        voxels = points_to_voxels(
            points_xyz, points_mask, self.xbound, self.ybound, self.zbound
        )
       
        points_feature = points[:, :, :4]
        '''torch.cat(#
            [points,  # 5
             torch.unsqueeze(voxels['voxel_point_count'], dim=-1),  # 1
             voxels['local_points_xyz'],  # 3
             voxels['point_centroids'],  # 3
             points_xyz - voxels['voxel_centers'],  # 3
             ], dim=-1
        )'''
        points_feature = self.pn(points_feature, voxels['points_mask'])#points_feature.shape: torch.Size([2, 64, 8192])
        voxel_feature = torch_scatter.scatter_mean(
            points_feature,
            torch.unsqueeze(voxels['voxel_indices'], dim=1),
            dim=2,
            dim_size=voxels['num_voxels'])
        batch_size = points.size(0)
        voxel_feature = voxel_feature.view(
            batch_size, -1, voxels['grid_size'][0], voxels['grid_size'][1])# voxel_feature.shape: torch.Size([2, 64, 120000])
        voxel_feature1 = self.block1(voxel_feature)
        voxel_feature2 = self.block2(voxel_feature1)
        neck_feature = self.block3(voxel_feature2)
        '''voxel_feature1 = self.up1(voxel_feature1)
        voxel_feature2 = self.up2(voxel_feature2)
        voxel_feature3 = self.up3(neck_feature)
        voxel_feature = torch.cat(
            [voxel_feature1, voxel_feature2, voxel_feature3], dim=1)#voxel_feature.shape: torch.Size([2, 448, 600, 200])
        # ic(voxel_feature.shape,self.conv_out(voxel_feature).transpose(3, 2).shape)#ic| voxel_feature.shape: torch.Size([1, 448, 600, 200]), self.conv_out(voxel_feature).transpose(3, 2).shape: torch.Size([1, 128, 200, 600])
        # exit(0) '''
        #print(neck_feature.size())
        return   neck_feature # self.conv_out(voxel_feature), 
