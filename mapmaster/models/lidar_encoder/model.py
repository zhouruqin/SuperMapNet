import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.lidar_encoder.pointpillar import PointPillarEncoder
import icecream as ic



class LiDARPointPillarEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(LiDARPointPillarEncoder, self).__init__()
        self.pp = PointPillarEncoder( **kwargs)  

    def forward(self, inputs):
        lidar_data = inputs["lidars"]
        lidar_mask = inputs["lidar_mask"]
        neck_feature = self.pp(lidar_data, lidar_mask)
        #ic(neck_feature.size())
        
        return {
            "lidar_enc_features": list(neck_feature),
        }
