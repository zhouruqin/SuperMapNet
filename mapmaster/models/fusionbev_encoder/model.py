import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.fusionbev_encoder.BEVFusion import AlignFAnew, Conv


class BevFusionEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(BevFusionEncoder, self).__init__()
        self.fuser_AlignFA = AlignFAnew(**kwargs)

    def forward(self, inputs):
        img_bev_feat = inputs["img_bev_feat"]
        lidar_enc_feat = (inputs["lidar_bev_feat"][-1]).unsqueeze(0) 
        #print(img_bev_feat.size(), lidar_enc_feat.size())
        
        bev = self.fuser_AlignFA(img_bev_feat, lidar_enc_feat)
        #print(bev.size())
        
        return {
            "bev_enc_features": list(bev.unsqueeze(0)),
        }


class ConcatBEV(nn.Module):
    def __init__(self, **kwargs):
        super(ConcatBEV, self).__init__()
        self.conv =  Conv(**kwargs) 
       

    def forward(self, inputs):
        img_bev_feat = inputs["img_enc_features"] [-1]
        lidar_enc_feat = (inputs["lidar_enc_features"][-1]).unsqueeze(0)
       
        bev = self.conv(img_bev_feat, lidar_enc_feat)
      
        
        return {
            "bev_enc_features": list(bev.unsqueeze(0)),
        }
