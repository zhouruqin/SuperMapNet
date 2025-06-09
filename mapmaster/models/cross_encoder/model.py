import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.cross_encoder.Cross_Attn import LidarPred


class CrossEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(CrossEncoder, self).__init__()
        self.lidar_pred = LidarPred(**kwargs)


    def forward(self, inputs):
        '''img_feat = inputs["im_bkb_features"] [-1]
        img_feat = img_feat.reshape(*inputs['images'].shape[:2], *img_feat.shape[-3:])
        img_feat = img_feat.permute(0, 2, 3, 1, 4) 
        #print('3', img_feats.size())
        img_feat = img_feat.reshape(*img_feat.shape[:3],  -1)'''
        img_feat = inputs["img_enc_features"] [-1]#
        lidar_enc_feat = (inputs["lidar_enc_features"][-1]).unsqueeze(0) 
       
        img_bev_feat, lidar_feature = self.lidar_pred(lidar_enc_feat, img_feat)
        #print(lidar_feature.size())
        
        return {
            "img_bev_feat": img_bev_feat,
            "lidar_bev_feat": lidar_feature
        }

  