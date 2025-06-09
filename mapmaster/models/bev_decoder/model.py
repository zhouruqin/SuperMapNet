import torch
import numpy as np
import torch.nn as nn
from mapmaster.models.bev_decoder.transformer import Transformer
from mapmaster.models.bev_decoder.deform_transformer import DeformTransformer
import icecream as ic

class TransformerBEVDecoder(nn.Module):
    def __init__(self, key='im_bkb_features', **kwargs):
        super(TransformerBEVDecoder, self).__init__()
        self.bev_encoder = Transformer(**kwargs)
        self.key = key

    def forward(self, inputs):
        assert "im_bkb_features" in inputs    #im_bkb_features
        feats = inputs["im_bkb_features"]
        print(len(feats))
        for i in range(len(feats)):
            feats[i] = feats[i].reshape(*inputs["images"].shape[:2], *feats[i].shape[-3:])
            feats[i] = feats[i].permute(0, 2, 3, 1, 4)
            feats[i] = feats[i].reshape(*feats[i].shape[:3], -1)

        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'ida_mats': inputs.get('ida_mats', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }

        _, _, bev_feats = self.bev_encoder(feats[-1], cameras_info=cameras_info)

        return {"bev_enc_features": list(bev_feats)}

class DeformTransformerBEVEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(DeformTransformerBEVEncoder, self).__init__()
        self.bev_encoder = DeformTransformer(**kwargs)

    def forward(self, inputs):
        assert "im_bkb_features" in inputs    #im_bkb_features
        feats = inputs["im_bkb_features"]
        #mask = inputs["im_masks"].permute(0, 2, 1, 3)
        #mask = mask.reshape(*mask.shape[:2], -1)
        #print('1',mask.size())
        for i in range(len(feats)):
            feats[i] = feats[i].reshape(*inputs["images"].shape[:2], *feats[i].shape[-3:])
            feats[i] = feats[i].permute(0, 2, 3, 1, 4)
            feats[i] = feats[i].reshape(*feats[i].shape[:3], -1)
        cameras_info = {
            'extrinsic': inputs.get('extrinsic', None),
            'intrinsic': inputs.get('intrinsic', None),
            'do_flip': inputs['extra_infos'].get('do_flip', None)
        }
        # src_feats: (N, H1 * W1, C)  tgt_feats: # (M, N, H2 * W2, C)
        #ic(feats.size())#[4, 1, 256, 64, 32]src_masks =mask.bool(),
        _, _, bev_feats = self.bev_encoder(feats,   cameras_info=cameras_info)
        
        
        return {#
            "bev_enc_features": list(bev_feats),#img_enc_features    bev_enc_features
        }
