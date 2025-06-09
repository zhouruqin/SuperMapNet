import torch
import torch.nn as nn
import torch.nn.functional as F
from mapmaster.models.backbone.resnet import ResNet
from mapmaster.models.backbone.efficientnet import EfficientNet
from mapmaster.models.backbone.swin_transformer import SwinTransformer
from mapmaster.models.backbone.bifpn import BiFPN
from mapmaster.models.backbone.sam2feature import SAM2Feature
from mmcv.cnn import build_conv_layer
import numpy as np

class ResNetBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(ResNetBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = ResNet(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape
        self.bkb.init_weights()

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
        bkb_features = list(self.bkb(images)[-self.ret_layers:])
        nek_features = self.fpn(bkb_features) if self.fpn is not None else None
        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}


class EfficientNetBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(EfficientNetBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = EfficientNet.from_pretrained(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape
        del self.bkb._conv_head
        del self.bkb._bn1
        del self.bkb._avg_pooling
        del self.bkb._dropout
        del self.bkb._fc

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
        endpoints = self.bkb.extract_endpoints(images)
        bkb_features = []
        for i, (key, value) in enumerate(endpoints.items()):
            if i > 0:
                bkb_features.append(value)
        bkb_features = list(bkb_features[-self.ret_layers:])
        nek_features = self.fpn(bkb_features) if self.fpn is not None else None
        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}


class SwinTRBackbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(SwinTRBackbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb = SwinTransformer(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape

    def forward(self, inputs):
        images = inputs["images"]
        images = images.view(-1, *images.shape[-3:])
       
        bkb_features = list(self.bkb(images)[-self.ret_layers:])
        nek_features = None
        if self.fpn is not None:
            nek_features = self.fpn(bkb_features)
        else:
            if self.up_shape is not None:
                nek_features = [torch.cat([self.up_sample(x, self.up_shape) for x in bkb_features], dim=1)]
        #print(len(bkb_features), bkb_features[-1].size())#  2, [6, 768, 16, 28]
        return {"im_bkb_features": bkb_features, "im_nek_features": nek_features}

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)


class SAM2Backbone(nn.Module):
    def __init__(self, bkb_kwargs, fpn_kwarg=None, up_shape=None, ret_layers=1):
        super(SAM2Backbone, self).__init__()
        assert 0 < ret_layers < 4
        self.ret_layers = ret_layers
        self.bkb, self.mask_generator = SAM2Feature.from_pretrained(**bkb_kwargs)
        self.fpn = None if fpn_kwarg is None else BiFPN(**fpn_kwarg)
        self.up_shape = None if up_shape is None else up_shape
        
        '''cfg1 = dict(
                type='Conv2d',
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride = 2,
                padding =1
            )
        self.down_conv1 = build_conv_layer(cfg1)
        cfg2 = dict(
                type='Conv2d',
                in_channels=256,
                out_channels=768,
                kernel_size=3,
                stride = 2,
                padding =1
            )
        self.down_conv2 = build_conv_layer(cfg2)'''

    def forward(self, inputs):
        images = inputs["images"]
        
        images = images.view(-1, *images.shape[-3:])
        imgs = [images[i].permute(1, 2, 0).cpu().numpy() for i in range(images.size()[0])]
        self.bkb.set_image_batch(imgs)
        bkb_features  = self.bkb._features['high_res_feats']
        #print(len(bkb_features), bkb_features[0].size(), bkb_features[1].size())#[6, 64, 64, 112], "im_masks": masks
        if self.up_shape is not None:
                bkb_features_new = [self.up_sample(bkb_features[-1], self.up_shape) ]
        
        '''max_masks = []
        for i in range(images.size()[0]):
            #print(i)
            mask_i = self.mask_generator.generate(imgs[i])
            max_pixel = 0
            max_idx = 0
            for j in range(len(mask_i)):
                selected_mask= mask_i[j]['segmentation']
                #print(type(selected_mask), selected_mask.shape)#[512, 896]
                #选择像素最多的作为地面
                #print(np.sum(selected_mask,))
                tmp = np.sum(selected_mask)
                if max_pixel < tmp:
                    max_pixel =  tmp
                    max_idx = j
                print(j)
            print(max_idx, len(mask_i))
            max_masks.append(torch.from_numpy(mask_i[max_idx]['segmentation']))

        max_masks = torch.stack( max_masks, 0).float().cuda()
        samped_mask = F.interpolate(max_masks.unsqueeze(0), size=self.up_shape,  mode="nearest")   ''' 
            
        #print(masks.size(), imgs[0].size())
        #print(len(bkb_features), bkb_features[0].size(), bkb_features[-1].size())
        #bkb_features_new = [ self.down_conv2(self.down_conv1(bkb_features[-1]))]
        nek_features = self.fpn(bkb_features_new[-1]) if self.fpn is not None else None
        #print(len(bkb_features_new),bkb_features_new[-1].size(),  nek_features, samped_mask.size())#[6, 64, 64, 112], "im_masks": samped_mask
        return {"im_bkb_features": bkb_features_new, "im_nek_features": nek_features}

    def up_sample(self, x, tgt_shape=None):
        tgt_shape = self.tgt_shape if tgt_shape is None else tgt_shape
        if tuple(x.shape[-2:]) == tuple(tgt_shape):
            return x
        return F.interpolate(x, size=tgt_shape, mode="bilinear", align_corners=True)