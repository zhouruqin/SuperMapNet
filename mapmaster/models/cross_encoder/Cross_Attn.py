import icecream as ic
from inplace_abn import InPlaceABNSync
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
  
 
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

       

class LidarPred(nn.Module):
    def __init__(self, tgt_shape,  dropout = 0, use_cross=True, num_heads=8, pos_emd=True, neck_dim=256, cross_dim=256):
        super(LidarPred, self).__init__()
        self.use_cross = use_cross
        self.pos_emd = pos_emd
        '''self.conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(256)

        self.conv21 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(256)

        self.conv41d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv21d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(256)

        self.conv11d = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)'''


        if self.use_cross:
            self.conv_reduce = nn.Conv2d(neck_dim, neck_dim, kernel_size=1)
            if pos_emd:
                self.pe_lidar = positionalencoding2d(
                    neck_dim, tgt_shape[0], tgt_shape[1]).cuda()
                self.pe_img = positionalencoding2d(
                    neck_dim, tgt_shape[0], tgt_shape[1]).cuda()
            self.multihead_attn = nn.MultiheadAttention(
                    neck_dim, num_heads, dropout=0.3, batch_first=True)
            self.conv = nn.Sequential(
                        nn.Conv2d(neck_dim+cross_dim,
                                  neck_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(neck_dim),
                        nn.ReLU(inplace=True),
                        #nn.Dropout(dropout),
                    )
            self.conv_cross = nn.Sequential(
                nn.Conv2d(neck_dim,
                            cross_dim, kernel_size=1),
                nn.BatchNorm2d(cross_dim),
                nn.ReLU(inplace=True),
                #nn.Dropout(dropout),  #
            )
            
           
            self.multihead_attn_img = nn.MultiheadAttention(
                    neck_dim, num_heads, dropout=0.3, batch_first=True)
            self.conv_img = nn.Sequential(
                        nn.Conv2d(neck_dim+cross_dim,
                                  neck_dim, kernel_size=3, padding=1),
                        nn.BatchNorm2d(neck_dim),
                        nn.ReLU(inplace=True),
                        #nn.Dropout(dropout),
                    )
            self.conv_cross_img = nn.Sequential(
                nn.Conv2d(neck_dim,
                            cross_dim, kernel_size=1),
                nn.BatchNorm2d(cross_dim),
                nn.ReLU(inplace=True),
                #nn.Dropout(dropout),
            )
            
            
    def cross_attention(self, x, img_feature):
        B, C, H, W = x.shape
        if self.pos_emd:
            pe_lidar = self.pe_lidar.repeat(B, 1, 1, 1)
            pe_img = self.pe_img.repeat(B, 1, 1, 1)
            x = x + pe_lidar
            img_feature = img_feature + pe_img
        query = x.reshape(B, C, -1).permute(0, 2, 1)
        key = img_feature.reshape(B, C, -1).permute(0, 2, 1)
        
        attn_output, attn_output_weights = self.multihead_attn(
            query, key, key)
        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)
        attn_output = self.conv_cross(attn_output)
        fused_feature = torch.cat([x, attn_output],dim=1)
        fused_feature = self.conv(fused_feature)
        
        attn_output_img, attn_output_weights_img = self.multihead_attn_img(
            key, query, query)
        attn_output_img = attn_output_img.permute(0, 2, 1).reshape(B, C, H, W)
        attn_output_img = self.conv_cross_img(attn_output_img)
        #print(img_feature.size(), x.size(), attn_output_img.size())
        fused_feature_img = torch.cat([img_feature, attn_output_img],  dim=1)
        fused_feature_img = self.conv_img(fused_feature_img)
        
        
        return fused_feature_img,  fused_feature

    def forward(self, x, img_feature=None):
        
        '''x11 = F.relu(self.bn11(self.conv11(x)))
        x1p, id1 = self.max_pool(x11)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x2p, id2 = self.max_pool(x21)

        x41 = F.relu(self.bn41(self.conv41(x2p))) # bottleneck'''
        #if self.use_cross:
        img_feature = self.conv_reduce(img_feature)
        img, x41 = self.cross_attention(x, img_feature)

        '''x41d = F.relu(self.bn41d(self.conv41d(x41)))

        x3d = self.max_unpool(x41d, id2)
        x21d = F.relu(self.bn21d(self.conv21d(x3d)))

        x2d = self.max_unpool(x21d, id1)

        x11d = self.conv11d(x2d)'''

        return img, x41


