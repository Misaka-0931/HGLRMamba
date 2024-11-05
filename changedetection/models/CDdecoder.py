import torch
import torch.nn as nn

import torch.nn.functional as F
from changedetection.models.HelperModule import GatedFusionBlock, HGLRBlock, DeepSemanticRefineBlock

class CDdecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(CDdecoder, self).__init__()

        self.hglr_4 = HGLRBlock(in_features=128, drop_path=0.2, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer, **kwargs)
        self.hglr_3 = HGLRBlock(in_features=128, drop_path=0.2, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer, **kwargs)
        self.hglr_2 = HGLRBlock(in_features=128, drop_path=0.2, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer,  **kwargs)
        self.hglr_1 = HGLRBlock(in_features=128, drop_path=0.2, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer, **kwargs)


        self.dsr_1 = DeepSemanticRefineBlock(dim=128, drop=0.1, drop_path=0.1, num_heads=4)

        self.fuse_4 = GatedFusionBlock(in_dim=encoder_dims[-1] * 2, out_dim=128, drop_path=0.1)
        self.fuse_3 = GatedFusionBlock(in_dim=encoder_dims[-2] * 2, out_dim=128, drop_path=0.1)
        self.fuse_2 = GatedFusionBlock(in_dim=encoder_dims[-3] * 2, out_dim=128, drop_path=0.1)
        self.fuse_1 = GatedFusionBlock(in_dim=encoder_dims[-4] * 2, out_dim=128, drop_path=0.1)

    def _upsample_add(self, x, y):
        # x.shape : B C  h/2 w/2
        # y.shape : B C h w
        _, _, H, W = y.shape
        x = F.interpolate(x, size=(H, W), mode='bilinear')
        return  x + y

    def forward(self, pre_features, post_features):

        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        # [8, 94, 64, 64], [8, 192, 32, 32], [8, 384, 16, 16], [8, 768, 8, 8]
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''

        p4 = self.fuse_4(torch.concat([pre_feat_4, post_feat_4], dim=1))
        p3 = self.fuse_3(torch.concat([pre_feat_3, post_feat_3], dim=1))
        p2 = self.fuse_2(torch.concat([pre_feat_2, post_feat_2], dim=1))
        p1 = self.fuse_1(torch.concat([pre_feat_1, post_feat_1], dim=1))

        p4 = self.hglr_4(p4)


        '''
            Stage II
        '''
        p3 = self._upsample_add(p4, p3)
        p3 = self.hglr_3(p3)

        '''
            Stage III
        '''
        p2 = self._upsample_add(p3, p2)
        p2 = self.hglr_2(p2)

        '''
            Stage IV
        '''

        p1 = self._upsample_add(p2, p1)
        p1 = self.hglr_1(p1)

        p1 = self.dsr_1(p1, p4)

        return p1