import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from changedetection.models.HGLR_backbone import Backbone_VSSM
from classification.models.vmamba import LayerNorm2d
from models.CDdecoder import CDdecoder
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


class HGLRMamba(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(HGLRMamba, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )


        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = CDdecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Sequential(
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1),
        )


    def forward(self, pre_data, post_data):
        # Encoder processing
        # print(pre_data.shape, post_data.shape)
        pre_features, post_features = self.encoder(pre_data, post_data)
        output = self.decoder(pre_features, post_features)
        output = self.main_clf(output)
        # B 128 64 64

        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output

if __name__ == '__main__':
    model = HGLRMamba(pretrained=None, localRefine=False).cuda()
    inputs = (torch.randn(2, 3, 256, 256).cuda(), torch.randn(2, 3, 256, 256).cuda())
    print(flop_count_str(FlopCountAnalysis(model, inputs)))