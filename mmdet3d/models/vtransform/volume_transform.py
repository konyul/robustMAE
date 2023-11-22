import torch
from mmcv.runner import force_fp32
from torch import nn

from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer


__all__ = ["VolumeTransform"]


class VolumeTransform(nn.Module):
    def __init__(
        self,
        embed_dims,
        in_channels,
        mask_ratio,
    ) -> None:
        super(VolumeTransform, self).__init__()
        
        self.mask_token = None
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self._init_weights()
    
    def _init_weights(self):
        
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=.02)
        
    def forward(
        self, 
        camera_x,
        img_shape,
        camera_ids_restore,
        img_metas
    ):
        B, N, C, H, W = img_shape
        if self.mask_token is not None:
            mask_tokens = self.mask_token.repeat(camera_x.shape[0], camera_ids_restore.shape[1] - camera_x.shape[1], 1)
            camera_x = torch.cat([camera_x, mask_tokens], dim=1)
        camera_x = torch.gather(camera_x, dim=1, index=camera_ids_restore.unsqueeze(-1).repeat(1, 1, camera_x.shape[2]))  # unshuffle
        camera_x = camera_x.permute(0, 2, 1).view(B, N, -1, H//32, W//32)
        return camera_x
    
def build_vtransform(cfg):
    model_name = cfg.pop('type')
    model_dict={'VolumeTransform': VolumeTransform}
    model_class = model_dict[model_name]
    model = model_class(**cfg)
    return model