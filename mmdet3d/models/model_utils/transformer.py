from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING, MultiheadAttention
from torch import nn as nn
import copy
from typing import Optional
from torch import Tensor, nn
import torch
from functools import partial
import torch.nn.functional as F
@ATTENTION.register_module()
class GroupFree3DMHA(MultiheadAttention):
    """A warpper for torch.nn.MultiheadAttention for GroupFree3D.

    This module implements MultiheadAttention with identity connection,
    and positional encoding used in DETR is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads. Same as
            `nn.MultiheadAttention`.
        attn_drop (float): A Dropout layer on attn_output_weights. Default 0.0.
        proj_drop (float): A Dropout layer. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropOut', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super().__init__(embed_dims, num_heads, attn_drop, proj_drop,
                         dropout_layer, init_cfg, batch_first, **kwargs)

    def forward(self,
                query,
                key,
                value,
                identity,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `GroupFree3DMHA`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims]. Same in `nn.MultiheadAttention.forward`.
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Same in `nn.MultiheadAttention.forward`. Defaults to None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        if hasattr(self, 'operation_name'):
            if self.operation_name == 'self_attn':
                value = value + query_pos
            elif self.operation_name == 'cross_attn':
                value = value + key_pos
            else:
                raise NotImplementedError(
                    f'{self.__class__.name} '
                    f"can't be used as {self.operation_name}")
        else:
            value = value + query_pos

        return super(GroupFree3DMHA, self).forward(
            query=query,
            key=key,
            value=value,
            identity=identity,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)


@POSITIONAL_ENCODING.register_module()
class ConvBNPositionalEncoding(nn.Module):
    """Absolute position embedding with Conv learning.

    Args:
        input_channel (int): input features dim.
        num_pos_feats (int): output position features dim.
            Defaults to 288 to be consistent with seed features dim.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass.

        Args:
            xyz (Tensor)： (B, N, 3) the coordinates to embed.

        Returns:
            Tensor: (B, num_pos_feats, N) the embeded position features.
        """
        xyz = xyz.permute(0, 2, 1)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
WEIGHT_INIT_DICT = {
    'xavier_uniform': nn.init.xavier_uniform_,
}
class BatchNormDim1Swap(nn.BatchNorm1d):
    
    def forward(self, x):
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)
        x = super(BatchNormDim1Swap, self).forward(x)
        x = x.permute(2, 0, 1)
NORM_DICT = {
    'bn': BatchNormDim1Swap,
    'bn1d': nn.BatchNorm1d,
    'id': nn.Identity,
    'ln': nn.LayerNorm,
}
ACTIVATION_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leakyrelu': partial(nn.LeakyReLU, negative_slope=0.1)
}
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers,
                 norm=None, weight_init_name="xavier_uniform"):
        super().__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self._reset_parameters(weight_init_name)

    def _reset_parameters(self, weight_init_name):
        func = WEIGHT_INIT_DICT[weight_init_name]
        for p in self.parameters():
            if p.dim() > 1:
                func(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                xyz: Optional [Tensor] = None,
                transpose_swap: Optional[bool] = False,
                return_attn_weights: Optional [bool] = False,
                ):
        attns = []
        if transpose_swap:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            if pos is not None:
                pos = pos.flatten(2).permute(2, 0, 1)
        output = src
        orig_mask = mask
        if orig_mask is not None and isinstance(orig_mask, list):
            assert len(orig_mask) == len(self.layers)
        elif orig_mask is not None:
            orig_mask = [mask for _ in range(len(self.layers))]

        for idx, layer in enumerate(self.layers):
            if orig_mask is not None:
                mask = orig_mask[idx]
                # mask must be tiled to num_heads of the transformer
                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)
            if return_attn_weights:
                output, attn = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos, return_attn_weights=True)
                attns.append(attn)
            else:
                output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        if transpose_swap:
            output = output.permute(1, 2, 0).view(bs, c, h, w).contiguous()

        xyz_inds = None

        if return_attn_weights:
            attns = torch.stack(attns)
            return xyz, output, xyz_inds, attns

        return xyz, output, xyz_inds
    

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead=4, dim_feedforward=128,
                 dropout=0.1, dropout_attn=None,
                 activation="gelu", normalize_before=True, norm_name="ln",
                 use_ffn=True,
                 ffn_use_bias=True):
        super().__init__()
        if dropout_attn is None:
            dropout_attn = dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout_attn)
        self.use_ffn = use_ffn
        if self.use_ffn:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=ffn_use_bias)
            self.dropout = nn.Dropout(dropout, inplace=True)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias=ffn_use_bias)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.norm2 = NORM_DICT[norm_name](d_model)
            self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.norm1 = NORM_DICT[norm_name](d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.activation = ACTIVATION_DICT[activation]()
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        value = src
        src2 = self.self_attn(q, k, value=value, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        if self.use_norm_fn_on_input:
            src = self.norm1(src)
        if self.use_ffn:
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    return_attn_weights: Optional [Tensor] = False):

        src2 = self.norm1(src)
        value = src2
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=value, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        if self.use_ffn:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        if return_attn_weights:
            return src, attn_weights
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                return_attn_weights: Optional [Tensor] = False):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, return_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

    def extra_repr(self):
        st = ""
        if hasattr(self.self_attn, "dropout"):
            st += f"attn_dr={self.self_attn.dropout}"
        return st

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x