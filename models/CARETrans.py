import pdb
import time
from functools import partial
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
import time

class TokenMixer(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, mem_dim=None, att_head_dim=1, if_att=False, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        self.att_head_dim = att_head_dim
        self.if_att = if_att
        self.gc = mem_dim
        self.att_head_num = int(self.gc/att_head_dim)
        
        if self.if_att:
            self.split_indexes = (in_channels - 3*self.gc, self.gc, self.gc, self.gc)
            self.dwconv1 = nn.Conv2d(self.gc, self.gc, kernel_size=3, padding=1, groups=self.gc)
            self.dwconv2 = nn.Conv2d(self.gc, self.gc, kernel_size=7, padding=3, groups=self.gc)
            self.qk = nn.Conv2d(self.gc, 2*self.gc, kernel_size=(1, 1))
            self.elu = nn.ELU()

        else:
            self.split_indexes = (in_channels - 3*self.gc, self.gc, self.gc, self.gc)
            self.dwconvatt_1 = nn.Conv2d(self.gc, self.gc, kernel_size=(11, 1), padding=(5, 0), groups=self.gc)
            self.dwconvatt_2 = nn.Conv2d(self.gc, self.gc, kernel_size=(1, 11), padding=(0, 5), groups=self.gc)
            self.dwconv1 = nn.Conv2d(self.gc, self.gc,  kernel_size=3, padding=1, groups=self.gc)
            self.dwconv2 = nn.Conv2d(self.gc, self.gc, kernel_size=7, padding=3, groups=self.gc)

    def grasp_memory(self, x):
        mem, feat = torch.split(x, [self.split_indexes[0], sum(self.split_indexes[1:])], dim=1)
        return mem, feat
    
    def asymmetric_decouple(self, x):
        feat_att, feat_conv1, feat_conv2 = torch.split(x, self.split_indexes[1:], dim=1)
        return feat_att, feat_conv1, feat_conv2 
    
    def forward(self, x):
        B, _, H, W = x.shape
        if self.if_att:
            mem, feats = self.grasp_memory(x)
            feat_att, feat_conv1, feat_conv2 = self.asymmetric_decouple(feats)
            feat_conv1 = self.dwconv1(feat_conv1)
            feat_conv2 = self.dwconv2(feat_conv2)
            x_qk = self.qk(feat_att).reshape(B, -1, H*W)
            (q, k), v = x_qk.split(self.gc, dim=1), feat_att.reshape(B, -1, H*W)
            q = q.reshape(B, self.att_head_num, self.att_head_dim, H*W).transpose(-2, -1) # (B, N, D)
            k = k.reshape(B, self.att_head_num, self.att_head_dim, H*W).transpose(-2, -1) # (B, N, D)
            v = v.reshape(B, self.att_head_num, self.att_head_dim, H*W).transpose(-2, -1) # (B, N, D)
            q = self.elu(q) + 1.0
            k = self.elu(k) + 1.0
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) @ v)/(H*W)
            feat_att = (q @ kv * z).transpose(-2, -1).reshape(B, self.gc, H, W)
            feats = torch.cat([feat_att, feat_conv1, feat_conv2], dim=1)
            return (mem, feats)
        else:
            mem, feats = self.grasp_memory(x)
            feat_att, feat_conv1, feat_conv2 = self.asymmetric_decouple(feats)
            feat_att = self.dwconvatt_1(feat_att) + self.dwconvatt_2(feat_att)
            feat_conv1 = self.dwconv1(feat_conv1)
            feat_conv2 = self.dwconv2(feat_conv2)
            feats = torch.cat([feat_att, feat_conv1, feat_conv2], dim=1)
            return (mem, feats)

class InterModule(nn.Module):
    def __init__(
            self, in_features, ratio, out_features=None, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, bias=True, drop=0., mem_dim=None, last=False):
        super().__init__()
        bias = to_2tuple(bias)
        self.last = last
        ratio1, ratio2 = ratio

        dim1 = mem_dim*3
        self.inter1_norm = norm_layer(dim1) if norm_layer else nn.Identity()
        self.inter1_fc1 = nn.Conv2d(dim1, ratio1*dim1, kernel_size=1, bias=bias[0])
        self.inter1_act = act_layer()
        self.inter1_conv = nn.Conv2d(ratio1*dim1, ratio1*dim1, kernel_size=3, padding=1, groups=ratio1*dim1)
        self.inter1_drop = nn.Dropout(drop)
        self.inter1_fc2 = nn.Conv2d(ratio1*dim1, dim1, kernel_size=1, bias=bias[1])

        dim2 = int(in_features)
        self.inter2_norm = norm_layer(dim2) if norm_layer else nn.Identity()
        self.inter2_fc1 = nn.Conv2d(dim2, ratio2*dim2 , kernel_size=1, bias=bias[0])
        self.inter2_act = act_layer()
        self.inter2_conv = nn.Conv2d(ratio2*dim2, ratio2*dim2, kernel_size=3, padding=1, groups=ratio2*dim2)
        self.inter2_drop = nn.Dropout(drop)
        if not self.last:
            self.inter2_fc2 = nn.Conv2d(ratio2*dim2, dim2, kernel_size=1, bias=bias[1])
        else:
            pass

    def forward(self, x):
        mem, feats = x
        res = feats
        feats = self.inter1_norm(feats)
        feats = self.inter1_fc1(feats)
        feats = self.inter1_act(feats)
        feats = self.inter1_conv(feats)
        feats = self.inter1_drop(feats)
        feats = self.inter1_fc2(feats) + res

        cat = torch.cat([mem, feats], dim=1)
        res = cat
        cat = self.inter2_norm(cat)
        cat = self.inter2_fc1(cat)
        cat = self.inter2_act(cat)
        cat = self.inter2_conv(cat)
        cat = self.inter2_drop(cat)
        if not self.last:
            cat = self.inter2_fc2(cat) + res
        else:
            pass
        return cat

class Head(nn.Module):
    def __init__(self, dim, num_classes=1000, ratio=3, act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True, cls_dim=768):
        super().__init__()
        self.act = act_layer()
        self.norm = norm_layer(cls_dim)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(cls_dim, num_classes, bias=bias)
    def forward(self, x):
        x = x.mean((2, 3))
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x

class CAREBlock(nn.Module):
    def __init__(
            self,
            dim,
            mem_dim=None,
            token_mixer=TokenMixer,
            norm_layer=nn.BatchNorm2d,
            inter_layer=InterModule,
            act_layer=nn.GELU,
            ratio=4,            
            ls_init_value=1e-6,
            drop_path=0.,
            att_head_dim=1,
            last=False,
            if_att=False            
    ):
        super().__init__()
        self.last=last
        self.token_mixer = token_mixer(in_channels=dim,
                                       mem_dim=mem_dim,
                                       att_head_dim=att_head_dim,
                                       if_att=if_att,
                                       norm_layer=norm_layer,
                                       act_layer=act_layer)
        self.inter_layer = inter_layer(dim, 
                                       mem_dim=mem_dim, 
                                       ratio=ratio, 
                                       act_layer=act_layer, 
                                       last=last)
        if not self.last: self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.inter_layer(x)
        if not self.last:
            if self.gamma is not None: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
            x = self.drop_path(x) + shortcut
        else:
            x = self.drop_path(x)
        return x


class CAREStage(nn.Module):
    def __init__(
            self,
            in_chs=None,
            out_chs=None,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            ratio=4,
            if_att=False,
            att_head_dim=1,
            mem_dim=None,
            stage=1
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                                norm_layer(in_chs),
                                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),)
        else: self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            if (stage+1)==4 and (i+1)==depth: last = True
            else: last = False
            stage_blocks.append(
                CAREBlock(
                    dim=out_chs,
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    token_mixer=token_mixer,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    ratio=ratio,
                    if_att=if_att,
                    att_head_dim=att_head_dim,
                    mem_dim=mem_dim,
                    last=last))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting(): x = checkpoint_seq(self.blocks, x)
        else: x = self.blocks(x)
        return x

class CARENets(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=None,
            feat_dims=None,
            mem_dims=None,
            token_mixers=TokenMixer,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            ratios=None,
            head_fn=Head,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            att_head_dim=(1, 1, 16, 32),
            cls_dim=None,
            **kwargs,
    ):
        super().__init__()
        num_stage = len(depths)
        dims = tuple(f + m for f, m in zip(feat_dims, mem_dims))
        att_layer = 2
        if not isinstance(token_mixers, (list, tuple)): token_mixers = [token_mixers] * num_stage
        if not isinstance(ratios, (list, tuple)): ratios = [ratios] * num_stage

        self.att_head_dim = att_head_dim
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4), norm_layer(dims[0]))
        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
    
        for i in range(num_stage):
            if i>=att_layer: if_att = True
            else: if_att = False

            out_chs = dims[i]
            stages.append(CAREStage(
                            prev_chs,
                            out_chs,
                            ds_stride=2 if i > 0 else 1,
                            depth=depths[i],
                            drop_path_rates=dp_rates[i],
                            ls_init_value=ls_init_value,
                            act_layer=act_layer,
                            token_mixer=token_mixers[i],
                            norm_layer=norm_layer,
                            ratio=ratios[i],
                            if_att=if_att,
                            att_head_dim=att_head_dim[i],
                            mem_dim=mem_dims[i],
                            stage=i,))
            prev_chs = out_chs

        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        self.head = head_fn(self.num_features, num_classes, ratio=ratios[-1], drop=drop_rate, cls_dim=cls_dim)
        self.dist_head = head_fn(self.num_features, num_classes, ratio=ratios[-1], drop=drop_rate, cls_dim=cls_dim)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages: s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self): return {'norm'}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward_dist_head(self, x):
        x = self.dist_head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        out = self.forward_head(x)
        dist_out = self.forward_dist_head(x)
        if self.training: output = out, dist_out
        else: output = (out+dist_out)/2
        return output

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

@register_model
def CARETrans_S0(pretrained=False, **kwargs):
    model = CARENets(
                token_mixers=TokenMixer,
                depths=(2, 4, 8, 4),
                feat_dims=(24, 48, 96, 192),
                mem_dims=(8, 16, 32, 64),
                cls_dim= 1024,
                ratios=((2, 2), (2, 4), (4, 4), (4, 4)),
                **kwargs)
    return model

@register_model
def CARETrans_S1(pretrained=False, **kwargs):
    model = CARENets(
                token_mixers=TokenMixer,
                depths=(3, 6, 10, 6),
                feat_dims=(24, 48, 96, 192),
                mem_dims=(8, 16, 32, 64),  
                cls_dim=1024,       
                ratios=((2, 4), (2, 4), (4, 4), (4, 4)),
                **kwargs)
    return model

@register_model
def CARETrans_S2(pretrained=False, **kwargs):
    model = CARENets(
                token_mixers=TokenMixer,
                depths=(3, 6, 10, 6),
                feat_dims=(24, 48, 144, 288),
                mem_dims=(8, 16, 48, 96),      
                cls_dim=1536,
                ratios=((4, 4), (4, 4), (4, 4), (4, 4)),
                **kwargs)
    return model