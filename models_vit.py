# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from models_mae import Block

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,
        decoder_embed_dim=512, 
        decoder_depth=8,
        decoder_num_heads=16,
        seg_class=1, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.seg_class=seg_class
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        # transformer decoder
        self.decoder_embed = nn.Linear(embed_dim,decoder_embed_dim, bias=True)
        num_patches = self.patch_embed.num_patches

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1,decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, kwargs["mlp_ratio"], qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # decoder pred_head is in orignal size, we will reset it manually in training
        self.decoder_pred = nn.Linear(decoder_embed_dim,kwargs["patch_size"]**2 *seg_class, bias=True) 
    
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            features = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            class_feature = self.fc_norm(features)
        else:
            features = self.norm(x)
            class_feature = features[:, 0]

        return class_feature,x
    def forward_decoder(self,x):
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward(self, x):
        class_feature,encoding = self.forward_features(x)
        class_result = self.head(class_feature)
        seg_result=self.forward_decoder(encoding)
        seg_mask=self.unpatchify(seg_result)
        return class_result,seg_mask
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *seg_class)
        imgs: (N, seg_class, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.seg_class))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.seg_class, h * p, h * p))
        return imgs

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4, 
        qkv_bias=True,
        decoder_embed_dim=512, 
        decoder_depth=8,
        decoder_num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

