# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from PIL import Image, ImageOps
import timm.models.vision_transformer
import numpy as np

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

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
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    def _get_attention_map(self, img_tensor):
        """
        img_tensor: torch.Tensor of shape (batch_size, 3, height, width)
        Returns attention heatmaps for each image in the batch.
        """
        # Ensure the model is in evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass through the model up to the last block
            B = img_tensor.shape[0]
            x = self.patch_embed(img_tensor)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for blk in self.blocks[:-1]:  # excluding the last block
                x = blk(x)

            # Get the attention weights from the last block
            last_block = self.blocks[-1]
            x, attn_weights = last_block(x, return_attention=True)
            
            # We are only interested in the attention from the class token
            attn_heatmap = attn_weights[:, 0, 1:].reshape(B, 1, 14, 14)  # Assuming an image size of 224x224

        # Upsample the attention heatmap to the size of the original image
        attn_heatmap = torch.nn.functional.interpolate(attn_heatmap, scale_factor=16, mode='bilinear', align_corners=False)
        
        return attn_heatmap.squeeze(1)  # Removing the single-channel dimension


def visual_heatmap(image_path, attention_heatmap, save_path):
    """
    Visualizes the heatmap and overlays it on the original image using only PIL.
    image_path: Path to the original image.
    attention_heatmap: A 2D numpy array representing the attention weights.
    save_path: Path to save the visualized heatmap.
    """
    # Open the image
    image = Image.open(image_path).convert("RGB")
    
    # Normalize the attention heatmap to be between 0 and 255
    heatmap = (attention_heatmap - attention_heatmap.min()) / (attention_heatmap.max() - attention_heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Create a color map using a LUT (look-up table)
    heatmap_color = ImageOps.colorize(Image.fromarray(heatmap, mode='L'), 
                                      black="black", white="red")
    
    # Resize the colorized heatmap to match the size of the original image
    heatmap_color = heatmap_color.resize(image.size, Image.LANCZOS)
    
    # Superimpose the heatmap onto the original image
    superimposed_img = Image.blend(image, heatmap_color, alpha=0.5)
    
    # Save and show the image
    superimposed_img.save(save_path)
    superimposed_img.show()  # This opens the default viewer and displays the image
    
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
