import argparse
import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from util.datasets import CropPadding

if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--data_path', default='../autodl-tmp/dataset_ROP', type=str)
    parser.add_argument('--threshold', default=3, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--resize', default=224, type=int)
    args = parser.parse_args()

    resize = args.resize
    patch_size = args.patch_size
    threshold = args.threshold
    data_path = args.data_path

    assert resize % patch_size == 0
    patch_number = resize // patch_size

    with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)

    os.makedirs(os.path.join(data_path, 'position_embedding'), exist_ok=True)

    for image_name in data_dict:
        data = data_dict[image_name]
        if 'ridge' in data:
            ridge_mask = Image.open(data['ridge_diffusion_path']).convert('L')
            ridge_mask = CropPadding(ridge_mask)
            ridge_mask = ridge_mask.resize((resize, resize))
            ridge_mask = transforms.ToTensor()(ridge_mask)
            ridge_mask[ridge_mask > 0] = 1

            # Patchify using a convolution operation
            conv_kernel = torch.ones(1, 1, patch_size, patch_size)
            position_heatmap = F.conv2d(ridge_mask.unsqueeze(0), conv_kernel, stride=patch_size)
            position_heatmap = (position_heatmap >= threshold).float().squeeze(0)

        else:
            # Zero torch tensor in size (patch_number, patch_number)
            position_heatmap = torch.zeros(1, patch_number, patch_number)

        save_path = os.path.join(data_path, 'position_embedding', image_name[:-4] + '.png')
        transforms.ToPILImage()(position_heatmap.squeeze(0)).save(save_path)
        data_dict[image_name]['pos_embed_path'] = save_path

    # Save the updated data_dict
    with open(os.path.join(data_path, 'annotations_updated.json'), 'w') as f:
        json.dump(data_dict, f, indent=4)
