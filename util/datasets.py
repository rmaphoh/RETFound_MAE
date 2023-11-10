# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import json,torch

def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class ridge_segmentataion_dataset(Dataset):
    def __init__(self, data_path, split, split_name,mode='train'):
        with open(os.path.join(data_path, 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
        self.split_list=split_list[split]
        self.split = split
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=45,interpolation=transforms.InterpolationMode.BILINEAR),
            Fix_RandomRotation(),
        ])
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        self.totenor=transforms.ToTensor()
        self.preprocess=transforms.Compose([
            CropPadding(),
            transforms.Resize((224,224)),
        ])
        self.mode=mode
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_list[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        
        img = self.preprocess(img)
        if self.split == "train":
            img = self.transforms(img)

        # Convert mask and pos_embed to tensor
        img = self.img_transforms(img)

        if 'ridge' in data:
            class_label=1
        else:
            class_label=0
        if self.mode=='visual':
            return img,class_label,data_name
        return img, class_label

    def __len__(self):
        return len(self.split_list)
    
class CropPadding:
    def __init__(self,box=(80, 0, 1570, 1200)):
        self.box=box
    def __call__(self,img) :
        return img.crop(self.box)
    
class Fix_RandomRotation(object):
    
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, F.InterpolationMode.NEAREST , self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class ridge_visual_dataset(Dataset):
    def __init__(self, data_path, split, split_name):
        with open(os.path.join(data_path, 'split', f'{split_name}.json'), 'r') as f:
            split_list=json.load(f)
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
        self.split_list=[]
        for image_name in split_list[split]:
            if 'ridge' in self.data_list[image_name]:
                self.split_list.append(image_name)
        self.split_list=self.split_list[:100]
        self.split = split
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        self.totenor=transforms.ToTensor()
        self.preprocess=transforms.Compose([
            CropPadding(),
            transforms.Resize((224,224)),
        ])
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_list[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        
        img = self.preprocess(img)
        # Convert mask and pos_embed to tensor
        img = self.img_transforms(img)

        if 'ridge' in data:
            class_label=1
        else:
            class_label=0
        return img, class_label,data_name

    def __len__(self):
        return len(self.split_list)
    
class ridge_getembeding_dataset(Dataset):
    def __init__(self, data_path):
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_list=json.load(f)
        self.split_list=self.data_list.keys()
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        self.totenor=transforms.ToTensor()
        self.preprocess=transforms.Compose([
            CropPadding(),
            transforms.Resize((224,224)),
        ])
    def __getitem__(self, idx):
        data_name = self.split_list[idx]
        data = self.data_list[data_name]
        
        img = Image.open(data['image_path']).convert('RGB')
        
        img = self.preprocess(img)
        # Convert mask and pos_embed to tensor
        img = self.img_transforms(img)

        if 'ridge' in data:
            class_label=1
        else:
            class_label=0
        return img, class_label,data_name

    def __len__(self):
        return len(self.split_list)