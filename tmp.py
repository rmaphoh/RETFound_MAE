import torch
import os,json
with open('../autodl-tmp/dataset_ROP/annotations.json','r') as f:
    data_dict=json.load(f)
for split_name in ["1","2","3","4"]:
    with open(f'../autodl-tmp/dataset_ROP/split/{split_name}.json','r') as f:
        split_dict = json.load(f)
    for split in split_dict:
        flag=0
        for image_name in split_dict[split]:
            if data_dict[image_name]['stage']>2:
                flag=1
        assert flag==1