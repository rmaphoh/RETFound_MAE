## RETFound - A foundation model for retinal image


This is official repo for RETFound, which heavily bases on [MAE](https://github.com/facebookresearch/mae):


### Key features

- RETFound was trained on 1.6 million retinal images
- RETFound has been validated in multiple disease detection tasks
- RETFound can be efficiently adapted to customised task


### Install enviroment

Create enviroment with conda:

```
conda create -n retfound python=3.6.15 -y
```

Install Pytorch 1.81 (cuda 11.1)
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install others
```
pip install -r requirement.txt
```


### Fine-tuning with RETFound weights

- RETFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">Colour fundus image</td>
<td align="center"><a href="https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

- Organise data (use IDRiD as [example](Example.ipynb))

<p align="left">
  <img src="./pic/file_index.jpg" width="160">
</p>


- Start fine-tuning (use IDRiD as example). A fine-tuned checkpoint will be saved during training. Evaluation will be run after training.


```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py  
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD_data/ \
    --task ./finetune_IDRiD/ \
    --finetune ./RETFound_cfp_weights.pth

```


- For evaluation only


```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py 
    --eval --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD_data/ \
    --task ./internal_IDRiD/ \
    --resume ./finetune_IDRiD/checkpoint-best.pth

```


