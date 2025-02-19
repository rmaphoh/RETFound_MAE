## RETFound - A foundation model for retinal imaging


Official repo including a series of retinal foundation models.<br>
[RETFound: a foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x), which is based on [MAE](https://github.com/facebookresearch/mae).<br>
[New checkpoints](https://huggingface.co/YukunZhou), some of which are based on [DINOV2](https://github.com/facebookresearch/dinov2):

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.

Keras version implemented by Yuka Kihara can be found [here](https://github.com/uw-biomedical-ml/RETFound_MAE)


### 📝Key features

- RETFound is pre-trained on 1.6 million retinal images with self-supervised learning
- RETFound has been validated in multiple disease detection tasks
- RETFound can be efficiently adapted to customised tasks


### 🎉News

- 🐉2025/02: **We organised the model weights on HuggingFace, no more manual downloads needed!**
- 🐉2025/02: **Multiple [pre-trained weights](https://huggingface.co/YukunZhou), including MAE-based and DINOV2-based, are added!**
- 🐉2025/02: **We update the version of packages, such as CUDA12+ and PyTorch 2.3+!**
- 🐉2024/01: [Feature vector notebook](https://github.com/rmaphoh/RETFound_MAE/blob/main/latent_feature.ipynb) are now online!
- 🐉2024/01: [Data split and model checkpoints](BENCHMARK.md) for public datasets are now online!
- 🎄2023/12: [Colab notebook](https://colab.research.google.com/drive/1_X19zdMegmAlqPAEY0Ao659fzzzlx2IZ?usp=sharing) is now online - free GPU & simple operation!
- 2023/10: change the hyperparameter of [input_size](https://github.com/rmaphoh/RETFound_MAE#:~:text=finetune%20./RETFound_cfp_weights.pth%20%5C-,%2D%2Dinput_size%20224,-For%20evaluation%20only) for any image size


### 🔧Install environment

1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound
```

2. Install dependencies

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/rmaphoh/RETFound_MAE/
cd RETFound_MAE
pip install -r requirements.txt
```


### 🌱Fine-tuning with RETFound weights

To fine tune RETFound on your own data, follow these steps:

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_meh">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_shanghai">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_meh</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_meh">access</a></td>
<td align="center">TBD</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_dinov2_shanghai</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai">access</a></td>
<td align="center">TBD</a></td>
</tr>
</tbody></table>

2. Login in your HuggingFace account, where HuggingFace token can be [created and copied](https://huggingface.co/settings/tokens).
```
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN
```

**Optional**: if your machine and server cannot access HuggingFace due to internet wall, run the command below (Do not run it if you can access):
```
export HF_ENDPOINT=https://hf-mirror.com
```

3. Organise your data into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 

4. Start fine-tuning (use IDRiD as example). A fine-tuned checkpoint will be saved during training. Evaluation will be automatically run after training.

The model and finetune can be selected:

| model           | finetune                 |
|-----------------|--------------------------|
| RETFound_mae    | RETFound_mae_natureCFP   |
| RETFound_mae    | RETFound_mae_natureOCT   |
| RETFound_mae    | RETFound_mae_meh         |
| RETFound_mae    | RETFound_mae_shanghai    |
| RETFound_dinov2 | RETFound_dinov2_meh      |
| RETFound_dinov2 | RETFound_dinov2_shanghai |

```
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 16 \
    --world_size 1 \
    --epochs 100 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD \
    --input_size 224 \
    --task RETFound_mae_meh-IDRiD \
    --finetune RETFound_mae_meh
```


4. For evaluation only (download data and model checkpoints [here](BENCHMARK.md); change the path below)


```
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --eval \
    --global_pool \
    --batch_size 16 \
    --world_size 1 \
    --epochs 100 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD \
    --input_size 224 \
    --task RETFound_mae_meh-IDRiD \
    --resume ./RETFound_mae_meh-IDRiD/checkpoint-best.pth
```


### 📃Citation

If you find this repository useful, please consider citing this paper:

```
TBD
```

```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  volume={622},
  number={7981},
  pages={156--163},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```


