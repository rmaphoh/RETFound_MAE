python -m torch.distributed.launch --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --batch_size 16 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
<<<<<<< HEAD
    --nb_classes 2 \
=======
    --nb_classes 4 \
>>>>>>> a4f6bc2855de87c5bf87a32dd7f6a47aa4a24ff7
    --data_path ../autodl-tmp/dataset_ROP \
    --task ./finetune_rop/ \
    --finetune ./RETFound_cfp_weights.pth
