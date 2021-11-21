python3 train_supervised.py \
    --model in_resnet12 \
    --model_remark IN_resnet12_SOI \
    --dataset miniImageNet \
	--trial pretrain \
	--model_path ./checkpoints/miniImageNet \
	--tb_path ./tensorboards \
	--data_root /opt/data/private/zli/data/rfs \
	--n_ways 5 \
	--n_shots 1 \
	--n_queries 15 \
	--n_aug_support_samples 5 \
    --learning_rate 0.05 \
    --pretrained_model_path ./checkpoints/miniImageNet/IN_resnet12_backbone/checkpoint_0049.pth.tar \

