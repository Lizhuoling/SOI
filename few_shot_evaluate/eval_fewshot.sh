export Data_Root=/home/twilight/twilight/Data/rfs/miniImageNet

python3 eval_fewshot.py \
    --model in_resnet50 \
	--model_path /home/twilight/twilight/Project/SOI/checkpoints/ImageNet_in_res50/checkpoint_0199.pth.tar \
	--data_root $Data_Root \
	--dataset miniImageNet \
    --classifier LR \
	--n_test_runs 600 \
	--n_ways 5 \
	--n_shots 1 \
	--n_queries 15 \
	--n_aug_support_samples 5 \


