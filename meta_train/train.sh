python3 train.py  \
    --data_path /home/twilight/twilight/Data/imagenet_2012/train \
    --model_name in_resnet50 \
    --workers 8 \
    --epochs 200 \
    --lr 0.015 \
    --batch_size 128 \
    --multiprocessing_distributed \
    --moco_dim 128 \
    --moco_k 65536 \
    --moco_m 0.999 \
    --moco_t 0.2 \
    --mlp \
    --aug_plus \
    --cos \
    --checkpoints ../../checkpoints/ImageNet_in_res50 \
    
    
