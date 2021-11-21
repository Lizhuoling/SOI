# SOI

Released code for the paper: Efficient Few-shot Classification via Contrastive Pre-training on Web Data

## Environment

```
pip install -r requirement.txt
```

## Crawl web data

```
cd crawl_data
python3 main.py \
    --target_path $Path_to_Save_Data \
    --num_per_class 3000 \
```

## Contrastive pre-training

```
cd meta_train
python3 train.py  \
    --data_path $Path_to_Save_Data \
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
