#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=1 python eval.py  --model hdnet  --backbone resnet50 --dataset whu_aerial --resume XXX/hdnet_resnet50_whu_aerial_best_model.pth   --batch-size 1