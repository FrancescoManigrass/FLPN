import itertools
import subprocess
import math
import random

random.seed(42)

subprocess.run("CUDA_VISIBLE_DEVICES=0 accelerate launch  model/main.py --dataset SUN --cuda "
               " --nepoch 100 --pretrain_epoch 0"
               " --pretrain_lr 1e-3 --classifier_lr 1e-3"
               " --manualSeed 214 --avg_pool"
               " --batch_size 32 --calibrated_stacking 0.9 --all --gzsl  --axioms_exists"
               " --optimizer sgd"
               " --weight_exists --weight_isofclass"
               " --ways 2  --shots 16 --workers 8   --model vit  --gpu 0"
               " --sat_agg_class"
               " --sat_agg_class_outlier_exists"
               " --sat_agg_same_attribute"
             
               " --logltn"
             
               " --size 224"
               " --optimizer sgd"
               " --train_mode distributed"
               " --scale 25.0"
               " --k 15"
               " --accelerator"
               " --neptune_flag"
               " --distant 0.7"  # 0.3
               " --closer 0.8"  # 0.5
               " --neptune_flag",
               shell=True
               )