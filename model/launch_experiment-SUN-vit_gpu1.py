import itertools
import subprocess
import math
import random

random.seed(42)

subprocess.run("CUDA_VISIBLE_DEVICES=1 accelerate launch  model/main.py --dataset SUN --cuda "
               " --nepoch 100 --pretrain_epoch 0 "
               " --pretrain_lr 1e-3 --classifier_lr 1e-3 --manualSeed 2347 --avg_pool "
               " --batch_size 32 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --optimizer sgd"
               " --weight_exists --weight_isofclass "
               " --ways 32  --shots 1 --workers 8   --model vit  --gpu 1 "

               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --size 224"
               " --train_mode distributed"
               " --sat_agg_same_attribute"
               " --scale 50.0"
               " --k 15"
               " --accelerator"
               " --neptune_flag"
               " --distant 0.3"  # 0.3
               " --closer 0.4"  # 0.5
               " --sat_agg_macroclass_implied"
               " --neptune_flag"
               " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower",
                shell=True
               )