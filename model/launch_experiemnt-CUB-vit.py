import itertools
import subprocess
import math
import random

random.seed(42)

subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 100 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-7 --manualSeed 3131 --avg_pool "
               " --batch_size 16 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 2 --workers 4   --model vit  --gpu 0 --sat_agg_class_outlier_exists --sat_agg_class_cluster_greater"
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --size 224"
               " --closer 0.5"
               " --logltn"
               " --distant 0.3"
               " --train_mode distributed"
               " --sat_agg_same_attribute" 
               " --scale 50.0"                 
               " --k 15"
               " --sat_agg_macroclass_implied"
               " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower"
               " --train_mode distributed --neptune_flag", shell=True
           )