import itertools
import subprocess

import math
import random
random.seed(42)


subprocess.run("accelerate launch model/main.py --dataset CUB --cuda --nepoch 300 --pretrain_epoch 0 "
               " --pretrain_lr 1e-2 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool "
               " --batch_size 32 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 12 --workers 4   --model resnet101  --gpu 0"
               " --sat_agg_class --sat_agg_class_outlier_exists"
               " --sat_agg_same_attribute"
               " --sat_agg_macroclass_implied"
               " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower"  
               " --scale 50.0"     
               " --k 15"
               " --optimizer sgd"
              " --neptune_flag"
               " --p 1", shell=True)




"""
subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 --sat_agg_class_outlier_exists --sat_agg_class"
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --sat_agg_same_attribute"
               
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --sat_agg_same_attribute"
               " --sat_agg_macroclass_implied"
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --sat_agg_same_attribute"
               " --sat_agg_macroclass_implied"
               " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower"
               " --neptune_flag --p 1"
               ,shell=True)


"""


#--cuda --cropped_image --neptune_flag #print_images --resume --avg_pool --print_images --resume --train_mode distributed



#python model/main.py --dataset CUB --cuda --nepoch 200 --pretrain_epoch 0 --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists  --weight_exists --weight_isofclass  --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 1 --sat_agg_class_outlier_exists --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower --sat_agg_same_attribute --sat_agg_macroclass_implied --neptune_flag