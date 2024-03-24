import itertools
import subprocess

import math



import itertools
import subprocess
import random
import math

random.seed(42)
"""

subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 8275 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 --sat_agg_class_outlier_exists --sat_agg_class"
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 8275 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --sat_agg_same_attribute"
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 8275 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --sat_agg_same_attribute"
               " --sat_agg_macroclass_implied"
               " --neptune_flag --p 1"
               ,shell=True)


subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 20 --pretrain_epoch 0 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 8275 --avg_pool "
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
subprocess.run("python model/main.py --dataset CUB --cuda --nepoch 300 --pretrain_epoch 4 "
               " --pretrain_lr 1e-4 --classifier_lr 1e-6 --manualSeed 3131 --avg_pool "
               " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
               " --sat_agg_class_outlier_exists --sat_agg_class"
               " --train_mode distributed"
                " --sat_agg_same_attribute"
               " --sat_agg_macroclass_implied"
               " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower"
               " --neptune_flag --p 1", shell=True)
"""
for i in range(2):
    subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 30 --pretrain_epoch 0 "
                   " --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed "+ random.randint(1,9999).__str__()+" --avg_pool "
                   " --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
                   " --weight_exists --weight_isofclass "
                   " --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 0 "
                   " --sat_agg_class_outlier_exists --sat_agg_class"
                    " --sat_agg_same_attribute"
                   " --sat_agg_macroclass_implied"
                   " --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower"
                   " --neptune_flag --p 1", shell=True)


"""

#--cuda --cropped_image --neptune_flag #print_images --resume --avg_pool --print_images --resume --train_mode distributed



#python model/main.py --dataset AWA2 --cuda --nepoch 200 --pretrain_epoch 0 --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 8275 --avg_pool --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists  --weight_exists --weight_isofclass  --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 1 --sat_agg_class_outlier_exists --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower --sat_agg_same_attribute --sat_agg_macroclass_implied --neptune_flag



#--cuda --cropped_image --neptune_flag #print_images --resume --avg_pool --print_images --resume --train_mode distributed



#python model/main.py --dataset CUB --cuda --nepoch 200 --pretrain_epoch 0 --pretrain_lr 1e-4 --classifier_lr 1e-3 --manualSeed 3131 --avg_pool --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists  --weight_exists --weight_isofclass  --ways 12  --shots 8 --workers 4   --model resnet101  --gpu 1 --sat_agg_class_outlier_exists --sat_agg_class_cluster_greater --sat_agg_class_cluster_lower --sat_agg_same_attribute --sat_agg_macroclass_implied --neptune_flag