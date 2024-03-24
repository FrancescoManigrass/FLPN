import itertools
import subprocess




subprocess.run("python model/main.py --dataset AWA2 --cuda --nepoch 100 --pretrain_epoch 4 "
               "--pretrain_lr 1e-4 --classifier_lr 1e-7 --manualSeed 8275 --avg_pool "
               "--batch_size 16 --calibrated_stacking 0.7 --all --gzsl  --axioms_exists "
               " --weight_exists --weight_isofclass "
               " --ways 12  --shots 2 --workers 4   --model vit  --gpu 0 --sat_agg_class_outlier_exists --sat_agg_class_cluster_greater"
               "--sat_agg_class_outlier_exists --sat_agg_class"
                "--sat_agg_same_attribute"
               "--sat_agg_macroclass_implied"
               "--sat_agg_class_cluster_greater --sat_agg_class_cluster_lower --sat_agg_class_global"
               "--train_mode distributed --neptune_flag"
           )
#--cuda --cropped_image --neptune_flag #print_images --resume --avg_pool print_images --resume --train_mode distributed



