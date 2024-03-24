
The current project page provides pytorch code that implements the following paper:
# Fuzzy Logic Visual Network (FLVN): A neuro-symbolic approach for visual features matching
## Authors: Francesco Manigrasso, Lia Morra, Fabrizio Lamberti
## Abstract:
From the beginning of zero-shot learning research, visual attributes have been shown to play an important role. In order to better transfer attribute-based knowledge from known to unknown classes, we argue that an image representation with integrated attribute localization ability would be beneficial for zero-shot learning. To this end, we propose a novel zero-shot representation learning framework that jointly learns discriminative global and local features using only class-level attributes. While a visual-semantic embedding layer learns global features, local features are learned through an attribute prototype network that simultaneously regresses and decorrelates attributes from intermediate features. We show that our locality augmented image representations achieve a new state-of-the-art on three zero-shot learning benchmarks. As an additional benefit, our model points to the visual evidence of the attributes in an image, e.g. for the CUB dataset, confirming the improved attribute localization ability of our image representation.

- Logic Tensor Network is a neural symbolic framework developed  from the paper Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur,
- The torch implementation of the framework is cloned from https://github.com/tommasocarraro/LTNtorch.git
- Datasets are cloned from [CUB](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html),[AWA2](https://cvml.ista.ac.at/AwA2/),[SUN](https://groups.csail.mit.edu/vision/SUN/hierarchy.html),
- All the material in the repository is the implementation of the paper accepted for publication *Fuzzy Logic Visual Network (FLVN): A  neuro-symbolic approach for visual features matching* by  Manigrasso Francesco , Morra Lia, Lamberti Fabrizio.
- Download the repository with, for example, `git clone https://gitlab.com/grains2/flvn.git`.


## Data
- `AWA2` it contains the dataset from *Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly* by Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata
- `CUB` it contains the dataset from *Caltech-UCSD Birds-200-2011 (CUB-200-2011)* by Catherine Wah1 , Steve Branson1 , Peter Welinder2 , Pietro Perona2 , Serge Belongie
- `SUN` it contains the dataset from *SUN Database: Large-scale Scene Recognition from Abbey to Zoo* by Jianxiong Xiao; James Hays; Krista A. Ehinger; Aude Oliva; Antonio Torralba

## Training and Evaluation  
- All weights of pretrained models are taken from repository https://github.com/wenjiaXu/APN-ZSL

- AWA2
```sh
$ python main.py python ./model/main.py --dataset AWA2 --cuda --nepoch 30 --pretrain_epoch 2 --pretrain_lr 1e-4 --classifier_lr 1e-7 --manualSeed 7048 \
--xe 1 --attri 1e-4 --regular 0.0005 --l_xe 1 --l_attri 1e-2 --l_regular 0.5e-6 --cpt 2e-9 --avg_pool --batch_size 64 --calibrated_stacking 0.7 --all --gzsl  --c 0.01
```
- CUB
```sh
$ python main.py python ./model/main.py --dataset CUB --calibrated_stacking 0.7 --cuda --nepoch 30 --batch_size 64 --train_id 0 --manualSeed 3131 --pretrain_epoch 5  \
--pretrain_lr 1e-4 --classifier_lr 1e-6 --xe 1 --attri 1e-2 --regular 5e-5 --l_xe 1 --l_attri 1e-1  --l_regular 4e-2 --cpt 1e-9 --use_group --gzsl --c 1```
```

- SUN
```sh
$ python main.py python ./model/main.py --calibrated_stacking 0.4 --dataset SUN --cuda --nepoch 30 --pretrain_epoch 4 --pretrain_lr 1e-3  \ 
--classifier_lr 1e-6 --manualSeed 2347 --xe 1 --attri 1e-4 --regular 1e-3 --l_xe 1 --l_attri 5e-2 --l_regular 5e-3  --avg_pool --use_group --cpt 2e-7 --gzsl   --c 1
```

# Citation

If you make use of the dataset in your research, please cite our paper:

Manigrasso Francesco ,Morra Lia, Lamberti Fabrizio,
"Fuzzy Logic Visual Network (FLVN): A neuro-symbolic approach for visual features matching", 22nd International Conference on IMAGE ANALYSIS AND PROCESSING.






# Contributors & Maintainers
Francesco Manigrasso,Lia Morra,  and Fabrizio Lamberti
