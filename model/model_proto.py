import os
import random

import torch.nn as nn
import torch
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import ltn
from losses import SupConLoss
from utils_zsl import create_fake
from nets import ProtoModel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_SOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_SOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class LINEAR_SOFTMAX(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LINEAR_SOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Parameter(2e-4 * torch.rand([9, 9]), requires_grad=True)  # 全局分支参数
        self.fc2 = nn.Parameter(2e-4 * torch.rand([9, 9]), requires_grad=True)  # 全局分支参数
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tanh(x.mm(self.fc1))
        x = torch.sigmoid(x.mm(self.fc2))
        return x


class LAYER_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LAYER_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        batch_size = x.size(0)
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, -1)
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


"""
class isofclass(nn.Module):
    def __init__(self,opt,attribute):
        super(isofclass,self).__init__()
        self.attribute = attribute
        self.opt = opt
        if opt.cuda == True :
            self.attribute=self.attribute.to(torch.device("cuda:"+opt.gpu))

        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.dim_dict = {'layer1': 56 * 56, 'layer2': 28 * 28, 'layer3': 14 * 14, 'layer4': 7 * 7, 'avg_pool': 1 * 1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.batch_size = opt.batch_size
        self.softmax = nn.Softmax(dim=1)

        if opt.dataset == 'CUB':
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'AWA1':
            exit(1)
            self.ALE = LINEAR_SOFTMAX_ALE(input_dim=self.channel_dict['avg_pool'], attri_dim=85)
        elif opt.dataset == 'AWA2':
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
        elif opt.dataset == 'SUN':
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 2048, 1, 1]), requires_grad=True)  # 全局分支参数



        self.avg_pool = opt.avg_pool

    def forward(self,x,label,attribute,get_prediction=False,size=0):

        batch_size = x.size(0)
        attention = dict()
        pre_attri = dict()
        pre_class = dict()
        ######### 全局分支
        if self.avg_pool:
            pre_attri['final'] = F.avg_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
        else:
            pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=7).view(batch_size, -1)
            # 全局分支 [64, 312]



        if get_prediction==False:
            #output_final = self.softmax(pre_attri['final'].mm(attribute))[label.value,1]  #[batchsize,类别数]
            output_final = torch.gather(self.softmax(pre_attri['final'].mm(attribute)),1,label)
        else:
            output_final = self.softmax(pre_attri['final'].view(-1,size,85)[:,0,:].mm(attribute.view(-1,size,85)[0,:,:].T))
        return output_final


class hasAttribute(nn.Module):
    def __init__(self,opt,attribute):
        super(hasAttribute,self).__init__()
        self.attribute = attribute
        self.opt = opt
        if opt.cuda == True :
            self.attribute=self.attribute.to(torch.device("cuda:"+opt.gpu))

        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.dim_dict = {'layer1': 56 * 56, 'layer2': 28 * 28, 'layer3': 14 * 14, 'layer4': 7 * 7, 'avg_pool': 1 * 1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}
        self.batch_size = opt.batch_size
        self.softmax = nn.Softmax(dim=1)

        if opt.dataset == 'CUB':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [312, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)

        elif opt.dataset == 'AWA1':
            exit(1)

        elif opt.dataset == 'AWA2':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [85, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)

        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [102, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape),
                                                            requires_grad=True)  # 原型参数
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)



        self.avg_pool = opt.avg_pool

    def forward(self,x,y=None,label=None,attribute=None,same_attribute=False):

        batch_size = x.size(0)
        attention = dict()
        pre_attri = dict()
        pre_class = dict()

        attention2 = dict()
        pre_attri2 = dict()
        pre_class2 = dict()
        ######### 全局分支

        for name in self.extract:
            #### 求内积，得到 M

            attention[name] = F.conv2d(input=x,
                                       weight=self.prototype_vectors[name])  # [64, 312, 7, 7]
            # attention[name] = torch.einsum('bfmn,vf->bvmn',record_features[name],self.prototype_vectors[name])

            #### 最大化，得到 a
            pre_attri[name] = F.max_pool2d(attention[name], kernel_size=self.kernel_size[name]).view(batch_size,
                                                                                                     -1)  # [64, 312]
            #### 得到分类结果
            if attribute!=None:
                pre_class[name] = self.softmax(pre_attri[name].mm(attribute))  # [64, 150]

        if same_attribute:
            for name in self.extract:
                #### 求内积，得到 M

                attention2[name] = F.conv2d(input=y.value,
                                           weight=self.prototype_vectors[name])  # [64, 312, 7, 7]
                # attention[name] = torch.einsum('bfmn,vf->bvmn',record_features[name],self.prototype_vectors[name])

                #### 最大化，得到 a
                pre_attri2[name] = F.max_pool2d(attention2[name], kernel_size=self.kernel_size[name]).view(batch_size,
                                                                                                         -1)  # [64, 312]
                #### 得到分类结果
                if attribute != None:
                    pre_class2[name] = self.softmax(pre_attri2[name].mm(attribute))  # [64, 150]



            #output_final = self.softmax(pre_attri['final'].mm(attribute))[label,1]  #[batchsize,类别数]
            return torch.exp(-(1-F.cosine_similarity(pre_attri[name],attribute.T[torch.squeeze(label.value)])))
        else:
            x = torch.gather(pre_class[name], 1, y )
            return  x.view(x.size(0), -1)


"""


def cosine_dis(pred_att, support_att, scale=1.0):
    pred_att_norm = torch.norm(pred_att, p=2, dim=1).unsqueeze(1)
    pred_att_normalized = pred_att.div(pred_att_norm + 1e-10)
    support_att_norm = torch.norm(support_att, p=2, dim=1).unsqueeze(1)
    support_att_normalized = support_att.div(support_att_norm + 1e-10)
    cos_dist = torch.einsum('bd,nd->bn', pred_att_normalized, support_att_normalized)
    score = cos_dist * scale  # B, cls_num
    return score, cos_dist


def same_attribute(x, y):
    return torch.gather(torch.gt(y, 0), 1, x)


def not_same_attribute(x, y):
    return torch.gather(torch.le(y, 0), 1, x)


def cosine_similarity(x, y):
    """
    dot_product = torch.sum(torch.multiply(x, y), dim=1)
    x = torch.sqrt(torch.sum(torch.square(x), dim=1))
    y = torch.sqrt(torch.sum(torch.square(y), dim=1))
    similarity = dot_product / (x * y)
    return 1 / (1 + torch.exp(-1 * similarity))
    """
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)
    # dist = 1. - x @ y.T
    # [i  for i in range(len(torch.nn.functional.cosine_similarity(x, y))) if torch.nn.functional.cosine_similarity(x, y)[i] > 1 ]

    dist = (1 - torch.clamp(torch.nn.functional.cosine_similarity(x, y), -1, 1))

    return torch.exp(-dist)


"""
cosine tra -1 e 1
1-cosine tra 2 e 0

"""

import torch


def euclidean_distance(prediction, support_att):
    N, S = prediction.shape
    C, S = support_att.shape

    # Calcola la norma L2 (Euclidea) di prediction e support_att
    prediction_norm = torch.norm(prediction, p=2, dim=1, keepdim=True)
    support_att_norm = torch.norm(support_att, p=2, dim=1, keepdim=True)

    # Normalizza prediction e support_att
    prediction_normalized = prediction / (prediction_norm + 1e-10)
    support_att_normalized = support_att / (support_att_norm + 1e-10)

    # Calcola la distanza euclidea tra prediction e support_att
    offset = torch.sum(torch.abs(prediction_normalized - support_att_normalized), dim=1)

    # Calcola la probabilità utilizzando la funzione logistica
    probability = 1 / (1 + torch.exp(-offset))

    return probability


def similarity2(x, y):
    scores = torch.softmax(x + y, dim=1)
    labels = torch.argmax(scores, dim=1)

    return torch.gather(scores, 1, labels.view(-1, 1)).view(-1)


# self.distance_pixel(images_view_features_x, pixel_position,picco_position),

def distance(x, y, z):
    # first = y[:, :, :, 0].view(-1, 1) - z[:,0].view(-1,1).repeat_interleave(49)
    # torch.sum(torch.pow(torch.abs(y-z.view(-1,1,1,2).repeat(1,7,7,1)),2),dim=3)*x
    d = torch.sum(torch.sum(torch.sqrt(torch.pow(torch.abs(y - z.view(-1, 1, 1, 2).repeat(1, 7, 7, 1)), 2)), dim=3) * x,
                  dim=(1, 2))

    # print("max",torch.max(torch.exp(-d)))
    # print("min", torch.min(torch.exp(-d)))
    # print(torch.exp(-d))
    return torch.exp(-1e-3 * d)

    pass


def similarity3(x, y):
    # x = F.normalize(x, p=2, dim=-1)
    # y = F.normalize(y, p=2, dim=-1)

    dot_product = torch.sum(torch.multiply(x, y), dim=1)
    x = torch.sqrt(torch.sum(torch.square(x), dim=1))
    y = torch.sqrt(torch.sum(torch.square(y), dim=1))
    similarity = dot_product / (x * y)
    return 1 / (1 + torch.exp(-1 * similarity))


def similarity4(x, y):
    # x = F.normalize(x, p=2, dim=-1)
    # y = F.normalize(y, p=2, dim=-1)

    # dot_product = torch.sum(torch.multiply(x, y), dim=1)
    # x = torch.sqrt(torch.sum(torch.square(x), dim=1))
    # y = torch.sqrt(torch.sum(torch.square(y), dim=1))
    # similarity = dot_product / (x * y)

    x_norm = x.norm(p=2, dim=1, keepdim=True)
    x_ = x / (x_norm + 1e-10)

    y_norm = y.norm(p=2, dim=1, keepdim=True)
    y_ = y / (y_norm + 1e-10)

    x_ = torch.where(x_ < torch.mean(x_, dim=1).view(-1, 1), x_, 0)
    y_ = torch.where(y_ < torch.mean(y_, dim=1).view(-1, 1), y_, 0)

    def create_fake(x_):
        false_values_for_row = 15
        false_indices = torch.randint(low=0, high=x_.shape[1],
                                      size=(x_.shape[0], false_values_for_row))
        mask = torch.zeros(x_.shape[0], x_.shape[1]).bool()

        mask[torch.arange(x_.shape[0]).unsqueeze(1).type(torch.LongTensor), false_indices] = True
        mask = mask.to(torch.device("cuda:"+opt.gpu))
        x_ = torch.where(mask, 0, x_)
        return x_

    x_ = create_fake(x_)
    y_ = create_fake(y_)

    return torch.exp(-10 * torch.mean(((x_ - y_) ** 2), dim=1)).reshape(-1)

    # return torch.exp(-1000000 * torch.square(torch.norm(x - y, dim=1)))


def same_attribute_error(x, y):
    return torch.gather(torch.gt(y, 0), 1, x)


def not_same_attribute_error(x, y):
    return torch.gather(torch.le(y, 0), 1, x)


def get_and_of_attributes(x, y):
    # x[torch.logical_not(torch.gt(y, 0))] = 1
    # x[torch.gt(y, 0))] = 1
    x = torch.where(torch.gt(y, 0), x, 1 - x)
    return torch.min(x, dim=1)[0]


def euclid_distance(x, y):
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    y = torch.nn.functional.normalize(y, p=2, dim=1)
    # dist = 1. - x@y.T
    return torch.exp(-torch.norm(x - y, dim=1))


def get_and_of_attributes_error(x, y):
    # x[torch.logical_not(torch.gt(y, 0))] = 1
    # x[torch.gt(y, 0))] = 1
    mask = torch.gt(y, 0)
    false_indices = torch.randint(low=0, high=x.shape[1], size=(x.shape[0], 15))
    mask[torch.arange(y.shape[0]).unsqueeze(1).type(torch.LongTensor), false_indices] = False
    x = torch.where(torch.gt(y, 0), x, 1 - x)
    return torch.min(x, dim=1)[0]


class resnet_proto_IoU(nn.Module):
    def __init__(self, opt, attribute, binary_att, data, original_att):
        super(resnet_proto_IoU, self).__init__()
        resnet = models.resnet101(pretrained=True)
        num_ftrs = resnet.fc.in_features
        num_fc_dic = {'cub': 150, 'awa2': 40, 'sun': 645}
        self.extract = ['layer4']  # 'layer1', 'layer2', 'layer3', 'layer4'
        self.dim_dict = {'layer1': 56 * 56, 'layer2': 28 * 28, 'layer3': 14 * 14, 'layer4': 7 * 7, 'avg_pool': 1 * 1}
        self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048, 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 7, 'avg_pool': 1}

        self.binary_att = binary_att
        self.parts = data.parts
        self.parts_key = data.parts_key
        self.data = data

        if 'c' in opt.resnet_path.split('/')[-1]:
            num_fc = num_fc_dic['cub']
        elif 'awa2' in opt.resnet_path.split('/')[-1]:
            num_fc = num_fc_dic['awa2']
        elif 'sun' in opt.resnet_path.split('/')[-1]:
            num_fc = num_fc_dic['sun']
        else:
            num_fc = 1000

        resnet.fc = nn.Linear(num_ftrs, num_fc)

        # 01 - load resnet to model1
        if opt.resnet_path != None:
            state_dict = torch.load(opt.resnet_path)
            resnet.load_state_dict(state_dict)
            print("resnet load state dict from {}".format(opt.resnet_path))

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fine_tune(True)

        # 02 - load cls weights
        # we left the entry for several layers, but here we only use layer4

        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.attr_proto_size = 2048
        if opt.dataset == 'CUB':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [312, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([312, 2048, 1, 1]), requires_grad=True)
            self.part_num=312
            hid_size=1024
        elif opt.dataset == 'AWA1':
            exit(1)
            self.ALE = LINEAR_SOFTMAX_ALE(input_dim=self.channel_dict['avg_pool'], attri_dim=85)
        elif opt.dataset == 'AWA2':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [85, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape), requires_grad=True)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
            self.ALE_vector_cropped = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
            self.MLP = MLP(9, 9)

            self.part_num = 85
            hid_size = 1024

            self.ALE_PARTS = nn.Parameter(2e-4 * torch.rand([85, 9]), requires_grad=True)
            # self.attribute_class_vector = nn.Parameter(2e-4 * torch.rand([85, 50]), requires_grad=False)
            """"
            self.hasAttributeofclass = ltn.Predicate(
                model=isOfClassLTN(name="hasAttributeofclass", num_features=85, num_layers=40))
            """
            # self.isOfClassLTN=isOfClassLTN(name="hasAttributeofclass", num_features=85, num_layers=40)
            # self.hasAttributeofclassLambda = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
            self.hasAttributeofclassLambda = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
            self.macroclass_vector = nn.Parameter(2e-4 * torch.rand([85, 9]), requires_grad=True)
            # self.part_of_vector = nn.Parameter(2e-2 * torch.rand([85, data.parts.shape[0]]), requires_grad=True)

        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [102, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape),
                                                            requires_grad=True)  # 原型参数
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 1024, 1, 1]), requires_grad=True)  # 全局分支参数
            # self.ALE_vector_cropped = nn.Parameter(2e-4 * torch.rand([102, 2048, 1, 1]), requires_grad=True)  # 全局分支参数

        self.avg_pool = opt.avg_pool
        self.avg_pool_part = opt.avg_pool_part

        # self.isofclass = ltn.Predicate(model=isofclass(opt,attribute))
        # self.hasAttribute = ltn.Predicate(model=hasAttribute(opt, attribute))

        self.isofclass = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
        self.distance_pixel = ltn.Function(func=lambda x, y, z: distance(x, y, z))
        self.isofpart = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))

        self.hasAttribute = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
        self.hasAttributeEnd = ltn.Function(func=lambda x, y: get_and_of_attributes(x, y))
        self.same_class = ltn.Function(func=lambda x, y: cosine_similarity(x, y))
        self.same_part = ltn.Function(func=lambda x, y: similarity3(x, y))
        self.euclidean_distance = ltn.Function(func=lambda x, y: euclid_distance(x, y))
        self.hasAttributeEndError = ltn.Function(func=lambda x, y: get_and_of_attributes_error(x,
                                                                                               y))  # ltn.Predicate(model = Has_attribute())#[ltn.Predicate(model=Predicate_ltn(num_features=85,name=f.__str__()))  for f in range(85)]

        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        if opt.focalloss:
            self.Forall = ltn.Quantifier(ltn.fuzzy_ops.Focalloss(p=2, stable=True), quantifier="f")
            self.ForallSum = ltn.Quantifier(ltn.fuzzy_ops.AggregSum(), quantifier="f")
            self.SatAgg = ltn.fuzzy_ops.SatAgg(ltn.fuzzy_ops.AggregSum())
        else:
            self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="f")
            self.ForallSum = self.Forall
            self.SatAgg = ltn.fuzzy_ops.SatAgg()


        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesKleeneDienes())
        self.dist = lambda x, y: torch.unsqueeze(torch.norm(x - y, dim=1), dim=1)
        self.same_attribute = lambda x, y: torch.unsqueeze(same_attribute(x, y), dim=1)
        self.not_same_attribute = lambda x, y: torch.unsqueeze(not_same_attribute(x, y), dim=1)
        self.w2v_att = torch.tensor(np.float32(data.w2v))
        _, self.w2v_length = self.w2v_att.shape
        self.med_dim = 300  # 1024
        self.feat_channel = 2048
        self.feat_w = 7
        self.feat_h = 7
        self.QueryW = nn.Sequential(nn.Linear(self.w2v_length, self.med_dim))  # L,M = 300,1024
        self.QueryW.weight = nn.Parameter(2e-4 * torch.rand([self.med_dim, self.med_dim]), requires_grad=True)
        self.KeyW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim))  # C,M = 2048,1024
        self.KeyW.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.med_dim]), requires_grad=True)
        self.ValueW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim))  # C,M = 2048,1024
        self.ValueW.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.med_dim]), requires_grad=True)
        self.W_o = nn.Sequential(nn.Linear(self.med_dim, self.feat_channel))  # M,C = 1024,2048
        self.W_o.weight = nn.Parameter(2e-4 * torch.rand([self.med_dim, self.feat_channel]), requires_grad=True)
        self.W = nn.Parameter(2e-4 * torch.rand([self.w2v_att.shape[1], self.feat_channel]),
                              requires_grad=True)  # 300 * 2048

        self.finale = nn.Sequential(nn.Linear(self.feat_channel, self.feat_channel))  # M,C = 1024,2048

        self.attritube_num = self.w2v_att.shape[0]
        # attributes = torch.where(self.data.original_att < 0.0, 0.0, self.data.original_att)
        # attributes = attributes / torch.max(attributes)
        # self.attributes = attributes.to(torch.cuda.current_device())

        self.V = nn.Sequential(nn.Linear(self.feat_channel, self.attritube_num))  # V, S = [2048,4096]
        self.V.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.attritube_num]), requires_grad=True)

        self.hid_dim = 2048
        self.V_att_hidden_branch = nn.Sequential(nn.Linear(self.feat_channel, self.hid_dim))  # H, C
        self.V_att_hidden_branch.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.hid_dim]),
                                                       requires_grad=True)
        self.ratio = 1.0

        self.fc_attention_channel = nn.Linear(self.attr_proto_size, self.part_num)
        nn.init.xavier_uniform_(self.fc_attention_channel.weight)

        self.proto_model = ProtoModel(self.part_num, hid_size, self.attr_proto_size, with_cn=True)
        self.fc_proto = nn.Identity()
        self.attribute_vector = nn.Parameter(torch.eye(self.part_num), requires_grad=False)
        self.macroclass_vector = nn.Parameter(torch.rand(self.data.attribute_macroclass.shape[0],self.part_num),requires_grad=True)
        out_channel = self.part_num
        self.extract_1 = torch.nn.Conv2d(256, out_channel, kernel_size=8, stride=8)
        self.extract_2 = torch.nn.Conv2d(512, out_channel, kernel_size=4, stride=4)
        self.extract_3 = torch.nn.Conv2d(1024, out_channel, kernel_size=2, stride=2)
        self.extract_4 = torch.nn.Conv2d(2048, out_channel, kernel_size=1, stride=1)

        nn.init.xavier_uniform_(self.extract_1.weight)
        nn.init.constant_(self.extract_1.bias, 0)
        nn.init.xavier_uniform_(self.extract_2.weight)
        nn.init.constant_(self.extract_2.bias, 0)
        nn.init.xavier_uniform_(self.extract_3.weight)
        nn.init.constant_(self.extract_3.bias, 0)
        nn.init.xavier_uniform_(self.extract_4.weight)
        nn.init.constant_(self.extract_4.bias, 0)

        self.cosine_d = lambda x, y: torch.unsqueeze(
            torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(x, p=2, dim=1),
                                                  torch.nn.functional.normalize(y, p=2, dim=1)), dim=1)

        self.scale = nn.Parameter(torch.ones(1) * 50.0, requires_grad=False)

    def forward(self, x, attribute, label=False, label_m=None, attribute_macroclass_seen=None, return_map=False,
                get_prediction=False, axioms_options=None, opt=None, extract_bb=None, original_attribute=None,
                paths=None):
        """out: predict class, predict attributes, maps, out_feature"""
        # print('x.shape', x.shape)
        # record_features = {}
        # batch_size = x.size(0)
        # init_image = x.clone()
        x1 = self.resnet[0:5](x)  # layer 1
        # record_features['layer1'] = x  # [64, 256, 56, 56]
        x2 = self.resnet[5](x1)  # layer 2
        # record_features['layer2'] = x  # [64, 512, 28, 28]
        x3 = self.resnet[6](x2)  # layer 3
        # record_features['layer3'] = x  # [64, 1024, 14, 14]
        x4 = self.resnet[7](x3)  # layer 4
        # record_features['layer4'] = x  # [64, 2048, 7, 7]

        attention = dict()
        pre_attri = dict()
        pre_class = dict()
        pre_macro = dict()
        N, C, W, H = x4.shape

        # seen_att_normalized = F.normalize(seen_att,dim=-1)
        parts_map = self.extract_1(x1) + self.extract_2(x2) + self.extract_3(x3) + self.extract_4(x4)  # 8,85,14,14
        # att_weight = F.max_pool2d(parts_map, kernel_size=(W, H)).squeeze().detach()  # 8,85
        global_semantic_feat = F.avg_pool2d(parts_map, kernel_size=(W, H)).squeeze()
        global_semantic_feat = F.normalize(global_semantic_feat, dim=-1)
        # semantic_score = self.global_semantic_feat @ seen_att_normalized.T * self.scale_semantic
        # semantic score è la classificazione 8,40

        global_feat = F.avg_pool2d(x4, kernel_size=(W, H)).squeeze()  # 8x2048
        global_feat = self.fc_proto(global_feat)  # 8x2048
        global_feat = F.normalize(global_feat, dim=-1)  # 8x2048
        # cls_proto = self.proto_model(seen_att_normalized*np.sqrt(self.part_num),True) #40 x 2048
        # cls_proto = torch.nn.functional.normalize(cls_proto,p=2,dim=1) #40 x 2048
        # visual_score = torch.einsum('bd,nd->bn', global_feat, cls_proto) * self.scale #8x40

        att_weight = F.max_pool2d(parts_map, kernel_size=(W, H)).squeeze().detach()  # 8,85
        # self.att_weight = att_weight.gt(self.atten_thr)
        parts_map_flatten = parts_map.reshape(N, -1, W * H).softmax(dim=1)  # 8,85,196

        x5 = x4.reshape(N, C, -1)
        part_feats = torch.einsum('blr,bvr->blv', parts_map_flatten, x5.detach())  # 8,85,2048
        part_feats = self.fc_proto(part_feats)  # 8,85,2048

        pre_attri['final'] = global_feat
        pre_attri['global_semantic_feat'] = global_semantic_feat
        pre_attri['part_feats'] = part_feats
        pre_attri['att_weight'] = att_weight

        return pre_attri, None

    def normalize_vector(self, vector):
        return vector / (torch.norm(vector, p=2) + 1e-10)

    def calculate_axioms(self, pre_attri, attribute, label=False, label_m=None, attribute_macroclass_seen=None,
                         return_map=False,
                         get_prediction=False, axioms_options=None, opt=None, labels_test=None,
                         original_attribute=None, epoch=0):

        if opt.cropped_image:
            # print(labels_test)
            output_final = self.softmax(pre_attri['final'].mm(attribute) +
                                        pre_attri['final_masked'].mm(attribute))
        else:
            prototypes_class = self.proto_model(F.normalize(attribute.T, dim=-1) * np.sqrt(self.part_num), "is_cls")
            prototypes_class = torch.nn.functional.normalize(prototypes_class, p=2, dim=1)
            visual_score = torch.einsum('bd,nd->bn', F.normalize(pre_attri['final'], dim=-1),
                                        prototypes_class) * self.scale
            output_final = self.softmax(visual_score)

        images_x = ltn.Variable("images_x", output_final)
        images_x_prototypes = ltn.Variable("images_x_prototypes", pre_attri['final'])
        images_y = ltn.Variable("images_y", output_final)
        th_distant = ltn.Constant(torch.tensor(0.6))
        th_closer = ltn.Constant(torch.tensor(0.7))

        # images_y = ltn.Variable("images_y", output_final)
        label_classes = ltn.Variable("label_classes", torch.tensor(range(0, opt.seenclasses.shape[0])))
        if get_prediction == False:

            visual_score_global = torch.einsum('bd,nd->bn', F.normalize(pre_attri['global_semantic_feat'], dim=-1),
                                               F.normalize(attribute.T, dim=-1)) * self.scale
            output_final_global = self.softmax(visual_score_global)
            output_final_global = ltn.Variable("output_final_global", output_final_global)

            prototypes_class_fake = self.proto_model(
                create_fake(F.normalize(attribute.T, dim=-1) * np.sqrt(self.part_num),false_values_for_row=opt.k), "is_cls")
            prototypes_class_fake = torch.nn.functional.normalize(prototypes_class_fake, p=2, dim=1)
            visual_score = torch.einsum('bd,nd->bn', F.normalize(pre_attri['final'], dim=-1),
                                        prototypes_class_fake) * self.scale
            output_final_fake = self.softmax(visual_score)
            images_x_masked = ltn.Variable("images_x_masked", output_final_fake)

            macroclass_proto = self.proto_model(
                F.normalize(self.macroclass_vector[self.data.seen_macroclass], -1)
                * np.sqrt(self.macroclass_vector[self.data.seen_macroclass].shape[0]), "is_macro")
            macroclass_proto = F.normalize(macroclass_proto, dim=-1)
            label_x_macroclass_proto = ltn.Variable("label_x_macroclass_proto", self.data.seen_macroclass)
            # macroclass_proto = torch.index_select(macroclass_proto, dim=0, index=label_m)
            #output_final = torch.einsum('bd,nd->bn', F.normalize(pre_attri['final'], dim=-1),# macroclass_proto) * 25.0
            x_global_macroclass = ltn.Variable("x_global_macroclass", macroclass_proto)

            label_x_m = ltn.Variable("label_x_m =", label_m)
            images_x_features = ltn.Variable("images_x_features", F.normalize(pre_attri['final'], dim=-1))
            images_y_features = ltn.Variable("images_y_features", F.normalize(pre_attri['final'], dim=-1))

            prototypes_class_fake = self.proto_model(
                create_fake(F.normalize(attribute.T, dim=-1) * np.sqrt(self.part_num),false_values_for_row=opt.k), "is_cls")
            prototypes_class_fake = torch.nn.functional.normalize(prototypes_class_fake, p=2, dim=1)
            visual_score = torch.einsum('bd,nd->bn', F.normalize(pre_attri['final'], dim=-1),
                                        prototypes_class_fake) * 25.0
            output_final_fake = self.softmax(visual_score)

            att_proto = self.proto_model(F.normalize(self.attribute_vector, -1) * np.sqrt(self.part_num), "is_att")
            att_proto = F.normalize(att_proto, dim=-1)
            att_proto = att_proto.unsqueeze(0).repeat(pre_attri['part_feats'].shape[0], 1, 1).reshape(-1,
                                                                                                      self.attr_proto_size)
            attribute_features = pre_attri['part_feats'].reshape(-1, self.attr_proto_size)
            att_weight = pre_attri['att_weight'].reshape(-1, self.attritube_num)
            prototypes_class = ltn.Variable("prototypes_class", prototypes_class)

            mask = ((attribute.T[label] > 0) & (att_weight > 9)).view(-1, 1)
            attribute_features = torch.masked_select(attribute_features, mask).view(-1, self.attr_proto_size)
            att_proto = torch.masked_select(att_proto, mask).view(-1, self.attr_proto_size)

            features=torch.arange(self.part_num).unsqueeze(0).repeat(visual_score.shape[0], 1).view(-1, 1).to(torch.device("cuda:"+opt.gpu))
            index_of_attribute = torch.masked_select(features , mask).view(-1, 1)
            attribute_features = ltn.Variable("attribute_features", attribute_features)
            att_proto = ltn.Variable("att_proto", att_proto)
            index_of_attribute_x = ltn.Variable("index_of_attribute_x", index_of_attribute)
            index_of_attribute_y = ltn.Variable("index_of_attribute_y", index_of_attribute)

            images_x_masked = ltn.Variable("images_x_masked", output_final_fake)
            label_classes = ltn.Variable("label_classes", torch.tensor(range(0, opt.seenclasses.shape[0])))

            label_x = ltn.Variable("label_x", label)
            label_y = ltn.Variable("label_y", label)

            my_dict = {

            }

            # globali
            if opt.sat_agg_class:
                sat_agg_class = self.Forall(
                    ltn.diag(images_x, label_x),
                    self.isofclass(images_x, label_x),
                    p=axioms_options["p_axioms_class"]
                )
                my_dict["sat_agg_class"] = sat_agg_class
            if opt.sat_agg_class_global:
                sat_agg_class_global = self.Forall(
                    ltn.diag(output_final_global, label_x),
                    self.isofclass(output_final_global, label_x),
                    p=axioms_options["p_axioms_class"]
                )
                my_dict["sat_agg_class_global"] = sat_agg_class_global

            # globali
            """
            sat_euclidean_distance = self.Forall(
                ltn.diag(images_x_prototypes, label_x),
                self.Forall(
                    ltn.diag(prototypes_class, label_classes),
                    self.euclidean_distance(images_x_prototypes, prototypes_class),

                    cond_vars=[label_classes, label_x],
                    cond_fn=lambda x, y: torch.eq(x.value, y.value),
                    p=axioms_options["p_axioms_class"]
                ))
            my_dict["sat_euclidean_distance"] = sat_euclidean_distance
            """
            if opt.sat_agg_class_outlier_exists:
                sat_agg_class_outlier_exists = self.Forall(
                    label_classes,
                    self.Exists(ltn.diag(images_x_masked, label_x),
                                self.isofclass(images_x_masked, label_classes),
                                p=axioms_options["p_axioms_class_exists"],
                                cond_vars=[label_classes, label_x],
                                cond_fn=lambda x, y: torch.eq(x.value, y.value),

                                ))
                my_dict["sat_agg_class_outlier_exists"] = sat_agg_class_outlier_exists

            # ATTRIBUTI NUOVO
            if opt.sat_agg_same_attribute:
                if attribute_features.value.shape[0] != 0:
                    sat_agg_same_attribute = self.ForallSum(
                        ltn.diag(attribute_features, index_of_attribute_x),
                        self.Forall(
                            ltn.diag(att_proto, index_of_attribute_y),

                            self.same_class(attribute_features, att_proto),
                            cond_vars=[index_of_attribute_x, index_of_attribute_y],
                            cond_fn=lambda x, y: torch.eq(x.value, y.value)),
                        p=axioms_options["p_axioms_class_exists"]
                    )
                    my_dict["sat_agg_same_attribute"] = sat_agg_same_attribute

            if opt.sat_agg_macroclass_implied:
                sat_agg_macroclass_implied = self.ForallSum(
                    ltn.diag(x_global_macroclass, label_x_macroclass_proto),
                    self.Forall(
                        ltn.diag(images_x_features, images_x, label_x_m, label_x),
                        self.Implies(self.isofclass(images_x, label_x),
                                     self.same_class(x_global_macroclass, images_x_features)),
                        cond_vars=[label_x_macroclass_proto, label_x_m],
                        cond_fn=lambda x, y: torch.eq(x.value, y.value)),
                    p=axioms_options["p_axioms_class_exists"]
                )
                my_dict["sat_agg_macroclass_implied"] = sat_agg_macroclass_implied
            if opt.sat_agg_class_cluster_greater:
                sat_agg_class_cluster_greater = self.ForallSum(
                    ltn.diag(images_x_features, images_x),
                    self.ForallSum(
                        ltn.diag(images_y_features, images_y),
                        self.Forall(
                            label_classes,
                            self.Equiv(self.isofclass(images_x, label_classes),
                                       self.isofclass(images_y, label_classes)),
                            cond_vars=[images_x_features, images_y_features],
                            cond_fn=lambda x, y: torch.gt(self.cosine_d(x.value, y.value), th_closer.value, ),
                            p=2.0
                        ))
                )
                my_dict["sat_agg_class_cluster_greater"] = sat_agg_class_cluster_greater

            # 0.7
            if opt.sat_agg_class_cluster_lower:
                sat_agg_class_cluster_lower = self.ForallSum(
                    ltn.diag(images_x_features, images_x),
                    self.ForallSum(
                        ltn.diag(images_y_features, images_y),
                        self.Forall(
                            label_classes,
                            self.Not(self.And(self.isofclass(images_x, label_classes),
                                              self.isofclass(images_y, label_classes))),
                            cond_vars=[images_x_features, images_y_features],
                            cond_fn=lambda x, y: torch.le(self.cosine_d(x.value, y.value), th_distant.value),
                            p=2.0

                        )))
                my_dict["sat_agg_class_cluster_lower"] = sat_agg_class_cluster_lower

            """




            if False:
                # 0.2

                sat_agg_same_features = self.Forall(
                    ltn.diag(images_x_features, label_x),
                    self.Forall(
                        ltn.diag(images_y_features, label_y),

                        self.same_class(images_x_features, images_y_features),
                        cond_vars=[label_x, label_y],
                        cond_fn=lambda x, y: torch.eq(x.value, y.value)),
                    p=axioms_options["p_axioms_class_exists"]
                )
                my_dict["sat_agg_same_features"] = sat_agg_same_features

            """


            if opt.focalloss:
                return self.SatAgg(
                    my_dict.values(),
                    p=axioms_options["p_all"]), my_dict
            else:
                return -torch.log(self.SatAgg(
                    my_dict.values(),
                    p=axioms_options["p_all"])), my_dict





        else:
            attribute = ltn.Variable("attribute", attribute.T)
            predictions = images_x
            return predictions.value

    def get_score(self):
        return self.softmax(pre_attri['final'].mm(attribute) + pre_attri['final'].mm(attribute))

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


