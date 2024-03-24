import torch.nn as nn
import torch
import timm
import numpy as np
import ltn
from utils_zsl import create_fake
from model_proto import  similarity3, similarity4, cosine_similarity
import torch.nn.functional as F
from losses import SupConLoss
from nets import ProtoModel, euclid_distance


class ViT(nn.Module):
    def __init__(self, opt, attribute, binary_att, data, original_att, model_name="vit_large_patch16_224_in21k",
                 pretrained=True):
        super(ViT, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

        # Others variants of ViT can be used as well
        '''
        1 --- 'vit_small_patch16_224'
        2 --- 'vit_base_patch16_224'
        3 --- 'vit_large_patch16_224',
        4 --- 'vit_large_patch32_224'
        5 --- 'vit_deit_base_patch16_224'
        6 --- 'deit_base_distilled_patch16_224',
        '''

        # Change the head depending of the dataset used
        self.vit.head = nn.Identity()

        self.binary_att = binary_att
        self.parts = data.parts
        self.parts_key = data.parts_key
        self.data = data

        self.epsilon = 1e-4

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        if opt.dataset == 'CUB':
            self.attr_proto_size = 1024
            self.part_num=312
            hid_size = 1024
        elif opt.dataset == 'AWA1':
            exit(1)
            self.ALE = LINEAR_SOFTMAX_ALE(input_dim=self.channel_dict['avg_pool'], attri_dim=85)
        elif opt.dataset == 'AWA2':
            """
            self.prototype_vectors = dict()
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            # self.ALE_vector = nn.Parameter(2e-4 * torch.rand([85, 768, 1, 1]), requires_grad=True)

            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([768, 85]), requires_grad=True)
            self.ALE_vector_cropped = nn.Parameter(2e-4 * torch.rand([85, 2048, 1, 1]), requires_grad=True)
            self.MLP = MLP(9, 9)


            self.ALE_PARTS = nn.Parameter(2e-4 * torch.rand([85, 9]), requires_grad=True)
            self.hasAttributeofclassLambda = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
            self.macroclass_vector = nn.Parameter(2e-4 * torch.rand([85, 9]), requires_grad=True)
            self.attribute_vector = nn.Parameter(torch.rand([85, 85]), requires_grad=True)  # 原型参数
            """
            self.attr_proto_size = 1024
            self.part_num = 85
            hid_size = 1024
        elif opt.dataset == 'SUN':
            self.prototype_vectors = dict()
            for name in self.extract:
                prototype_shape = [102, self.channel_dict[name], 1, 1]
                self.prototype_vectors[name] = nn.Parameter(2e-4 * torch.rand(prototype_shape),
                                                            requires_grad=True)  # 原型参数
            self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)
            self.ALE_vector = nn.Parameter(2e-4 * torch.rand([102, 1024, 1, 1]), requires_grad=True)  # 全局分支参数

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
        self.same_part = ltn.Function(func=lambda x, y: similarity4(x, y))
        self.hasAttributeEndError = ltn.Function(func=lambda x, y: get_and_of_attributes_error(x,
                                                                                               y))  # ltn.Predicate(model = Has_attribute())#[ltn.Predicate(model=Predicate_ltn(num_features=85,name=f.__str__()))  for f in range(85)]
        # self.hasAttributeModel = Has_attribute()
        """
        if opt.cuda:
            for f in range(len(self.hasAttribute)):
                self.hasAttribute[f]=self.hasAttribute[f].cuda()
        """
        """
        self.hasAttribute = ltn.Function(func=lambda x, y: torch.exp(-(1-0.99*F.cosine_similarity(x,y))).view(-1))
        self.hasAttributeofclass = ltn.Function(func=lambda x, y: torch.gather(x,1,y).view(-1))
        """
        # self.hasAttribute = Predicate_ltn_same_class(name="hasAttribute",num_features=85*2)
        # self.hasAttributelambda = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
        # self.hasAttribute = ltn.Function(func=lambda x, y: torch.gather(x, 1, y).view(-1))
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="f")

        self.SatAgg = ltn.fuzzy_ops.SatAgg()
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.dist = lambda x, y: torch.unsqueeze(torch.norm(x - y, dim=1), dim=1)
        self.cosine_d = lambda x, y: torch.unsqueeze(
            torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(x, p=2, dim=1),
                                                  torch.nn.functional.normalize(y, p=2, dim=1)), dim=1)
        self.same_attribute = lambda x, y: torch.unsqueeze(same_attribute(x, y), dim=1)
        self.not_same_attribute = lambda x, y: torch.unsqueeze(not_same_attribute(x, y), dim=1)
        self.w2v_att = torch.tensor(np.float32(data.w2v))
        _, self.w2v_length = self.w2v_att.shape
        self.med_dim = 300  # 1024
        self.feat_channel = 2048
        self.feat_w = 7
        self.feat_h = 7
        """
        self.QueryW = nn.Sequential(nn.Linear(self.w2v_length, self.med_dim))  # L,M = 300,1024
        self.QueryW.weight = nn.Parameter(2e-4 * torch.rand([self.med_dim, self.med_dim]), requires_grad=False)
        self.KeyW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim))  # C,M = 2048,1024
        self.KeyW.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.med_dim]), requires_grad=False)
        self.ValueW = nn.Sequential(nn.Linear(self.feat_channel, self.med_dim))  # C,M = 2048,1024
        self.ValueW.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.med_dim]), requires_grad=False)
        self.W_o = nn.Sequential(nn.Linear(self.med_dim, self.feat_channel))  # M,C = 1024,2048
        self.W_o.weight = nn.Parameter(2e-4 * torch.rand([self.med_dim, self.feat_channel]), requires_grad=False)
        self.W = nn.Parameter(2e-4 * torch.rand([self.w2v_att.shape[1], self.feat_channel]),
                              requires_grad=True)  # 300 * 2048
        """

        self.finale = nn.Sequential(nn.Linear(self.feat_channel, self.feat_channel))  # M,C = 1024,2048

        self.attritube_num = self.w2v_att.shape[0]
        attributes = torch.where(self.data.original_att.double() < 0.0, 0.0, self.data.original_att.double())
        attributes = attributes / torch.max(attributes)
        self.attributes = attributes.to(opt.device)

        self.V = nn.Sequential(nn.Linear(self.feat_channel, self.attritube_num))  # V, S = [2048,4096]
        self.V.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.attritube_num]), requires_grad=True)
        self.hid_dim = 2048
        self.V_att_hidden_branch = nn.Sequential(nn.Linear(self.feat_channel, self.hid_dim))  # H, C
        self.V_att_hidden_branch.weight = nn.Parameter(2e-4 * torch.rand([self.feat_channel, self.hid_dim]),
                                                       requires_grad=True)
        self.ratio = 1.0
        self.V_att_final_branch = nn.Parameter(2e-4 * torch.rand([self.attritube_num, self.hid_dim]),
                                               requires_grad=True)
        self.V_att_final_classification = nn.Parameter(2e-4 * torch.rand([self.attritube_num, 50]),
                                                       requires_grad=True)
        self.V_att_conversion = nn.Sequential(nn.Dropout(0.5), nn.Linear(self.med_dim, 85))
        self.fc_attention_channel = nn.Linear(self.attr_proto_size, self.part_num)
        nn.init.xavier_uniform_(self.fc_attention_channel.weight)

        self.con_loss = SupConLoss()

        self.proto_model = ProtoModel(self.part_num, hid_size, self.attr_proto_size, with_cn=True)
        self.fc_proto = nn.Identity()
        self.fc_proto = nn.Identity()
        self.attribute_vector = nn.Parameter(torch.eye(self.part_num), requires_grad=False)
        self.macroclass_vector = nn.Parameter(2e-4 * torch.rand(self.data.attribute_macroclass.shape[0], self.part_num),
                                              requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1) * 25.0, requires_grad=False)
        self.euclidean_distance = ltn.Function(func=lambda x, y: euclid_distance(x, y,opt.p))

    def forward(self, x, attribute, label=False, label_m=None, attribute_macroclass_seen=None, return_map=False,
                get_prediction=False, axioms_options=None, opt=None, extract_bb=None, original_attribute=None,
                paths=None):

        pre_attri = {}
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        features = x[:, 0]
        patch_feat = x[:, 1:]
        N, S, C = patch_feat.shape
        W = int(np.sqrt(S))
        H = W
        parts_map = self.fc_attention_channel(patch_feat)
        parts_map = parts_map.transpose(1, 2).reshape(N, -1, W, H)
        global_semantic_feat = F.avg_pool2d(parts_map, kernel_size=(W, H)).squeeze()
        global_semantic_feat = F.normalize(global_semantic_feat, dim=-1)
        # semantic_score = self.global_semantic_feat @ attribute.T * 25.

        att_weight = F.max_pool2d(parts_map, kernel_size=(W, H)).squeeze().detach()
        parts_map_flatten = parts_map.reshape(N, -1, W * H).softmax(dim=1)
        part_feats = torch.einsum('blr,brv->blv', parts_map_flatten, patch_feat.detach())

        # init_features = features.mm(self.ALE_vector)
        pre_attri['final'] = features
        pre_attri['global_semantic_feat'] = global_semantic_feat
        pre_attri['part_feats'] = part_feats
        pre_attri['att_weight'] = att_weight

        return pre_attri, None

    def calculate_axioms(self, pre_attri, attribute, label=False, label_m=None, attribute_macroclass_seen=None,
                         return_map=False,
                         get_prediction=False, axioms_options=None, opt=None, labels_test=None,
                         original_attribute=None, epoch=None):

        if opt.cropped_image:
            # print(labels_test)
            output_final = self.softmax(pre_attri['final'].mm(attribute) +
                                        pre_attri['final_masked'].mm(attribute))
        else:
            prototypes_class = self.proto_model(F.normalize(attribute.T, dim=-1) * np.sqrt(self.part_num), "is_cls").to(opt.device)
            prototypes_class = torch.nn.functional.normalize(prototypes_class, p=2, dim=1).to(opt.device)
            visual_score = torch.einsum('bd,nd->bn', F.normalize(pre_attri['final'], dim=-1), prototypes_class) * 25.0
            prototypes_class = ltn.Variable("prototypes_class",prototypes_class)
            output_final = self.softmax(visual_score)

        images_x = ltn.Variable("images_x", output_final)
        images_y = ltn.Variable("images_y", output_final)
        th_distant = ltn.Constant(torch.tensor(0.3))
        th_closer = ltn.Constant(torch.tensor(0.5))
        images_x_prototypes = ltn.Variable("images_x_prototypes", pre_attri['final'])

        # images_y = ltn.Variable("images_y", output_final)
        if get_prediction == False:
            visual_score_global = torch.einsum('bd,nd->bn', F.normalize(pre_attri['global_semantic_feat'], dim=-1),
                                               F.normalize(attribute.T, dim=-1)) * self.scale
            output_final_global = self.softmax(visual_score_global)
            output_final_global = ltn.Variable("output_final_global", output_final_global)

            macroclass_proto = self.proto_model(
                F.normalize(self.macroclass_vector[self.data.seen_macroclass], -1)
                * np.sqrt(self.macroclass_vector[self.data.seen_macroclass].shape[0]), "is_macro")
            macroclass_proto = F.normalize(macroclass_proto, dim=-1)
            label_x_macroclass_proto = ltn.Variable("label_x_macroclass_proto", self.data.seen_macroclass)
            # macroclass_proto = torch.index_select(macroclass_proto, dim=0, index=label_m)
            x_global_macroclass = ltn.Variable("x_global_macroclass", macroclass_proto)

            label_x_m = ltn.Variable("label_x_m =", label_m)
            images_x_features = ltn.Variable("images_x_features", F.normalize(pre_attri['final'], dim=-1))
            images_y_features = ltn.Variable("images_y_features", F.normalize(pre_attri['final'], dim=-1))

            prototypes_class_fake = self.proto_model(
                create_fake(F.normalize(attribute.T, dim=-1) * np.sqrt(self.part_num),opt.k), "is_cls")
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

            mask = ((original_attribute.T[label] > 0) & (att_weight > 9)).view(-1, 1)
            attribute_features = torch.masked_select(attribute_features, mask).view(-1, self.attr_proto_size)
            att_proto = torch.masked_select(att_proto, mask).view(-1, self.attr_proto_size)
            elemenets = torch.arange(self.part_num).unsqueeze(0).repeat(visual_score.shape[0], 1).view(-1, 1).to(opt.device)
            index_of_attribute = torch.masked_select(elemenets , mask).view(-1, 1).to(opt.device)
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
                sat_agg_same_attribute = self.Forall(
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
                sat_agg_macroclass_implied = self.Forall(
                    ltn.diag(x_global_macroclass, label_x_macroclass_proto),
                    self.Forall(
                        ltn.diag(images_x_features, images_x, label_x_m, label_x),
                        self.Implies(self.isofclass(images_x, label_x),
                                     self.same_class(images_x_features, x_global_macroclass)),
                        cond_vars=[label_x_macroclass_proto, label_x_m],
                        cond_fn=lambda x, y: torch.eq(x.value, y.value)),
                    p=axioms_options["p_axioms_class_exists"]
                )
                my_dict["sat_agg_macroclass_implied"] = sat_agg_macroclass_implied

            if opt.sat_agg_class_cluster_greater:
                sat_agg_class_cluster_greater = self.Forall(
                    ltn.diag(images_x_features, images_x),
                    self.Forall(
                        ltn.diag(images_y_features, images_y),
                        self.Forall(
                            label_classes,
                            self.Equiv(self.isofclass(images_x, label_classes),
                                       self.isofclass(images_y, label_classes)),
                            cond_vars=[images_x_features, images_y_features],
                            cond_fn=lambda x, y: torch.gt(self.cosine_d(x.value, y.value), th_closer.value, ),
                            p=1.0
                        ))
                )
                my_dict["sat_agg_class_cluster_greater"] = sat_agg_class_cluster_greater

            # 0.7
            if opt.sat_agg_class_cluster_lower:
                sat_agg_class_cluster_lower = self.Forall(
                    ltn.diag(images_x_features, images_x),
                    self.Forall(
                        ltn.diag(images_y_features, images_y),
                        self.Forall(
                            label_classes,
                            self.Not(self.And(self.isofclass(images_x, label_classes),
                                              self.isofclass(images_y, label_classes))),
                            cond_vars=[images_x_features, images_y_features],
                            cond_fn=lambda x, y: torch.le(self.cosine_d(x.value, y.value), th_distant.value),
                            p=1.0

                        )))
                my_dict["sat_agg_class_cluster_lower"] = sat_agg_class_cluster_lower


            if opt.sat_agg_class_global:
                sat_agg_class_global = self.Forall(
                    ltn.diag(output_final_global, label_x),
                    self.isofclass(output_final_global, label_x),
                    p=axioms_options["p_axioms_class"]
                )
                my_dict["sat_agg_class_global"] = sat_agg_class_global

            # globali

            return -torch.log((self.SatAgg(
                list(my_dict.values()),
                p=axioms_options["p_all"]))), my_dict








        else:
            attribute = ltn.Variable("attribute", attribute.T)
            predictions = images_x
            return predictions.value


if __name__ == '__main__':
    # r18_features = resnet18_features(pretrained=True)
    # print(r18_features)
    #
    # r34_features = resnet34_features(pretrained=True)
    # print(r34_features)
    #
    # r50_features = resnet50_features(pretrained=True)
    # print(r50_features)

    vit_features = ViT(model_name='vit_base_patch16_224', pretrained=True)
    print(vit_features)
