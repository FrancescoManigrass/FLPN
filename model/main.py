from __future__ import print_function

import itertools
import os
from datetime import datetime
from os.path import join

import neptune
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
import visual_utils
import sys
import random

from custom_transformations import crop_images
from ViT import ViT

from neptune_logging import log_loss_training
from visual_utils import ImageFilelist, compute_per_class_acc, compute_per_class_acc_gzsl, \
    prepare_attri_label, add_glasso, add_dim_glasso
from logger import Logger
# from utils import init_log



import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import json
from main_utils import test_zsl, calibrated_stacking, test_gzsl, test_gzsl_loss, \
    calculate_average_IoU, test_with_IoU
from main_utils import set_randomseed, get_loader, get_middle_graph, Loss_fn, Result, SupConLoss_clear, \
    mse_loss, update_ema_variables, get_current_consistency_weight
from opt import get_opt

# from setproctitle import setproctitle
# setproctitle('wanggerong')


cudnn.benchmark = True

opt = get_opt()
# set random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)


from model_proto import resnet_proto_IoU



torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
random.seed(opt.manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic=True

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


def main():
    torch.cuda.set_device("cuda:" + opt.gpu)
    opt.device="cuda:" + opt.gpu
    # load data
    data = visual_utils.DATA_LOADER(opt)
    # print(opt)
    opt.test_seen_label = data.test_seen_label  # weird

    if opt.neptune_flag:

        opt.experiment = neptune.init(opt.project_name.__str__(), api_token=opt.token.__str__())

        # create experiment with defined parameters, uploaded source code and tags

        neptune.create_experiment(name=opt.experiment.__str__(), params=vars(opt))
        files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(".") for f in filenames if
                 os.path.splitext(f)[1] == '.py']

        for f in files:
            neptune.log_artifact(f.replace("//", "/"))

        opt.neptune = neptune

    print("neptune finished")

    # define test_classes
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # prepare the attribute labels
    class_attribute = data.attribute
    class_attribute_original = data.original_att
    print("attribute to cuda")
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).to(torch.device("cuda:" + opt.gpu))
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).to(torch.device("cuda:" + opt.gpu))
    attribute_zsl_original = prepare_attri_label(class_attribute_original, data.unseenclasses).to(torch.device("cuda:" + opt.gpu))
    attribute_seen_original = prepare_attri_label(class_attribute_original, data.seenclasses).to(torch.device("cuda:" + opt.gpu))
    if opt.dataset != "SUN":
        attribute_macroclass_seen = prepare_attri_label(data.attribute_macroclass, data.seen_macroclass).to(torch.device("cuda:" + opt.gpu))
    else:

        attribute_macroclass_seen = torch.tensor(1, requires_grad=False)

        attribute_macroclass_seen = attribute_macroclass_seen.to("cuda:" + opt.gpu)

    attribute_gzsl = torch.transpose(class_attribute, 1, 0).to(torch.device("cuda:" + opt.gpu))
    attribute_gzsl_original = torch.transpose(class_attribute_original, 1, 0).to(torch.device("cuda:" + opt.gpu))

    print("init dataloader")
    # Dataloader for train, test, visual
    opt.unseenclasses = data.unseenclasses

    opt.seenclasses = data.seenclasses
    opt.total_classes = torch.sort(torch.cat((opt.seenclasses, opt.unseenclasses)))
    print("create dataloaders")
    trainloader, testloader_unseen, testloader_seen, visloader = get_loader(opt, data)
    print("train data loader initialized")

    # experiment name
    experiment_name = 'pretrainepoch{}_pretrainlr{}_classifierlr{}_{}ways-{}shots-{}'.format(opt.pretrain_epoch,
                                                                                             opt.pretrain_lr,
                                                                                             opt.classifier_lr,
                                                                                             opt.ways, opt.shots,
                                                                                             opt.contrative_loss_weight)
    # visual_loss_path
    visual_loss_path = './out/visual/alpha/' + experiment_name + '/'
    logger = Logger(visual_loss_path)
    print("logger ready")

    # store_result_path
    save_dir = './out/result/alpha/{}.txt'.format(experiment_name)
    # logging = init_log(save_dir)
    # _print = logging.info

    weight_path = './out/weight/alpha/' + experiment_name + '/'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # define attribute groups
    if opt.dataset == 'CUB':
        # parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
        # group_dic = json.load(open(os.path.join(os.path.dirname(os.getcwd()),opt.root, 'data', opt.dataset, 'attri_groups_8.json')))
        # sub_group_dic = json.load(open(os.path.join(os.path.dirname(os.getcwd()),opt.root, 'data', opt.dataset, 'attri_groups_8_layer.json')))
        opt.resnet_path = 'pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'AWA2':
        # parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat', 'character']
        # `group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_9.json')))
        # sub_group_dic = {}

        opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'
    elif opt.dataset == 'SUN':
        # parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
        # group_dic = json.load(open(os.path.join(opt.root, 'data', opt.dataset, 'attri_groups_4.json')))
        # sub_group_dic = {}
        opt.resnet_path = 'pretrained_models/resnet101-5d3b4d8f.pth'
    # initialize model
    print('Create Model...')

    def create_model(attribute, binary_att, data, original_att):

        if opt.model == "resnet101":
            model = resnet_proto_IoU(opt, attribute, binary_att, data, original_att)
        else:
            model = ViT(opt=opt, attribute=attribute, binary_att=binary_att, data=data, original_att=original_att, model_name='vit_large_patch16_224_in21k',
                        pretrained=True)

        return model

    model = create_model(attribute=data.attribute, binary_att=data.binary_att, data=data,
                         original_att=data.original_att)

    if torch.cuda.is_available():
        model.to(torch.device("cuda:" + opt.gpu))

        attribute_zsl = attribute_zsl.to(torch.device("cuda:" + opt.gpu))
        attribute_seen = attribute_seen.to(torch.device("cuda:" + opt.gpu))
        attribute_macroclass_seen = attribute_macroclass_seen.to(torch.device("cuda:" + opt.gpu))
        attribute_gzsl = attribute_gzsl.to(torch.device("cuda:" + opt.gpu))
        model.binary_att = model.binary_att.to(torch.device("cuda:" + opt.gpu))
        #model.parts_key = model.parts_key.to(torch.device("cuda:" + opt.gpu))
        #model.parts = model.parts.to(torch.device("cuda:" + opt.gpu))
        attribute_seen_original = attribute_seen_original.to(torch.device("cuda:" + opt.gpu))

    # layer_name = model.extract[0]  # only use one layer currently
    # compact loss configuration, define middle_graph
    # middle_graph = get_middle_graph(reg_weight[layer_name]['cpt'], model)

    # train and test

    result_gzsl_student = Result()
    result_zsl_student = Result()

    if opt.only_evaluate:
        print('Evaluate ...')

        model.load_state_dict(torch.load(opt.resume_path))
        print("eval mode from ", opt.resume_path)
        model.eval()
        # test zsl
        if not opt.gzsl:
            acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses)
            print('ZSL test accuracy is {:.1f}%'.format(acc_ZSL))
        else:
            # test gzsl
            acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses)
            acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses)

            if (acc_GZSL_unseen + acc_GZSL_seen) == 0:
                acc_GZSL_H = 0
            else:
                acc_GZSL_H = 2 * acc_GZSL_unseen * acc_GZSL_seen / (
                        acc_GZSL_unseen + acc_GZSL_seen)

            print('GZSL test accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'.format(acc_GZSL_unseen, acc_GZSL_seen,
                                                                                      acc_GZSL_H))
    else:
        print('Train and test...')

        if opt.resume:
            print("Train and test... mode from ", opt.resume_path)
            model.load_state_dict(torch.load(opt.resume_path))

        global_step = 0
        for epoch in range(opt.nepoch):
            # print("training")
            model.train()

            current_lr = opt.classifier_lr * (0.5 ** (epoch // 10))
            realtrain = epoch > opt.pretrain_epoch
            # if epoch <= opt.pretrain_epoch:   # pretrain ALE for the first several epoches
            if not opt.resume and epoch < opt.pretrain_epoch:  # pretrain ALE for the first several epoches
                print('Pretraining with freezed layers')

                # parameters = [model.ALE_vector]  #
                parameters = [model.macroclass_vector]

                for i in [model.proto_model]:  # ,model.finale
                    parameters.extend(list(i.parameters()))

                if opt.model == "resnet101":
                    for i in [model.extract_1, model.extract_2, model.extract_3, model.extract_4, model.fc_proto]:
                        parameters.extend(list(i.parameters()))
                else:
                    model.vit.training = False

                # parameters.extend(list(itertools.chain(*[list(f.parameters()) for f in model.isOfClassLTN.layers])))
                if opt.optimizer == "adam":
                    optimizer = optim.Adam(params=parameters, lr=opt.pretrain_lr, betas=(opt.beta1, 0.999))
                if opt.optimizer == "sgd":
                    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, parameters), lr=current_lr,
                                          momentum=0.9, weight_decay=0.0001)
                # optimizer = optim.SGD(params=parameters, lr=opt.pretrain_lr, momentum=0.9,weight_decay=0.0001)
            else:
                if opt.model == "vit":
                    model.vit.training = True
                print('All layers training')
                #parameters = [f for f in model.parameters()]

                params_to_update = []
                params_names = []

                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        params_names.append(name)
                if opt.optimizer == "adam":
                    optimizer = optim.Adam(params=params_to_update, lr=current_lr,
                                           betas=(opt.beta1, 0.999))
                if opt.optimizer == "sgd":
                    optimizer = optim.SGD(params=params_to_update, lr=current_lr,
                                          momentum=0.9, weight_decay=0.00001)

            loss_log = {'ave_loss': 0}

            # trainloader, testloader_unseen, testloader_seen, visloader
            # i, (input, target, impath)
            # for i, (batch_input, batch_input2, batch_target, impath)

            batch = len(trainloader)  #### �n� n_batch = 1000
            i = batch - 1
            if True:
                step = 0
                # trainloader
                for i, (batch_input, batch_input2, batch_target, impath) in tqdm(enumerate(trainloader),
                                                                                 total=len(trainloader)):

                    start_time = datetime.now()
                    # print(datetime.now() - start_time)
                    step += 1
                    model.zero_grad()
                    # map target labels
                    if len(batch_target) == 2:
                        batch_target_macroclass = batch_target[1]
                        batch_target = batch_target[0]
                        """
                        if torch.max(batch_target_macroclass) > 8:
                            print("dfdfd")
                        """
                        batch_target_macroclass2 = visual_utils.map_label(batch_target_macroclass, data.seen_macroclass)
                        """
                        if torch.max(batch_target_macroclass2) > 8:
                            print("dfdfd")
                        """

                    batch_target = visual_utils.map_label(batch_target, data.seenclasses)

                    input_v = Variable(batch_input)

                    label_v = Variable(batch_target)
                    if opt.dataset == "SUN":
                        batch_target_macroclass2 = torch.tensor(1, requires_grad=False)
                    label_m = Variable(batch_target_macroclass2)

                    if opt.cuda:
                        input_v = input_v.to(torch.device("cuda:" + opt.gpu))
                        label_v = label_v.to(torch.device("cuda:" + opt.gpu))
                        label_m = label_m.to(torch.device("cuda:" + opt.gpu))

                    axioms_options = {"p_axioms_class": 2.0, "p_axioms_macroclass": 1.0,
                                      "p_axioms_class_exists": 1.0, "p_axioms_class_cluster": 1.0, "p_all": 2.0}

                    if epoch >= 4:
                        axioms_options = {"p_axioms_class": 2.0, "p_axioms_macroclass": 2.0,
                                          "p_axioms_class_exists": 2.0, "p_axioms_class_cluster": 2.0, "p_all": 2.0}

                    if epoch >= 8:
                        if opt.axioms_exists:
                            axioms_options["p_axioms_class"] = 2.0
                            axioms_options["p_axioms_class_cluster"] = 4.0
                        if opt.weight_exists:
                            axioms_options["p_axioms_class_exists"] = 2.0
                        if opt.axioms_exists:
                            axioms_options["p_axioms_class"] = 4.0
                        if opt.macroclass_exists:
                            axioms_options["p_axioms_macroclass"] = 4.0
                        axioms_options["p_all"] = 4.0

                    if epoch >= 12:
                        if opt.axioms_exists:
                            axioms_options["p_axioms_class"] = 2.0
                            axioms_options["p_axioms_class_cluster"] = 6.0
                        if opt.weight_exists:
                            axioms_options["p_axioms_class_exists"] = 2.0
                        if opt.macroclass_exists:
                            axioms_options["p_axioms_macroclass"] = 6.0
                        axioms_options["p_all"] = 6.0

                    pre_attri, boxes = model(input_v, attribute=attribute_seen, label=label_v,
                                             label_m=label_m, attribute_macroclass_seen=attribute_macroclass_seen,
                                             axioms_options=axioms_options, opt=opt, extract_bb=True,
                                             original_attribute=attribute_seen_original, paths=impath)

                    if opt.cropped_image:
                        # print(datetime.now() - start_time)
                        cropped_images = crop_images(input_v, boxes, opt)

                        if opt.print_images:
                            print("-------- printing images-------------")
                            paths = [os.path.split(f)[-1] for f in impath]
                            os.makedirs("output_images", exist_ok=True)
                            for k in range(len(boxes)):
                                folder = paths[k].split("_")[0]
                                name_image = paths[k].split("_")[1]

                                os.makedirs(join("output_images", folder), exist_ok=True)

                                image = Image.open(
                                    opt.image_root + "/" + opt.dataset + "/" + "JPEGImages" + "/"
                                    + folder + "/" + folder + "_" + name_image)

                                image = image.resize((224, 224))

                                cropped_img = image.crop(
                                    (int(boxes[0][0]), int(boxes[0][1]), int(boxes[0][2]), int(boxes[0][3])))
                                cropped_img.save(f'output_images/{folder}/{folder}_{name_image}')

                                """
                                fig = plt.figure()
                                plt.imshow(cropped_images[k][0, :].cpu().detach().numpy())
                                # plt.show()
                                folder = paths[k].split("_")[0]
                                name_image =  paths[k].split("_")[1]
                                os.makedirs(join("output_images",folder),exist_ok=True)
                                fig.savefig(f'output_images/{folder}/{name_image}', dpi=fig.dpi)

                                """
                                """

                                fig = plt.figure()
                                plt.imshow(input_v[k][0, :].cpu().detach().numpy())
                                # plt.show()
                                fig.savefig(f'output_images/{k}_original.png', dpi=fig.dpi)
                                """

                        if opt.cuda:
                            cropped_images = cropped_images.to(torch.device("cuda:" + opt.gpu))
                        # print(datetime.now() - start_time)

                        pre_attri, boxes = model(cropped_images, attribute=attribute_seen, label=label_v,
                                                 label_m=label_m, attribute_macroclass_seen=attribute_macroclass_seen,
                                                 axioms_options=axioms_options, opt=opt,
                                                 original_attribute=attribute_seen_original, paths=paths)

                        # print(datetime.now() - start_time)

                    def print_features(t, j, data):
                        paths = [os.path.split(f)[-1] for f in impath]

                        for k in range(pre_attri["part_attention"].shape[0]):

                            os.makedirs("features", exist_ok=True)
                            folder = paths[k].split("_")[0]
                            name_image = paths[k].split("_")[1]

                            os.makedirs(join("output_images", folder), exist_ok=True)

                            image = Image.open(
                                opt.image_root + "/" + opt.dataset + "/" + "JPEGImages" + "/"
                                + folder + "/" + folder + "_" + name_image)

                            image = image.resize((224, 224))
                            fig = plt.figure(figsize=(7, 7))

                            plt.imshow(image)
                            fig.savefig(f'features/{name_image}.png', dpi=fig.dpi)
                            part_index = 0
                            for i in range(pre_attri["part_attention"].shape[1]):
                                features = pre_attri["part_attention"][k][i].view(7, 7)
                                fig = plt.figure(figsize=(7, 7))
                                features = features.cpu().detach().numpy()
                                plt.imshow(features)
                                fig.savefig(f'features/{name_image}_{data.attri_name[part_index]}.png', dpi=fig.dpi)
                                part_index += 1

                    if opt.print_images:
                        print_features(0, 36, data)

                    loss, list_axioms = model.calculate_axioms(pre_attri, attribute=attribute_seen, label=label_v,
                                                               label_m=label_m,
                                                               attribute_macroclass_seen=attribute_macroclass_seen,
                                                               axioms_options=axioms_options, opt=opt,
                                                               labels_test=data.seenclasses,
                                                               original_attribute=attribute_seen_original, epoch=epoch)

                    # print(datetime.now() - start_time)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()

                    label_a = attribute_seen[:, label_v].t()

                    alpha = -23
                    beta = -18
                    mu = 1 / (7 * 7)
                    # loss1 = torch.exp(alpha * (torch.max(pre_attri["map_attention"], dim=1)[0] + beta * mu))
                    # loss1 = torch.sum(loss1) / loss1.shape[0]

                    # loss2 = torch.exp(alpha * (torch.max(pre_attri["map_attention_cropped"], dim=1)[0] + beta * mu))
                    # loss2 = torch.sum(loss2) / loss2.shape[0]

                    loss_log['ave_loss'] += loss.item()  # + loss1.item() + loss2.item()
                    for f in list_axioms.keys():
                        if f in loss_log:
                            loss_log[f] += list_axioms[f].value
                        else:
                            loss_log[f] = list_axioms[f].value

                    # print(datetime.now() - start_time)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # print(datetime.now() - start_time)
                    if opt.neptune_flag:
                        log_loss_training(opt, loss_log, step)

                    global_step += 1
                    # print("finished")
                    del input_v
                    del boxes
                    del pre_attri
                    torch.cuda.empty_cache()

            # print('\nLoss log: {}'.format({key: loss_log[key] / batch for key in loss_log}))
            print('\n[Epoch %d, Batch %5d] Train loss: %.3f ' % (epoch + 1, batch, loss_log['ave_loss'] / batch))
            # logger.scalar_summary('Train loss', loss_log['ave_loss'] / batch, epoch+1)
            # logger.scalar_summary('Train contrastive loss', loss_log['contrastive_loss'] / batch, epoch+1)
            # logger.scalar_summary('Train consistency loss', loss_log['consistency_loss'] / batch, epoch+1)
            axioms_log = {'zsl': 0, 'top1_seen': 0, 'top1_unseen': 0, "H_gzsl": 0}
            if (i + 1) == batch:  # or epoch % 200 == 0
                ###### test #######
                # print("testing")
                model.eval()

                # test zsl

                #### test zsl student
                student_acc_ZSL = test_zsl(opt, model, testloader_unseen, attribute_zsl, data.unseenclasses,
                                           attribute_original=attribute_zsl_original)
                # student_loss_ZSL, student_loss_contrastive_ZSL = test_gzsl_loss(opt, model, testloader_unseen, attribute_gzsl,reg_weight,criterion, criterion_regre,realtrain,middle_graph,parts,group_dic,sub_group_dic)

                # print('Test student ZSL loss: %.3f'% student_loss_ZSL)
                # logger.scalar_summary('Test ZSL loss', student_loss_ZSL, epoch+1)
                if student_acc_ZSL > result_zsl_student.best_acc:
                    # save model state
                    if opt.neptune_flag:
                        model_save_path = os.path.join(
                            weight_path + '/{}_{}__ZSL_id_{}.pth'.format(opt.experiment._experiments_stack[0], opt.dataset,
                                                                         opt.train_id))
                        torch.save(model.state_dict(), model_save_path)
                        print('model saved to:', model_save_path)
                result_zsl_student.update(epoch + 1, student_acc_ZSL)
                axioms_log["zsl"] += student_acc_ZSL
                print('\n[Epoch {}] ZSL test accuracy is {:.1f}%, Best_acc [{:.1f}% | Epoch-{}]'.format(epoch + 1,
                                                                                                        student_acc_ZSL,
                                                                                                        result_zsl_student.best_acc,
                                                                                                        result_zsl_student.best_iter))

                # test gzsl student model
                student_acc_GZSL_unseen = test_gzsl(opt, model, testloader_unseen, attribute_gzsl, data.unseenclasses,
                                                    data.allclasses, attribute_gzsl_original)
                student_acc_GZSL_seen = test_gzsl(opt, model, testloader_seen, attribute_gzsl, data.seenclasses,
                                                  data.allclasses, attribute_gzsl_original)
                axioms_log["top1_seen"] += student_acc_GZSL_seen
                axioms_log["top1_unseen"] += student_acc_GZSL_unseen

                if (student_acc_GZSL_unseen + student_acc_GZSL_seen) == 0:
                    student_acc_GZSL_H = 0
                else:
                    student_acc_GZSL_H = 2 * student_acc_GZSL_unseen * student_acc_GZSL_seen / (
                            student_acc_GZSL_unseen + student_acc_GZSL_seen)

                axioms_log["H_gzsl"] += student_acc_GZSL_H

                log_loss_training(opt, axioms_log)

                if student_acc_GZSL_H > result_gzsl_student.best_acc:
                    # save model state
                    # model_save_path = os.path.join('./out/{}_GZSL_id_{}.pth'.format(opt.dataset, opt.train_id))
                    model_save_path = os.path.join(
                        weight_path + '{}_{}_student_GZSL_id_{}.pth'.format(opt.experiment._experiments_stack[0],
                                                                            opt.dataset, opt.train_id))
                    torch.save(model.state_dict(), model_save_path)
                    print('model saved to:', model_save_path)

                result_gzsl_student.update_gzsl(epoch + 1, student_acc_GZSL_unseen, student_acc_GZSL_seen,
                                                student_acc_GZSL_H)

                print('\n[Epoch {}] GZSL test student accuracy is Unseen: {:.1f} Seen: {:.1f} H:{:.1f}'
                      '\n           Best_H student [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.
                      format(epoch + 1, student_acc_GZSL_unseen, student_acc_GZSL_seen, student_acc_GZSL_H,
                             result_gzsl_student.best_acc_U, result_gzsl_student.best_acc_S,
                             result_gzsl_student.best_acc, result_gzsl_student.best_iter))

                # X�.txt
                print('--' * 60)  # Sp50*'--'
                print('student')
                print('epoch:{} - Train loss: {:.3f}    '.format(epoch + 1, loss_log['ave_loss'] / batch))
                print(
                    'epoch:{} - student_acc_seen: {:.1f}          student_acc_novel: {:.1f}            student_H: {:.1f}'.format(
                        epoch + 1, student_acc_GZSL_seen, student_acc_GZSL_unseen, student_acc_GZSL_H))
                print('epoch:{} - Test student ZSL accuracy: {:.3f}   '.format(epoch + 1, student_acc_ZSL))

        print('\nBest_student_H [Unseen: {:.1f}% Seen: {:.1f}% H: {:.1f}% | Epoch-{}]'.format(
            result_gzsl_student.best_acc_U, result_gzsl_student.best_acc_S,
            result_gzsl_student.best_acc, result_gzsl_student.best_iter))
        print('\nBest_student__T1 [T1: {:.1f}%  | Epoch-{}]'.format(result_zsl_student.best_acc,
                                                                    result_zsl_student.best_iter))


if __name__ == '__main__':
    main()
