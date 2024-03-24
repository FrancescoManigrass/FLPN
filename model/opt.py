import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AWA2', help='FLO, CUB')
    parser.add_argument('--optimizer', default='adam', help='adam, sgd')
    parser.add_argument('--model', help='resnet, vit')
    parser.add_argument('--root', default='data/wgr/code/APN-ZSL-master/', help='path to project')
    parser.add_argument('--image_root', default='data/wgr/code/APN-ZSL-master/data', type=str, metavar='PATH',
                        help='path to image root')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--k', type=int, default=15, help='k for random fake')
    parser.add_argument('--nepoch', type=int, default=30000, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='start_epoch')
    parser.add_argument('--init_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--classifier_lr', type=float, default=1e-6, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed 3483')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--regular', type=float, default=0.000005)
    # parser.add_argument('--imagelist', default=CC_HOME + '/ZSL_REG/data/CUB/cub_imagelist.txt', type=str,
    #                     metavar='PATH',
    #                     help='path to imagelist (default: none)')
    parser.add_argument('--resnet_path', default=None,
                        # resnet101_cub.pth.tar resnet101-5d3b4d8f.pth
                        help="path to pretrain resnet classifier")

    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--pretrained', default=None, help="path to pretrain classifier (to continue training)")
    # parser.add_argument('--checkpointroot', default=CC_HOME + '/ZSL_REG/checkpoint', help='path to checkpoint')
    parser.add_argument('--image_type', default='test_unseen_loc', type=str, metavar='PATH',
                        help='image_type to visualize, usually test_unseen_small_loc, test_unseen_loc, test_seen_loc')
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--scale', type=float, default=25.0)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate to pretrain model')
    parser.add_argument('--all', action='store_true', default=False)

    parser.add_argument('--gzsl', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default="0",  help='used gpu')
    parser.add_argument('--p', type=float, default="1", help='p euclidean distance')

    parser.add_argument('--additional_loss', action='store_true', default=True)
    parser.add_argument('--sat_agg_class_outlier_exists', action='store_true', default=False)
    parser.add_argument('--sat_agg_class', action='store_true', default=True)
    parser.add_argument('--sat_agg_class_global', action='store_true', default=False)
    parser.add_argument('--sat_euclidean_distance', action='store_true', default=False)
    parser.add_argument('--sat_agg_same_attribute', action='store_true', default=False)
    parser.add_argument('--sat_agg_macroclass_implied', action='store_true', default=False)
    parser.add_argument('--sat_agg_class_cluster_greater', action='store_true', default=False)
    parser.add_argument('--sat_agg_class_cluster_lower', action='store_true', default=False)

    parser.add_argument('--calibrated_stacking', type=float, default=False,
                        help='calibrated_stacking, shrinking the output score of seen classes')

    parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')

    # about calculating IoU
    parser.add_argument('--save_att', default=False, help='./visualize_IoU/00/')
    parser.add_argument('--IoU_scale', type=int, default=4)  # The scale of IoU bounding box
    parser.add_argument('--IoU_thr', type=float, default=0.5)
    parser.add_argument('--resize_WH', action='store_true', default=False)
    parser.add_argument('--out_of_edge', action='store_true', default=False)
    parser.add_argument('--max_area_center', action='store_true', default=False)
    parser.add_argument('--KNOW_BIRD_BB', action='store_true', default=False)

    # for distributed loader
    parser.add_argument('--train_mode', type=str, default='random', help='loader: random or distributed or ltnsampler')
    parser.add_argument('--n_batch', type=int, default=1000, help='batch numbers per epoch')
    parser.add_argument('--false_negative', type=int, default=15, help='batch numbers per epoch')
    parser.add_argument('--ways', type=int, default=16, help='class numbers per episode')
    parser.add_argument('--shots', type=int, default=2, help='image numbers per class')

    parser.add_argument('--transform_complex', action='store_true', default=False, help='complex transform')
    # additional for SUN and AWA
    parser.add_argument('--awa_finetune', action='store_true', default=False)
    parser.add_argument('--use_group', action='store_true', default=False)
    parser.add_argument('--avg_pool', action='store_true', default=False)
    parser.add_argument('--avg_pool_part', action='store_true', default=False)

    parser.add_argument('--l_regular', type=float, default=0)
    # evaluation
    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path',
                        default='C:\\Users\\lab2O\\Documents\\Francesco Manigrasso\\polito\\flvn_official_extended\\out\\weight\\alpha\\pretrainepoch4_pretrainlr0.0001_classifierlr1e-07_8ways-12shots-1/Experiment(NEW-1298)_AWA2_student_GZSL_id_0.pth',
                        help='weights resume missed')

    parser.add_argument('--neptune_flag', action='store_true', default=False)
    parser.add_argument('--print_images', action='store_true', default=False)

    parser.add_argument('--cropped_image', action='store_true', default=False)

    parser.add_argument('--contrative_loss_weight', type=int, default=1)
    parser.add_argument('--cpt', type=float, default=0)

    # axioms_opt
    parser.add_argument('--axioms_exists', action='store_true', default=False)
    parser.add_argument('--macroclass_exists', action='store_true', default=False)
    parser.add_argument('--weight_exists', action='store_true', default=False)
    parser.add_argument('--weight_isofclass', action='store_true', default=False)

    parser.add_argument('--accelerator', action='store_true', default=False)

    # opt for finetune ALE
    opt = parser.parse_args()
    opt.dataroot = opt.root + 'data'
    opt.checkpointroot = opt.root + 'checkpoint'
    opt.project_name = "frankissimo/newprotoltn"
    opt.token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5YTY1M2U5Ni05ZTU0LTQ0YjAtYWM0OC1jNzUyZTIwOWNiNDQifQ=="
    # print('opt:', opt)

    return opt
