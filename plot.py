import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
from scipy.spatial.distance import cdist as distance
import warnings
import os.path as osp
from train import train
import argparse
import os
from models.eval import EvalInfo
warnings.filterwarnings('always')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='D2LAD', choices=['D2LAD', 'D2AwA'])
    parser.add_argument('--dataroot', type=str, default='/home/scott/Work/Dataset')
    parser.add_argument('--source', type=str, default='AwA2_10', choices=['Real_10/40', 'Painting_10/40', 'AwA2_10', 'LAD_40'])
    parser.add_argument('--target', type=str, default='Painting_17', choices=['Real_17/56', 'Painting_17/56', 'AwA2_17', 'LAD_56'])

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--outputdir', type=str, default='./exp')
    parser.add_argument('--expname', type=str, default='exp_test')

    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    # Model
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--finetune_backbone', type=bool, default=False)
    parser.add_argument('--test_interval', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--pretrain_epochs', type=int, default=3)
    parser.add_argument('--net', type=str, default='resnet50')
    parser.add_argument('--bottleneck', type=int, default=512, help="Gz feat dim")
    parser.add_argument('--ppnet_last_layer_type', type=str, default='sigmoid', help="[softmax | sigmoid]")

    parser.add_argument('--alpha', type=float, default= 0.001, help="mix centroids")

    parser.add_argument('--alpha1', type=float, default=1, help="L_clst^P")
    parser.add_argument('--alpha2', type=float, default=0.01, help="L_sep^P")
    parser.add_argument('--alpha3', type=float, default=1e-4, help="L_l1^P")

    parser.add_argument('--beta1', type=float, default=0.01, help="L^R")
    parser.add_argument('--beta2', type=float, default=1, help="L^A")
    parser.add_argument('--beta3', type=float, default=1, help="L^D")

    parser.add_argument('--hard_neg', default=False, action='store_true', help="Use hard negative in contrast")
    # 3. train
    parser.add_argument('--resume', type=str, default='path of saved checkpoints')
    parser.add_argument('--prototypes_K', type=int, default=3)
    parser.add_argument('--att_dim', type=int, default=85, help='[85 | 337] for AwA | LAD')
    parser.add_argument('--ppnet_warm_start', type=int, default=0)
    parser.add_argument('--ppnet_warm_num', type=int, default=5)

    # 4. ablation analysis
    parser.add_argument('--feat_fusion_gt', type=bool, default=True, help='If use gt att to fuse feature')
    parser.add_argument('--feat_fusion_pseudo', type=bool, default=True, help='If use pseudo att to fuse feature')
    parser.add_argument('--feat_fusion_pred', type=bool, default=True, help='If use pred att to fuse feature')
    parser.add_argument('--att_propagation', type=bool, default=True, help='If use visual guide att propagation')
    parser.add_argument('--prop_unk', default=False, action='store_true',
                        help="Label propagate to samples from unknown classes as target for self-training GA")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    ''' ============================================ '''
    # src = ['Real_10/40', 'Painting_10/40', 'AwA2_10', 'LAD_40']
    # tgt = ['Real_17/56', 'Painting_17/56', 'AwA2_17', 'LAD_56']
    args.dataset = 'D2AwA'
    args.source = 'AwA2_10'
    args.target = 'Painting_17'
    args.resume = './exp/AwA2_102Painting_17/exp_test/warmup'

    args.test_interval = 1
    args.pretrain_epochs = 3
    args.ppnet_warm_num = 10
    args.lr = 3e-3


    # Analysis
    args.prop_unk = False  # Proved! True harm performance; False is better
    args.finetune_backbone = True # Proved! True is better
    # proto fix_DC - doesn't influence much; set as False, which means update allowed
    args.feat_fusion_gt = False # Proved! not contribute much
    args.feat_fusion_pseudo = False # Proved! not contribute much
    args.hard_neg = False  # didn't notice much difference
    args.att_propagation = True # Observe: not much harm, sometimes False is better.
    args.feat_fusion_pred = True

    # protoModel
    args.alpha1 = 0.1 # cluster
    args.alpha2 = 0.01 # separate
    args.alpha3 = 1e-4 # L1
    # baseModel
    args.beta1 = 0.01 # partial align, LR
    args.beta2 = 1 # attributes, LA
    args.beta3 = 0 # to see if Dis is needed

    args.ppnet_last_layer_type = 'sigmoid' #'softmax'
    ''' -------------------------------------------- '''

    args.save_folder = osp.join(args.outputdir, args.source + '2' + args.target, args.expname)
    if not os.path.exists(args.save_folder):
        os.system('mkdir -p ' + args.save_folder)

    if args.dataset == 'D2AwA':
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_17.npy')
        args.shared_class_num = 10
        args.total_class_num = 17
        args.att_dim = 85

        s = args.source.split('_')[0]
        t = args.target.split('_')[0]
        s_dtset = 'AwA2' if s == 'AwA2' else 'DomainNet'
        t_dtset = 'AwA2' if t == 'AwA2' else 'DomainNet'

        args.source_path = osp.join(args.datadir, s_dtset, s+'_src_0-9.txt')
        args.target_path = osp.join(args.datadir, t_dtset, t+'_tgt_0-16.txt')

    elif args.dataset == 'D2LAD':
        args.attri_path = osp.join(args.datadir, 'LAD', 'att_56_bi.npy')
        args.shared_class_num = 40
        args.total_class_num = 56
        args.att_dim = 337

        s = args.source.split('_')[0]
        t = args.target.split('_')[0]
        s_dtset = 'LAD' if s == 'LAD' else 'Domainnet'
        t_dtset = 'LAD' if t == 'LAD' else 'Domainnet'
        args.source_path = osp.join(args.datadir, s_dtset, s+'_src_0-39.txt')
        args.target_path = osp.join(args.datadir, t_dtset, t+'_tgt_0-55.txt')

    elif args.dataset == '3D2':
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 40
        args.total_class_num = 50
        args.att_dim = 85

    else:
        raise Exception("Given dataset is not correct!")

    eval_info = EvalInfo(args)
    checkpoint_path = './exp/AwA2_102Painting_17/sigmoid_without_dis/'
    eval_info.base_info = np.load(checkpoint_path+'base_info.npy', allow_pickle=True).item()
    eval_info.prot_info = np.load(checkpoint_path+'prot_info.npy', allow_pickle=True).item()
    eval_info.init_info = np.load(checkpoint_path+'init_info.npy', allow_pickle=True).item()
    eval_info.print_info(epoch=eval_info.base_info['epochs'][-1])
    eval_info.plot(epoch=eval_info.base_info['epochs'][-1])