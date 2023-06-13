import os.path as osp
from train import train
from opts import get_opts
import os

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    ''' ============================================ '''
    args.test_interval = 1
    args.pretrain_epochs = 3
    args.ppnet_warm_num = 5

    # Analysis
    args.prop_unk = False  # Proved! True harm performance; False is better
    args.feat_fusion_gt = False # Proved! not contribute much
    args.feat_fusion_pseudo = False # Proved! not contribute much
    args.hard_neg = False  # didn't notice much difference
    args.att_propagation = True # Observe: not much harm, sometimes False is better.
    args.feat_fusion_pred = True
    args.feat_fusion_separate = False # both branches use visual feat_s, (z_s/t)

    args.ppnet_last_layer_type = 'sigmoid'
    args.ppnet_dist_type = 'cosine' if args.ppnet_last_layer_type == 'sigmoid' else 'euclidean'
    args.prototype_activation_function = 'log' if args.ppnet_dist_type == 'euclidean' else 'linear'
    ''' -------------------------------------------- '''

    print(print_args(args))

    args.save_folder = osp.join(args.outputdir, args.source + '2' + args.target, args.expname)
    param_list = 'al1={}_al2={}_be1={}_be2={}_m={}'.format(
        args.alpha1, args.alpha2, args.beta1, args.beta2, args.prototypes_K
    )
    args.save_folder = args.save_folder + '/' + param_list

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
        s_dtset = 'LAD' if s == 'LAD' else 'DomainNet'
        t_dtset = 'LAD' if t == 'LAD' else 'DomainNet'
        args.source_path = osp.join(args.datadir, s_dtset, s+'_src_0-39.txt')
        args.target_path = osp.join(args.datadir, t_dtset, t+'_tgt_0-55.txt')

    elif args.dataset == '3D2':
        args.attri_path = osp.join(args.datadir, 'AwA2', 'att_50.npy')
        args.shared_class_num = 40
        args.total_class_num = 50
        args.att_dim = 85

    else:
        raise Exception("Given dataset is not correct!")

    train(args)


if __name__ == '__main__':
    # # # D2AwA
    # # src = ['Real_10', 'Painting_10', 'AwA2_10']
    # # tgt = ['Real_17', 'Painting_17', 'AwA2_17']
    #
    # # D2A
    # args = get_opts()
    # args.dataset = 'D2AwA'
    # args.source = 'AwA2_10'
    # args.target = 'Real_17'
    # args.finetune_backbone = True
    # args.batchsize = 64 # 64 if finetune backbone
    # args.resume = ''
    # args.lr = 3e-3  # 3e-3 #1e-3
    # args.ppnet_last_layer_type = 'sigmoid'  # 'softmax'
    # args.beta1 = 0.1

    # -------------------------------
    # # D2LAD
    # src = ['Real_40', 'Painting_40', 'LAD_40']
    # tgt = ['Real_56', 'Painting_56', 'LAD_56']
    args = get_opts()
    args.dataset = 'D2LAD'
    domains = ['Real', 'Painting', 'LAD']
    i = 1
    j = 2
    args.source = domains[i] + '_40'
    args.target = domains[j] + '_56'
    args.resume = '' #'./checkpoints/'+args.source+'2'+args.target+'/osda_best_baseModel/'
    args.finetune_backbone = True
    args.batchsize = 64
    args.lr = 3e-3  # 3e-3 #1e-3
    args.ppnet_last_layer_type = 'sigmoid'  # 'softmax'
    args.beta1 = 0.1
    # args.beta3 = 1.0
    # --------------------------------

    main(args)
    #

    # # Parameter analysis
    # # alpha1, beta1, beta2 -> {0, 0.01, 0.05, 0.2, 0.5, 1} # no need 0.1
    # # alpha2 -> {0, 1e-4, 1e-2, 1e-1, 1} # no need 1e-3
    # # mk -> {1, 2, 5, 10} # no need 3
    # # ToDo: al1, al2, be1, be2, mk, (optional: lambda)
    # # Default: 0.1, 0.001, 0.1, 0.1, 3, (0.001)
    # # Finished: alpha1,
    # for v in [0, 1e-4, 1e-2, 1e-1, 1]:
    #     args = get_opts()
    #     # protoModel
    #     # wclst = 1 x beta2, cluster (in paper: beta2=0.1, w_clst=beta2*0.1)
    #     # wsep = 0.1 * beta2, alpha2=0.1
    #     args.alpha1 = 0.1 #0.1 # separate (in paper: beta2=0.1, w_sep=alpha1*beta2=0.1*0.1=0.01
    #     args.alpha2 = v #1e-3 # L1 (in paper: beta2=0.1, w_l1=alpha2*beta2=1e-3*0.1=1e-4)
    #     # baseModel
    #     args.beta1 = 0.1 # LR
    #     args.beta2 = 0.1
    #     args.beta3 = 0.# set 0 to  remove L_dis # to see if Dis is needed # not need anymore
    #     args.prototypes_K = 3
    #     # D2A
    #     args.dataset = 'D2AwA'
    #     args.source = 'Real_10'
    #     args.target = 'AwA2_17'
    #     args.finetune_backbone = True
    #     args.batchsize = 64 # 64 if finetune backbone
    #     args.resume = ''
    #     args.lr = 3e-3  # 3e-3 #1e-3
    #     args.ppnet_last_layer_type = 'sigmoid'  # 'softmax'
    #
    #     main(args)