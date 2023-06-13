import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='D2LAD', choices=['D2LAD', 'D2AwA'])
    parser.add_argument('--dataroot', type=str, default='/home/scott/Work/Dataset')
    parser.add_argument('--source', type=str, default='AwA2_10', choices=['Real_10/40', 'Painting_10/40', 'AwA2_10', 'LAD_40'])
    parser.add_argument('--target', type=str, default='Painting_17', choices=['Real_17/56', 'Painting_17/56', 'AwA2_17', 'LAD_56'])

    parser.add_argument('--datadir', type=str, default='./data')
    parser.add_argument('--outputdir', type=str, default='./checkpoints')
    parser.add_argument('--expname', type=str, default='')

    parser.add_argument('--batchsize', type=int, default=256)
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
    parser.add_argument('--ppnet_dist_type', type=str, default='cosine', help='[cosine | euclidean] for [sigmoid | softmax]')
    parser.add_argument('--prototype_activation_function', type=str, default='log',
                        help='euclidea: log | cosine: 1-distance (named as: linear)')
    parser.add_argument('--add_on_layers_type', type=str, default='regular',
                        help='regular: 2 layer add-on, relu hidden, output no activation (original PropNet has Sigmoid())')
    parser.add_argument('--feat_fusion_separate', type=bool, default=True, help='[True | False] - if fuse feat-att and att propagation based on each branch own feat, or all based on baseModel z')


    parser.add_argument('--lambd', type=float, default= 0.001, help="mix centroids")
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
    return args