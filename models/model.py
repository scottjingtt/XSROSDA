import torch
import os
import numpy as np
import models.network as network
import models.propagation as propagation
from models.utils import AverageMeter
import torch.nn.functional as F
import torch.nn as nn
from sklearn import preprocessing
from sklearn.cluster import KMeans
import torch.optim as optim
import os.path as osp
import copy
from models.eval import EvalInfo


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        if 'lr0' in param_group:
            param_group['lr'] = param_group['lr0'] * decay

        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class XSROSDA:
    def __init__(self, args, s_loader_len, t_loader_len):
        self.args = args

        # 1. Base model
        # backbone, output both feature and feature-map
        self.netF = network.ResBase(res_name=args.net).cuda()
        # G_Z: resnet(2048) --> bottlenet(512), two layer
        self.netB = network.BaseFeature(output_dim=args.bottleneck, input_dim=self.netF.in_features, hidden_dim=1024).cuda()

        self.net = nn.Sequential(self.netF, self.netB)

        # G_A
        self.netA = network.AttributeProjector(in_feat=args.bottleneck, hidden_feat=256, out_feat=args.att_dim).cuda()

        self.fused_feat_dim = args.bottleneck+self.args.att_dim
        # C: Z(512+85/337) --> open-set(Ks+1), one layer
        self.netC_all = network.Classifier(class_num=self.args.shared_class_num+1, input_dim=self.fused_feat_dim, hidden_dim=256).cuda()

        # D: (512+85/337) --> 2
        self.netD = network.Classifier(class_num=2, input_dim=self.fused_feat_dim, hidden_dim=256).cuda()

        # Attributes propagation, visual graph guidance
        self.AttributeProp = propagation.AttributePropagation()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # for protoModel, conv_features
        self.maxpool1d = nn.MaxPool1d(kernel_size=self.args.prototypes_K, stride=self.args.prototypes_K, padding=0)

        self.optimizer = None
        self.lr_scheduler = None

        # 2. ProtoModel - PPNet
        ''' PPNet '''
        self.ppnet = network.PPNet(img_size=7,
                 prototype_shape=(args.prototypes_K*self.args.att_dim, 512, 1, 1),
                 num_classes=self.args.att_dim,
                 init_weights=True,
                 prototype_activation_function=args.prototype_activation_function,
                 dist_type=args.ppnet_dist_type,
                 add_on_layers_type=args.add_on_layers_type,
                 take_best_prototypes=True,
                 args=args).cuda()

        self.protModel_C = network.Classifier(class_num=self.args.shared_class_num+1, input_dim=self.fused_feat_dim, hidden_dim=256).cuda()

        # D: (512+85/337) --> 2
        self.protModel_D = network.Classifier(class_num=2, input_dim=self.fused_feat_dim, hidden_dim=256).cuda()


        self.optimizer = None # change during training
        self.scheduler = None # change during training
        self.base_optimizer = None
        self.base_lr_scheduler = None
        self.proto_optimizer = None
        self.proto_lr_scheduler = None
        self.proto_warm_optimizer = None
        self.proto_warm_lr_scheduler = None
        self.proto_ll_optimizer = None
        self.proto_ll_lr_scheduler = None

        # 3. Data and records
        # The the number of batches of the smaller dataset between source/target as the max iteration number
        # self.max_iter = args.epochs * min(s_loader_len, t_loader_len)
        # self.iter_num = 0

        self.estimate_meter = AverageMeter(self.args.shared_class_num + 1)
        self.estimate_refine_meter = AverageMeter(self.args.shared_class_num + 1)

        self.eval_info = EvalInfo(args)

    # (0) Initialize/ Estimate Pseudo labels

    def k_means_clustering(self, all_feat, init_centroids=None):
        """
        :param all_feat: torch, cuda
        :param init_centroids: torch, cuda
        :return:np
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        tgt_pseudo_lb = []
        tgt_prob = []

        for i in range(all_feat.size(0)):
            cur_feat = all_feat[i].repeat(init_centroids.size(0), 1)  # [N_cls, d]
            cos_dist = 0.5 * (1 - cos(cur_feat, init_centroids))
            prob = F.softmax(-1.0 * cos_dist, dim=0)
            tgt_pseudo_lb.append(prob.argmax().item())  # int
            tgt_prob.append(prob.max().item())

        tgt_prob = np.array(tgt_prob) # only max probabilities returned
        tgt_pseudo_lb = np.array(tgt_pseudo_lb, dtype=int) # only idx of max probabilities returned

        return tgt_prob, tgt_pseudo_lb

    def get_centroids(self, all_feat, all_lb, num_cls, empty_fill=None):
        """
        :param all_feat: np
        :param all_lb: np
        :param num_cls: int
        :return: np
        """
        cls_feat_dict = {}
        for i in range(num_cls):
            cls_feat_dict[i] = []

        for i in range(all_feat.shape[0]):
            cls_feat_dict[all_lb[i]].append(all_feat[i].reshape(1,-1))

        centroids = np.zeros((num_cls, all_feat.shape[1]))

        for i in range(num_cls):
            if len(cls_feat_dict[i]) > 0:
                centroids[i] = np.concatenate(cls_feat_dict[i], axis=0).mean(axis=0)
            else:
                centroids[i] = empty_fill[i]

        return centroids

    @torch.no_grad()
    def initialize_pseudo_label(self, source_loader, target_loader):
        print('Estimating Pseudo Labels...')
        self.estimate_meter.reset()
        self.estimate_refine_meter.reset()

        self.netF.eval()
        self.netB.eval()

        source_cls_feat = {}
        for i in range(self.args.shared_class_num):
            source_cls_feat[i] = []

        all_src_feat = []
        all_src_gt_lb = []
        for batch_idx, batch in enumerate(source_loader):
            img = batch[0].cuda()
            lb = batch[1]

            res_feat, res_featmap = self.netF(img)
            # feat = self.netB(res_feat)  # [bs, 256]
            feat = res_feat  # (bs, 2048)

            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_src_feat.append(feat)
            all_src_gt_lb.append(lb)

            for i in range(lb.size()[0]):
                source_cls_feat[lb[i].item()].append(feat[i].view(1, -1))

        source_cls_centroids = torch.zeros((self.args.shared_class_num, 2048)).cuda()  # [shared_cls_n, 256]
        for k in source_cls_feat.keys():
            source_cls_centroids[k] = torch.cat(source_cls_feat[k], dim=0).mean(dim=0)  # [256,]

        all_tgt_feat = []
        all_tgt_gt_lb = []
        for batch_idx, batch in enumerate(target_loader):
            img = batch[0].cuda()
            lb = batch[1]

            res_feat, res_featmap = self.netF(img)
            # feat = self.netB(res_feat)  # [bs, 256]
            feat = res_feat
            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_tgt_feat.append(feat)
            all_tgt_gt_lb.append(lb)

        all_tgt_feat = torch.cat(all_tgt_feat, dim=0)
        all_tgt_gt_lb = torch.cat(all_tgt_gt_lb, dim=0).numpy()

        # Inital clustering
        tgt_prob, tgt_pseudo_lb = self.k_means_clustering(all_tgt_feat, init_centroids=source_cls_centroids)

        # thresh = tgt_prob.mean()
        # tgt_pseudo_lb[tgt_prob<thresh] = self.args.shared_class_num  # modify as Unk

        # Reject class-wise
        for i in range(self.args.shared_class_num):
            cur_cls_indice = (tgt_pseudo_lb == i)
            cur_cls_prob_mean = tgt_prob[cur_cls_indice].mean()

            cur_cls_rej_id = (tgt_prob[cur_cls_indice] < cur_cls_prob_mean)

            tgt_pseudo_lb[np.where(cur_cls_indice == True)[0][cur_cls_rej_id]] = self.args.shared_class_num

        ### Initial clustering result
        self.estimate_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb == tgt_pseudo_lb).astype(int))

        target_cls_feat = {}
        for i in range(self.args.shared_class_num + 1):
            target_cls_feat[i] = []

        for i in range(all_tgt_feat.size(0)):
            target_cls_feat[tgt_pseudo_lb[i]].append(all_tgt_feat[i].view(1, -1))

        target_knw_centroids = torch.zeros((self.args.shared_class_num, 2048)).cuda()  # [shared_cls_n, 256]
        for i in range(self.args.shared_class_num):
            if target_cls_feat[i] == []:
                target_cls_feat[i].append(source_cls_centroids[i].view(1, -1))

            target_knw_centroids[i] = torch.cat(target_cls_feat[i], dim=0).mean(dim=0)

        # Adjust shared class centroids
        # [shared_cls_n, 256]
        shared_cls_centroids = (1 - self.args.lambd) * source_cls_centroids + self.args.lambd * target_knw_centroids
        shared_cls_centroids = shared_cls_centroids.cpu().numpy()

        # [K, 256]: Clustered K Centroids on T UNK
        tgt_unk_feat = torch.cat(target_cls_feat[self.args.shared_class_num], dim=0).cpu().numpy()
        tgt_unk_feat_norm = preprocessing.normalize(tgt_unk_feat, axis=1)
        # unk_cls_centroids = KMeans(n_clusters=self.args.total_class_num-self.args.shared_class_num).fit(tgt_unk_feat).cluster_centers_
        unk_cls_lb = KMeans(n_clusters=self.args.total_class_num - self.args.shared_class_num).fit(
            tgt_unk_feat_norm).labels_  # 0-(K-1)

        unk_cls_feat_dict = {}
        for i in range(self.args.total_class_num - self.args.shared_class_num):
            unk_cls_feat_dict[i] = []

        for i in range(tgt_unk_feat.shape[0]):
            unk_cls_feat_dict[unk_cls_lb[i]].append(tgt_unk_feat[i].reshape(1, -1))

        unk_cls_centroids = np.zeros((self.args.total_class_num - self.args.shared_class_num, tgt_unk_feat.shape[1]),
                                     dtype=float)
        for i in range(unk_cls_centroids.shape[0]):
            unk_cls_centroids[i] = np.concatenate(unk_cls_feat_dict[i], axis=0).mean(axis=0)

        unk_cls_lb += self.args.shared_class_num  # shared_cls_n -- (shared_cls_n+K-1)

        # Second-time clustering refinement
        all_cls_centroids = np.concatenate([shared_cls_centroids, unk_cls_centroids], axis=0)
        all_tgt_lb = copy.deepcopy(tgt_pseudo_lb)
        all_tgt_lb[np.where(all_tgt_lb == self.args.shared_class_num)[0]] = unk_cls_lb

        _, refined_lb = self.k_means_clustering(all_tgt_feat,
                                                init_centroids=torch.tensor(all_cls_centroids).float().cuda())
        refined_unk_cls_centroids = self.get_centroids(all_tgt_feat.cpu().numpy(), refined_lb,
                                                       self.args.total_class_num, all_cls_centroids)  # [shared_n+K, d]

        refined_centroids = copy.deepcopy(refined_unk_cls_centroids)
        refined_centroids[:self.args.shared_class_num] = (1 - self.args.lambd) * source_cls_centroids.cpu().numpy() + \
                                                         self.args.lambd * refined_centroids[
                                                                           :self.args.shared_class_num]

        all_tgt_refined_lb = copy.deepcopy(refined_lb)
        all_tgt_refined_lb[all_tgt_refined_lb >= self.args.shared_class_num] = self.args.shared_class_num
        self.estimate_refine_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb == all_tgt_refined_lb).astype(int))

        print('Estimation Done!')

        ### Pre-refinement performance
        os_star = self.estimate_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)

        os = (os_star * self.args.shared_class_num + unk_acc) / (self.args.shared_class_num + 1)
        print('Pre-refinement >> OS*: {:.2%} | UNK: {:.2%} | H: {:.2%} | OS: {:.2%}'.format(os_star, unk_acc, h_score,
                                                                                            os))
        print('Acc: ', end='')
        for k in range(self.estimate_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_meter.avg[k]), end='')
        print('\n')

        ### Post-refinement performance
        os_star = self.estimate_refine_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_refine_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)
        os = (os_star * self.args.shared_class_num + unk_acc) / (self.args.shared_class_num + 1)
        print('Post-refinement >> OS*: {:.2%} | UNK: {:.2%} | H: {:.2%} | OS: {:.2%}'.format(os_star, unk_acc, h_score,
                                                                                             os))
        print('Acc: ', end='')
        for k in range(self.estimate_refine_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_refine_meter.avg[k]), end='')
        print('\n')

        # self.cluster_acc_logger.add_record([os_star * 100, unk_acc * 100, h_score * 100, os * 100])

        return torch.tensor(refined_centroids).float(), refined_lb.tolist()

    @torch.no_grad()
    def estimate_pseudo_label(self, source_loader, target_loader):
        print('Estimating Pseudo Labels...')
        self.estimate_meter.reset()
        self.estimate_refine_meter.reset()

        self.netF.eval()
        self.netB.eval()

        source_cls_feat = {}
        for i in range(self.args.shared_class_num):
            source_cls_feat[i] = []

        all_src_feat = []
        all_src_gt_lb = []
        for batch_idx, batch in enumerate(source_loader):
            img = batch[0].cuda()
            lb = batch[1]

            res_feat, res_featmap = self.netF(img)

            feat = self.netB(res_feat)  # [bs, 256]
            # feat = res_feat # (bs, 2048)

            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_src_feat.append(feat)
            all_src_gt_lb.append(lb)

            for i in range(lb.size()[0]):
                source_cls_feat[lb[i].item()].append(feat[i].view(1,-1))


        source_cls_centroids = torch.zeros((self.args.shared_class_num, self.args.bottleneck)).cuda()  # [shared_cls_n, 256]
        for k in source_cls_feat.keys():
            source_cls_centroids[k] = torch.cat(source_cls_feat[k], dim=0).mean(dim=0)  # [256,]

        all_tgt_feat = []
        all_tgt_gt_lb = []
        for batch_idx, batch in enumerate(target_loader):
            img = batch[0].cuda()
            lb = batch[1]

            res_feat, res_featmap = self.netF(img)
            feat = self.netB(res_feat)  # [bs, 256]
            # feat = res_feat
            if feat.dim() == 1:
                feat = feat.view(1, -1)

            all_tgt_feat.append(feat)
            all_tgt_gt_lb.append(lb)

        all_tgt_feat = torch.cat(all_tgt_feat, dim=0)
        all_tgt_gt_lb = torch.cat(all_tgt_gt_lb, dim=0).numpy()

        # Inital clustering
        tgt_prob, tgt_pseudo_lb = self.k_means_clustering(all_tgt_feat, init_centroids=source_cls_centroids)

        #thresh = tgt_prob.mean()
        #tgt_pseudo_lb[tgt_prob<thresh] = self.args.shared_class_num  # modify as Unk

        # Reject class-wise
        for i in range(self.args.shared_class_num):
            cur_cls_indice = (tgt_pseudo_lb == i)
            cur_cls_prob_mean = tgt_prob[cur_cls_indice].mean()

            cur_cls_rej_id = (tgt_prob[cur_cls_indice] < cur_cls_prob_mean)

            tgt_pseudo_lb[np.where(cur_cls_indice==True)[0][cur_cls_rej_id]] = self.args.shared_class_num


        ### Initial clustering result
        self.estimate_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb==tgt_pseudo_lb).astype(int))


        target_cls_feat = {}
        for i in range(self.args.shared_class_num+1):
            target_cls_feat[i] = []

        for i in range(all_tgt_feat.size(0)):
            target_cls_feat[tgt_pseudo_lb[i]].append(all_tgt_feat[i].view(1,-1))

        target_knw_centroids = torch.zeros((self.args.shared_class_num, self.args.bottleneck)).cuda()  # [shared_cls_n, 256]
        for i in range(self.args.shared_class_num):
            if target_cls_feat[i] == []:
                target_cls_feat[i].append(source_cls_centroids[i].view(1,-1))

            target_knw_centroids[i] = torch.cat(target_cls_feat[i], dim=0).mean(dim=0)


        # Adjust shared class centroids
        # [shared_cls_n, 256]
        shared_cls_centroids = (1-self.args.lambd) * source_cls_centroids + self.args.lambd * target_knw_centroids
        shared_cls_centroids = shared_cls_centroids.cpu().numpy()

        # [K, 256]: Clustered K Centroids on T UNK
        tgt_unk_feat = torch.cat(target_cls_feat[self.args.shared_class_num], dim=0).cpu().numpy()
        tgt_unk_feat_norm = preprocessing.normalize(tgt_unk_feat, axis=1)
        #unk_cls_centroids = KMeans(n_clusters=self.args.total_class_num-self.args.shared_class_num).fit(tgt_unk_feat).cluster_centers_
        unk_cls_lb = KMeans(n_clusters=self.args.total_class_num-self.args.shared_class_num).fit(tgt_unk_feat_norm).labels_  # 0-(K-1)

        unk_cls_feat_dict = {}
        for i in range(self.args.total_class_num-self.args.shared_class_num):
            unk_cls_feat_dict[i] = []

        for i in range(tgt_unk_feat.shape[0]):
            unk_cls_feat_dict[unk_cls_lb[i]].append(tgt_unk_feat[i].reshape(1,-1))

        unk_cls_centroids = np.zeros((self.args.total_class_num-self.args.shared_class_num, tgt_unk_feat.shape[1]), dtype=float)
        for i in range(unk_cls_centroids.shape[0]):
            unk_cls_centroids[i] = np.concatenate(unk_cls_feat_dict[i], axis=0).mean(axis=0)

        unk_cls_lb += self.args.shared_class_num  # shared_cls_n -- (shared_cls_n+K-1)

        # Second-time clustering refinement
        all_cls_centroids = np.concatenate([shared_cls_centroids, unk_cls_centroids], axis=0)
        all_tgt_lb = copy.deepcopy(tgt_pseudo_lb)
        all_tgt_lb[np.where(all_tgt_lb == self.args.shared_class_num)[0]] = unk_cls_lb

        _, refined_lb = self.k_means_clustering(all_tgt_feat, init_centroids=torch.tensor(all_cls_centroids).float().cuda())
        refined_unk_cls_centroids = self.get_centroids(all_tgt_feat.cpu().numpy(), refined_lb,
                                                       self.args.total_class_num, all_cls_centroids)  # [shared_n+K, d]

        refined_centroids = copy.deepcopy(refined_unk_cls_centroids)
        refined_centroids[:self.args.shared_class_num] = (1-self.args.lambd) * source_cls_centroids.cpu().numpy() + \
                                                         self.args.lambd * refined_centroids[:self.args.shared_class_num]

        all_tgt_refined_lb = copy.deepcopy(refined_lb)
        all_tgt_refined_lb[all_tgt_refined_lb >= self.args.shared_class_num] = self.args.shared_class_num
        self.estimate_refine_meter.update(all_tgt_gt_lb, (all_tgt_gt_lb == all_tgt_refined_lb).astype(int))

        print('Estimation Done!')

        ### Pre-refinement performance
        os_star = self.estimate_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)

        os = self.estimate_meter.avg.mean()
        print('Pre-refinement >> OS*: {:.2%} | OS^: {:.2%} | H: {:.2%} | OS: {:.2%}'.format(os_star,unk_acc,h_score, os))
        print('Acc: ', end='')
        for k in range(self.estimate_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_meter.avg[k]), end='')
        print('\n')


        ### Post-refinement performance
        os_star = self.estimate_refine_meter.avg[:self.args.shared_class_num].mean()
        unk_acc = self.estimate_refine_meter.avg[self.args.shared_class_num]
        h_score = 2 * os_star * unk_acc / (os_star + unk_acc)
        os = self.estimate_refine_meter.avg.mean()
        print('Post-refinement >> OS*: {:.2%} | OS^: {:.2%} | H: {:.2%} | OS: {:.2%}'.format(os_star, unk_acc, h_score, os))
        print('Acc: ', end='')
        for k in range(self.estimate_refine_meter.avg.shape[0]):
            print('{:.2%} '.format(self.estimate_refine_meter.avg[k]), end='')
        print('\n')

        return torch.tensor(refined_centroids).float(), refined_lb.tolist()


    # (1) Pre-train, refine pseudo labels
    def pre_train(self, epoch, source_loader, target_loader, centroids, attri_dict, mode='pre-train', args=None):
        self.netF.train()
        self.netB.train()

        self.netA.train()
        self.netC_all.train()
        self.netD.train()
        self.ppnet.train()

        total_batch = min(len(source_loader), len(target_loader))
        max_pretrain_iter = args.pretrain_epochs * total_batch

        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):

            self.optimizer.zero_grad()
            img_s = batch_s[0].cuda()
            lb_s = batch_s[1].cuda()  # 0--(shared_cls-1)

            img_t = batch_t[0].cuda()
            pseudo_lb_t = batch_t[3].cuda()  # 0--(shared_cls+K-1)
            # lb_t = batch_t[1].cuda()  # 0--shared_cls [1][2] are ground-truth labels

            s_res_feat, s_res_featmap = self.netF(img_s)
            t_res_feat, t_res_featmap = self.netF(img_t)
            feat_s = self.netB(s_res_feat)
            feat_t = self.netB(t_res_feat)

            # 1. BaseModel - optimize (GZ, GA, C, D)
            # 1.1 Partial alignment
            partial_align_loss_s = torch.tensor(0.).cuda()
            if centroids.shape[1] == self.args.bottleneck:
                for i in range(feat_s.size(0)):
                    L2 = torch.norm(feat_s[i] - centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[lb_s[i]]
                    if self.args.hard_neg:
                        # print("LR - hard_neg = True")
                        neg = torch.arange(0, L2.size(0)).cuda() != lb_s[i]
                        distract = -1 * L2[neg].min()
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)
                    partial_align_loss_s += attract + distract
                partial_align_loss_s /= feat_s.size(0)
            # 1.1.2 target
            partial_align_loss_t = torch.tensor(0.).cuda()
            if centroids.shape[1] == self.args.bottleneck:
                for i in range(feat_t.size(0)):
                    L2 = torch.norm(feat_t[i] - centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[pseudo_lb_t[i]]
                    if self.args.hard_neg:
                        neg = torch.arange(0, L2.size(0)).cuda() != pseudo_lb_t[i]
                        distract = -1 * L2[neg].min()  # original reproduce is 1.0
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)  # original reproduce is 1.0

                    partial_align_loss_t += attract + distract
                partial_align_loss_t /= feat_t.size(0)
            partial_align_loss = partial_align_loss_s + partial_align_loss_t

            # 1.2 Attributes propagation
            att_pred_loss = torch.tensor([0.]).cuda()
            att_pred_loss_knw = torch.tensor([0.]).cuda()
            att_pred_loss_unk = torch.tensor([0.]).cuda()

            all_feat = torch.cat([feat_s, feat_t], dim=0)
            all_lb = torch.cat([lb_s, pseudo_lb_t])
            knw_feat = all_feat[all_lb < self.args.shared_class_num]
            knw_lb = all_lb[all_lb < self.args.shared_class_num]

            all_pred_attri, _ = self.netA(all_feat)
            if self.args.att_propagation:
                prop_attri_all = self.AttributeProp(all_feat, all_pred_attri)
            else:
                prop_attri_all = all_pred_attri

            # 1.2.1 Known Attri loss - (The first ns rows, must be source data)
            # pred_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]
            if self.args.att_propagation:
                prop_attri_knw = prop_attri_all[all_lb < self.args.shared_class_num].clamp(min=0., max=1.)
            else:
                prop_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]

            knw_attri = torch.zeros((knw_lb.size(0), self.args.att_dim)).cuda()
            for i in range(knw_attri.size(0)):
                knw_attri[i] = attri_dict[knw_lb[i].item()]

            att_pred_loss_knw = F.binary_cross_entropy(prop_attri_knw.flatten(), knw_attri.flatten())

            att_pred_loss = att_pred_loss_knw #+ att_pred_loss_unk

            # 1.3 Visual-semantic joint classification
            # attri_s_pred, _ = self.netA(feat_s)  # [Ns, 85]
            attri_s_pred = prop_attri_all[:feat_s.size(0), :]

            cls_loss_s = torch.tensor([0.]).cuda()
            if self.args.feat_fusion_gt:
                feat_joint_s_gt = torch.zeros((feat_s.size(0), feat_s.size(1) + self.args.att_dim)).cuda()
                for i in range(feat_joint_s_gt.size(0)):
                    feat_joint_s_gt[i] = torch.cat([feat_s[i], attri_dict[lb_s[i].item()]])
                logits_s_gt = self.netC_all(feat_joint_s_gt)  # [Ns, K+1]
                cls_loss_s += F.cross_entropy(logits_s_gt, lb_s)
            if self.args.feat_fusion_pred:
                feat_joint_s_pred = torch.zeros((feat_s.size(0), feat_s.size(1) + self.args.att_dim)).cuda()
                for i in range(feat_joint_s_pred.size(0)):
                    feat_joint_s_pred[i] = torch.cat([feat_s[i], attri_s_pred[i]])
                logits_s_pred = self.netC_all(feat_joint_s_pred)
                cls_loss_s += F.cross_entropy(logits_s_pred, lb_s)

            # Knw T: feat + pseudo attri
            cls_loss_t = torch.tensor([0.]).cuda()
            if self.args.feat_fusion_pseudo:
                feat_joint_t_knw_pseudo = []
                lb_t_knw_pseudo = []
                for i in range(feat_t.size(0)):
                    if pseudo_lb_t[i] < self.args.shared_class_num:
                        feat_joint_t_knw_pseudo.append(
                            torch.cat([feat_t[i], attri_dict[pseudo_lb_t[i].item()]]).view(1, -1))
                        lb_t_knw_pseudo.append(pseudo_lb_t[i])

                feat_joint_t_knw_pseudo = torch.cat(feat_joint_t_knw_pseudo, dim=0)
                lb_t_knw_pseudo = torch.tensor(lb_t_knw_pseudo).cuda()
                logits_t_pseudo = self.netC_all(feat_joint_t_knw_pseudo)
                cls_loss_t += F.cross_entropy(logits_t_pseudo, lb_t_knw_pseudo)

            # All T: feat + pred attri
            if self.args.feat_fusion_pred:
                # attri_t_pred, _ = self.netA(feat_t)  # [Ns, 85]
                attri_t_pred = prop_attri_all[feat_s.size(0):, :]

                feat_joint_t_pred = torch.zeros((feat_t.size(0), feat_t.size(1) + self.args.att_dim)).cuda()

                for i in range(feat_joint_t_pred.size(0)):
                    feat_joint_t_pred[i] = torch.cat([feat_t[i], attri_t_pred[i]])

                logits_t_pred = self.netC_all(feat_joint_t_pred)  # [Nt, K+1]
                lb_t_all_pred = copy.deepcopy(pseudo_lb_t)  # if not deepcopy, modify lb_t_all_pred will change pseudo_lb_t
                lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = self.args.shared_class_num
                cls_loss_t += F.cross_entropy(logits_t_pred, lb_t_all_pred)

            cls_loss = cls_loss_s + cls_loss_t

            # 1.4 Visual-semantic known/unknown discrimination
            # Knw--1, Unk--0
            dis_loss = torch.tensor([0.]).cuda()
            # if self.args.beta3 != 0:
            #     if self.args.feat_fusion_pseudo:
            #         logits_t_knw_pseudo = self.netD(feat_joint_t_knw_pseudo)  # [Nts, 2]
            #         bi_lb_knw = torch.ones(logits_t_knw_pseudo.size(0), dtype=torch.int64).cuda()
            #         dis_loss += F.cross_entropy(logits_t_knw_pseudo, bi_lb_knw)
            #     if self.args.feat_fusion_pred:
            #         logits_t_all_pred = self.netD(feat_joint_t_pred)  # [Nt, 2]
            #         bi_lb_all = torch.zeros(logits_t_all_pred.size(0), dtype=torch.int64).cuda()  # 0 is unknown
            #         bi_lb_all[pseudo_lb_t < self.args.shared_class_num] = 1  # 1 means shared categories
            #         dis_loss += F.cross_entropy(logits_t_all_pred, bi_lb_all)

            L_S = cls_loss #+ self.args.beta3 * dis_loss
            L_R = partial_align_loss
            L_A = att_pred_loss
            loss_base = L_S + L_A + self.args.beta1 * L_R
            # (GZ, GA, C, D)

            loss = loss_base
            loss.backward()
            self.optimizer.step()

            iter_num = epoch * total_batch + (batch_idx + 1)
            # pre-train, max_epochs = self.args.pretrain_epochs
            lr_scheduler(self.optimizer, iter_num=iter_num, max_iter=max_pretrain_iter)

            if (batch_idx + 1) % self.args.print_freq == 0:
                print("Pre-train baseModel: Epoch {}/{} Batch {}/{} - LS({:.2}+{:.2})+{}*LR({:.2}+{:.2})+{}*LA({:.2}+{:.2})".format(
                    epoch, self.args.epochs, batch_idx+1, total_batch, cls_loss.item(), dis_loss.item(), self.args.beta1,
                    partial_align_loss_s.item(), partial_align_loss_t.item(), self.args.beta2, att_pred_loss_knw.item(), att_pred_loss_unk.item()))
        return

    def train_base(self, epoch, source_loader, target_loader, centroids, attri_dict, mode='base', args=None):
        self.netF.train()
        self.netB.train()

        self.netA.train()
        self.netC_all.train()
        self.netD.train()
        self.ppnet.train()
        print("Train baseModel: ")

        total_batch = min(len(source_loader), len(target_loader))
        max_iter = args.epochs * total_batch

        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):

            self.optimizer.zero_grad()
            img_s = batch_s[0].cuda()
            lb_s = batch_s[1].cuda()  # 0--(shared_cls-1)

            img_t = batch_t[0].cuda()
            pseudo_lb_t = batch_t[3].cuda()  # 0--(shared_cls+K-1)
            lb_t = batch_t[1].cuda()  # 0--shared_cls

            s_res_feat, s_res_featmap = self.netF(img_s)
            t_res_feat, t_res_featmap = self.netF(img_t)
            feat_s = self.netB(s_res_feat)
            feat_t = self.netB(t_res_feat)

            # 1. BaseModel - optimize (GZ, GA, C, D)
            # 1.1 Partial alignment
            partial_align_loss_s = torch.tensor(0.).cuda()
            if centroids.shape[1] == self.args.bottleneck:
                for i in range(feat_s.size(0)):
                    L2 = torch.norm(feat_s[i] - centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[lb_s[i]]
                    if self.args.hard_neg:
                        # print("LR - hard_neg = True")
                        neg = torch.arange(0, L2.size(0)).cuda() != lb_s[i]
                        distract = -1 * L2[neg].min()
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)
                    partial_align_loss_s += attract + distract
                partial_align_loss_s /= feat_s.size(0)
            # 1.1.2 target
            partial_align_loss_t = torch.tensor(0.).cuda()
            if centroids.shape[1] == self.args.bottleneck:
                for i in range(feat_t.size(0)):
                    L2 = torch.norm(feat_t[i] - centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[pseudo_lb_t[i]]
                    if self.args.hard_neg:
                        neg = torch.arange(0, L2.size(0)).cuda() != pseudo_lb_t[i]
                        distract = -1 * L2[neg].min()  # original reproduce is 1.0
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)  # original reproduce is 1.0

                    partial_align_loss_t += attract + distract
                partial_align_loss_t /= feat_t.size(0)
            partial_align_loss = partial_align_loss_s + partial_align_loss_t

            # 1.2 Attributes propagation
            att_pred_loss = torch.tensor([0.]).cuda()
            att_pred_loss_knw = torch.tensor([0.]).cuda()
            att_pred_loss_unk = torch.tensor([0.]).cuda()

            all_feat = torch.cat([feat_s, feat_t], dim=0)
            all_lb = torch.cat([lb_s, pseudo_lb_t])
            knw_feat = all_feat[all_lb < self.args.shared_class_num]
            knw_lb = all_lb[all_lb < self.args.shared_class_num]

            all_pred_attri, _ = self.netA(all_feat)
            if self.args.att_propagation:
                prop_attri_all = self.AttributeProp(all_feat, all_pred_attri)
            else:
                prop_attri_all = all_pred_attri

            # 1.2.1 Known Attri loss - (The first ns rows, must be source data)
            # pred_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]
            if self.args.att_propagation:
                prop_attri_knw = prop_attri_all[all_lb < self.args.shared_class_num].clamp(min=0., max=1.)
            else:
                prop_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]

            knw_attri = torch.zeros((knw_lb.size(0), self.args.att_dim)).cuda()
            for i in range(knw_attri.size(0)):
                knw_attri[i] = attri_dict[knw_lb[i].item()]

            att_pred_loss_knw = F.binary_cross_entropy(prop_attri_knw.flatten(), knw_attri.flatten()) \
                                + torch.mean(torch.norm(prop_attri_knw-knw_attri, p=2, dim=1))

            att_pred_loss = att_pred_loss_knw #+ att_pred_loss_unk

            # 1.3 Visual-semantic joint classification
            # attri_s_pred, _ = self.netA(feat_s)  # [Ns, 85]
            attri_s_pred = prop_attri_all[:feat_s.size(0), :]

            cls_loss_s = torch.tensor([0.]).cuda()
            if self.args.feat_fusion_gt:
                feat_joint_s_gt = torch.zeros((feat_s.size(0), feat_s.size(1) + self.args.att_dim)).cuda()
                for i in range(feat_joint_s_gt.size(0)):
                    feat_joint_s_gt[i] = torch.cat([feat_s[i], attri_dict[lb_s[i].item()]])
                logits_s_gt = self.netC_all(feat_joint_s_gt)  # [Ns, K+1]
                cls_loss_s += F.cross_entropy(logits_s_gt, lb_s)
            if self.args.feat_fusion_pred:
                feat_joint_s_pred = torch.zeros((feat_s.size(0), feat_s.size(1) + self.args.att_dim)).cuda()
                for i in range(feat_joint_s_pred.size(0)):
                    feat_joint_s_pred[i] = torch.cat([feat_s[i], attri_s_pred[i]])
                logits_s_pred = self.netC_all(feat_joint_s_pred)
                cls_loss_s += F.cross_entropy(logits_s_pred, lb_s)

            # Knw T: feat + pseudo attri
            cls_loss_t = torch.tensor([0.]).cuda()
            if self.args.feat_fusion_pseudo:
                feat_joint_t_knw_pseudo = []
                lb_t_knw_pseudo = []
                for i in range(feat_t.size(0)):
                    if pseudo_lb_t[i] < self.args.shared_class_num:
                        feat_joint_t_knw_pseudo.append(
                            torch.cat([feat_t[i], attri_dict[pseudo_lb_t[i].item()]]).view(1, -1))
                        lb_t_knw_pseudo.append(pseudo_lb_t[i])

                feat_joint_t_knw_pseudo = torch.cat(feat_joint_t_knw_pseudo, dim=0)
                lb_t_knw_pseudo = torch.tensor(lb_t_knw_pseudo).cuda()
                logits_t_pseudo = self.netC_all(feat_joint_t_knw_pseudo)
                cls_loss_t += F.cross_entropy(logits_t_pseudo, lb_t_knw_pseudo)

            # All T: feat + pred attri
            if self.args.feat_fusion_pred:
                # attri_t_pred, _ = self.netA(feat_t)  # [Ns, 85]
                attri_t_pred = prop_attri_all[feat_s.size(0):, :]

                feat_joint_t_pred = torch.zeros((feat_t.size(0), feat_t.size(1) + self.args.att_dim)).cuda()

                for i in range(feat_joint_t_pred.size(0)):
                    feat_joint_t_pred[i] = torch.cat([feat_t[i], attri_t_pred[i]])

                logits_t_pred = self.netC_all(feat_joint_t_pred)  # [Nt, K+1]
                lb_t_all_pred = copy.deepcopy(pseudo_lb_t)  # if not deepcopy, modify lb_t_all_pred will change pseudo_lb_t
                lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = self.args.shared_class_num
                cls_loss_t += F.cross_entropy(logits_t_pred, lb_t_all_pred)

            cls_loss = cls_loss_s + cls_loss_t

            # 1.4 Visual-semantic known/unknown discrimination
            # Knw--1, Unk--0
            dis_loss = torch.tensor([0.]).cuda()
            # if self.args.beta3 != 0:
            #     if self.args.feat_fusion_pseudo:
            #         logits_t_knw_pseudo = self.netD(feat_joint_t_knw_pseudo)  # [Nts, 2]
            #         bi_lb_knw = torch.ones(logits_t_knw_pseudo.size(0), dtype=torch.int64).cuda()
            #         dis_loss += F.cross_entropy(logits_t_knw_pseudo, bi_lb_knw)
            #     if self.args.feat_fusion_pred:
            #         logits_t_all_pred = self.netD(feat_joint_t_pred)  # [Nt, 2]
            #         bi_lb_all = torch.zeros(logits_t_all_pred.size(0), dtype=torch.int64).cuda()  # 0 is unknown
            #         bi_lb_all[lb_t < self.args.shared_class_num] = 1  # 1 means shared categories
            #         dis_loss += F.cross_entropy(logits_t_all_pred, bi_lb_all)

            L_S = cls_loss #+ self.args.beta3 * dis_loss
            L_R = partial_align_loss
            L_A = att_pred_loss
            loss_base = L_S + L_A + self.args.beta1 * L_R
            # (GZ, GA, C, D)

            loss = loss_base
            loss.backward()
            self.optimizer.step()

            # self.iter_num += 1
            # lr_scheduler(self.optimizer, iter_num=self.iter_num, max_iter=self.max_iter)
            iter_num = epoch * total_batch + (batch_idx + 1)
            # pre-train, max_epochs = self.args.pretrain_epochs
            lr_scheduler(self.optimizer, iter_num=iter_num, max_iter=max_iter)


            if (batch_idx + 1) % self.args.print_freq == 0:
                print("Epoch {}/{} Batch {}/{} - LS({:.2}+{:.2})+{}*LR({:.2}+{:.2})+{}*LA({:.2}+{:.2})".format(
                    epoch, self.args.epochs, batch_idx+1, total_batch, cls_loss.item(), dis_loss.item(), self.args.beta1,
                    partial_align_loss_s.item(), partial_align_loss_t.item(), self.args.beta2, att_pred_loss_knw.item(), att_pred_loss_unk.item()))
                # print(self.netC_all.fc2.weight.grad.mean(), self.netD.fc2.weight.grad.mean())
        return

    def train_ppnet(self, epoch, source_loader, target_loader, centroids, attri_dict, mode='all', args=None):
        self.netF.train()
        self.netB.train()
        self.netA.train()
        self.netC_all.train()
        self.netD.train()
        self.ppnet.train()
        self.protModel_C.train()
        self.protModel_D.train()

        print("Train protoModel: ")
        if mode == 'proto_warm':
            total_batch = min(len(source_loader), len(target_loader))
            max_iter = args.ppnet_warm_num * total_batch
        elif mode == 'proto':
            total_batch = min(len(source_loader), len(target_loader))
            max_iter = args.epochs * total_batch
        else:
            raise Exception("protoModel training mode is not correctly defined!")

        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_loader, target_loader)):

            self.optimizer.zero_grad()
            img_s = batch_s[0].cuda()
            lb_s = batch_s[1].cuda()         # 0--(shared_cls-1)

            img_t = batch_t[0].cuda()
            pseudo_lb_t = batch_t[3].cuda()  # 0--(shared_cls+K-1)
            #lb_t = batch_t[1].cuda()         # 0--shared_cls, [1][2] are ground-truth

            s_res_feat, s_res_featmap = self.netF(img_s)
            t_res_feat, t_res_featmap = self.netF(img_t)
            feat_s = self.netB(s_res_feat)
            feat_t = self.netB(t_res_feat)

            # 2. ProtoModel - optimize (GZ, P, E)
            # source
            # ns x 85       ns x 85 x 2        ns x (85*3)      ns x 512 x 7 x 7
            s_att_logits, s_open_cls_logits, s_min_distances, s_conv_features = self.ppnet(s_res_featmap)

            # target
            # nt x 85       nt x 85 x 2        nt x (85*3)      nt x 512 x 7 x 7
            t_att_logits, t_open_cls_logits, t_min_distances, t_conv_features = self.ppnet(t_res_featmap)

            if self.args.ppnet_last_layer_type == 'softmax':
                all_pred_attri_logits = torch.cat([s_open_cls_logits, t_open_cls_logits], dim=0) # (ns+nt) x 85 x 2
                all_prob_attri = torch.softmax(all_pred_attri_logits, dim=2)
                all_pred_attri = all_prob_attri[:, :, 1] # positive probability, 1 prob.
                all_pred_att_onehot = torch.argmax(all_prob_attri, dim=2) # 0 means negative, 1 for positive
            elif self.args.ppnet_last_layer_type == 'sigmoid':
                all_pred_attri_logits = torch.cat([s_att_logits, t_att_logits], dim=0)
                all_pred_attri = torch.sigmoid(all_pred_attri_logits)
            else:
                raise Exception("The ppnet last layer type is not correctly defined!")

            # 2.1 partial alignment
            ppnet_feat_s = self.avgpool(s_conv_features).squeeze() # ns x 512
            ppnet_feat_t = self.avgpool(t_conv_features).squeeze()  # nt x 512

            ppnet_centroids = centroids # just use baseModel centroids now, coud be replaced

            partial_align_loss_s = torch.tensor(0.).cuda()
            if ppnet_centroids.shape[1] == ppnet_feat_s.shape[1]:
                for i in range(ppnet_feat_s.size(0)):
                    L2 = torch.norm(ppnet_feat_s[i] - ppnet_centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[lb_s[i]]
                    if self.args.hard_neg:
                        # print("LR - hard_neg = True")
                        neg = torch.arange(0, L2.size(0)).cuda() != lb_s[i]
                        distract = -1 * L2[neg].min()
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)
                    partial_align_loss_s += attract + distract
                partial_align_loss_s /= ppnet_feat_s.size(0)
            # 1.1.2 target
            partial_align_loss_t = torch.tensor(0.).cuda()
            if ppnet_centroids.shape[1] == ppnet_feat_t.shape[1]:
                for i in range(ppnet_feat_t.size(0)):
                    L2 = torch.norm(ppnet_feat_t[i] - ppnet_centroids, dim=1)  # [shared_cls+K,]
                    attract = L2[pseudo_lb_t[i]]
                    if self.args.hard_neg:
                        neg = torch.arange(0, L2.size(0)).cuda() != pseudo_lb_t[i]
                        distract = -1 * L2[neg].min()  # original reproduce is 1.0
                    else:
                        distract = -1 * (L2.sum() - attract) / (L2.size(0) - 1)  # original reproduce is 1.0

                    partial_align_loss_t += attract + distract
                partial_align_loss_t /= ppnet_feat_t.size(0)
            partial_align_loss = partial_align_loss_s + partial_align_loss_t

            # 2.2 attributes propagation
            att_pred_loss = torch.tensor([0.]).cuda()
            att_pred_loss_knw = torch.tensor([0.]).cuda()
            att_pred_loss_unk = torch.tensor([0.]).cuda()
            all_min_distances = torch.cat([s_min_distances, t_min_distances], dim=0) # (bs*2, prot_num(3*85))

            if self.args.feat_fusion_separate:
                all_feat = torch.cat([ppnet_feat_s, ppnet_feat_t], dim=0)  # use baseModel feats to propagate
            else:
                all_feat = torch.cat([feat_s, feat_t], dim=0) # use baseModel feats to propagate
            all_lb = torch.cat([lb_s, pseudo_lb_t])

            # knw_feat = all_feat[all_lb < self.args.shared_class_num]
            knw_lb = all_lb[all_lb < self.args.shared_class_num]

            if self.args.att_propagation:
                prop_attri_all = self.AttributeProp(all_feat, all_pred_attri)
            else:
                prop_attri_all = all_pred_attri

            # 2.1.1 Knw Attri loss
            if self.args.att_propagation:
                prop_attri_knw = prop_attri_all[all_lb < self.args.shared_class_num].clamp(min=0., max=1.)
            else:
                prop_attri_knw = all_pred_attri[all_lb < self.args.shared_class_num]

            knw_attri = torch.zeros((knw_lb.size(0), self.args.att_dim)).cuda()
            for i in range(knw_attri.size(0)):
                knw_attri[i] = attri_dict[knw_lb[i].item()]


            if self.args.ppnet_last_layer_type == 'sigmoid':
                att_pred_loss_knw += F.binary_cross_entropy(prop_attri_knw, knw_attri) # n x 85
                att_pred_loss += att_pred_loss_knw + torch.mean(torch.norm(prop_attri_knw-knw_attri, p=2, dim=1))#+ att_pred_loss_unk
            elif self.args.ppnet_last_layer_type == 'softmax':
                att_pred_loss_knw += F.binary_cross_entropy(prop_attri_knw, knw_attri) + torch.mean(torch.norm(prop_attri_knw-knw_attri, p=1, dim=1))# n x 85
                att_pred_loss += att_pred_loss_knw + torch.mean(torch.norm(prop_attri_knw-knw_attri, p=2, dim=1))
            # 2.2 Prototypes clustering and separation loss
            # knw_lb = all_lb[all_lb < self.args.shared_class_num]
            gt_attri_knw = torch.zeros((knw_lb.size(0), self.args.att_dim)).cuda()
            for i in range(gt_attri_knw.size(0)):
                gt_attri_knw[i] = attri_dict[knw_lb[i].item()]

            mx = (self.ppnet.prototype_shape[1]  # 512
                        * self.ppnet.prototype_shape[2]  # 1
                        * self.ppnet.prototype_shape[3])  # 1

            # feat_s - ground-truth attr;feat_t - propagated attri
            attri_knw_binary = gt_attri_knw #prop_gt_attri_all >= 0.5 # (bs*2, att_dim) {0, 1}
            # print(attri_knw_binary.shape, (attri_knw_binary==1).sum(), (attri_knw_binary==0).sum())
            knw_min_distances = all_min_distances[all_lb < self.args.shared_class_num] #torch.cat([s_min_distances, t_min_distances]) # (bs*2, prot_num(3*85))
            # nknw x (85*3)

            # 2.2.1 Clustering cost
            cls_identity = self.ppnet.prototype_class_identity # prot_num x att_dim (255 x 85)
            # knw_n x 255    <---      knw_n x 85    dot   85 x 255
            correct_mask = torch.mm(attri_knw_binary, cls_identity.t())
            wrong_mask = 1 - correct_mask

            # prototype_class_identity: (prot_num x att_dim) e.g., ((3*85) x 85)
            # min_distances: knw_n x 255
            inverted_distances = mx-knw_min_distances #mx - knw_min_distances
            inverted_correct_dists = inverted_distances * correct_mask
            # knw_n x 255 -> knw_n x 1 x 255 --> knw_n x 1 x 85 -> knw_n x 85
            inverted_min_correct_dists = self.maxpool1d(inverted_correct_dists.unsqueeze(1)).squeeze()
            cluster_cost = mx-torch.mean(inverted_min_correct_dists.sum(1) / attri_knw_binary.sum(1))

            # 2.2.2 Separation cost
            # bs x 85 x 255 --> bs x 85 x 1
            inverted_wrong_dists = inverted_distances * wrong_mask
            inverted_wrong_dists = self.maxpool1d(inverted_wrong_dists.unsqueeze(1)).squeeze()
            separation_cost = mx-torch.mean(inverted_wrong_dists.sum(1) / (1-attri_knw_binary).sum(1))

            # if True: #use_l1_mask:
            l1_mask = 1 - torch.t(self.ppnet.prototype_class_identity).cuda()
            l1 = ((self.ppnet.last_layer.weight * l1_mask).norm(p=1) \
                 + (self.ppnet.last_layer_negative.weight * l1_mask).norm(p=1)) / 2

            # 2.3 Visual-semantic joint classification
            # 2.3.1 classification
            attri_s_pred = prop_attri_all[:feat_s.size(0), :]
            cls_loss_s = torch.tensor([0.]).cuda()

            # S: feat + pseudo attri
            if self.args.feat_fusion_pred:
                feat_joint_s_pred = torch.zeros((feat_s.size(0), feat_s.size(1) + self.args.att_dim)).cuda()
                for i in range(feat_joint_s_pred.size(0)):
                    if self.args.feat_fusion_separate: # use protoModel own ppnet_feat_s
                        feat_joint_s_pred[i] = torch.cat([ppnet_feat_s[i], attri_s_pred[i]])
                    else:
                        feat_joint_s_pred[i] = torch.cat([feat_s[i], attri_s_pred[i]])
                # logits_s_pred = self.netC_all(feat_joint_s_pred)
                logits_s_pred = self.protModel_C(feat_joint_s_pred)
                cls_loss_s += F.cross_entropy(logits_s_pred, lb_s)
            # Knw T: feat + pseudo attri
            cls_loss_t = torch.tensor([0.]).cuda()
            # All T: feat + pred attri
            if self.args.feat_fusion_pred:
                attri_t_pred = prop_attri_all[feat_s.size(0):, :]

                feat_joint_t_pred = torch.zeros((feat_t.size(0), feat_t.size(1) + self.args.att_dim)).cuda()

                for i in range(feat_joint_t_pred.size(0)):
                    if self.args.feat_fusion_separate:
                        feat_joint_t_pred[i] = torch.cat([ppnet_feat_t[i], attri_t_pred[i]])
                    else:
                        feat_joint_t_pred[i] = torch.cat([feat_t[i], attri_t_pred[i]])

                # logits_t_pred = self.netC_all(feat_joint_t_pred)  # [Nt, K+1]
                logits_t_pred = self.protModel_C(feat_joint_t_pred)
                lb_t_all_pred = copy.deepcopy(pseudo_lb_t)  # if not deepcopy, modify lb_t_all_pred will change pseudo_lb_t
                lb_t_all_pred[lb_t_all_pred >= self.args.shared_class_num] = self.args.shared_class_num
                cls_loss_t += F.cross_entropy(logits_t_pred, lb_t_all_pred)

            cls_loss = cls_loss_s + cls_loss_t

            # 2.3.2 discrimination
            # Knw--1, Unk--0
            dis_loss = torch.tensor([0.]).cuda()
            # if self.args.beta3 != 0:
            #     if self.args.feat_fusion_pred:
            #         # logits_t_all_pred = self.netD(feat_joint_t_pred)  # [Nt, 2]
            #         logits_t_all_pred = self.protModel_D(feat_joint_t_pred)  # [Nt, 2]
            #
            #         bi_lb_all = torch.zeros(logits_t_all_pred.size(0), dtype=torch.int64).cuda()  # 0 is unknown
            #         bi_lb_all[pseudo_lb_t < self.args.shared_class_num] = 1  # 1 means shared categories
            #         dis_loss += F.cross_entropy(logits_t_all_pred, bi_lb_all)

            L_S = cls_loss #+ self.args.beta3 * dis_loss
            L_A = att_pred_loss
            L_R = partial_align_loss
            L_P = cluster_cost - self.args.alpha1 * separation_cost + self.args.alpha2 * l1


            loss_ppnet = L_S + L_A + self.args.beta1 * L_R  + self.args.beta2 * L_P

            loss = loss_ppnet
            loss.backward()
            self.optimizer.step()

            iter_num = epoch * total_batch + (batch_idx + 1)

            lr_scheduler(self.optimizer, iter_num=iter_num, max_iter=max_iter)

            if (batch_idx + 1) % self.args.print_freq == 0:
                print("Epoch {}/{} Batch {}/{} - LP({:.2})=LS({:.2}+{:.2})+{}*LR({:.2})+{}*LA({:.2})+{}*lclst({:.2})-{}*lsep({:.2})+{}*l1({:.2})".format(
                    epoch, self.args.epochs, batch_idx + 1, total_batch, loss_ppnet.item(),
                    cls_loss.item(), dis_loss.item(), self.args.beta1, L_R.item(), self.args.beta2, L_A.item(),
                    self.args.alpha1, cluster_cost.item(), self.args.alpha2, separation_cost.item(), self.args.alpha3,
                    l1.item()
                ))

                # print(self.ppnet.prototype_vectors.grad.mean(), self.ppnet.last_layer.weight.grad.mean())
        return


    def save_net(self, epoch=None, centroids=None, refined_lb=None, save_net=False):
        model_save_path = self.args.save_folder
        if epoch:
            model_save_path = self.args.save_folder + '/' + str(epoch)
        if not os.path.exists(model_save_path):
            print("Create model save path: ", model_save_path)
            os.system('mkdir -p ' + model_save_path)
        if save_net:
            torch.save(self.netF.state_dict(), osp.join(model_save_path, "net_F.pt"))
            torch.save(self.netB.state_dict(), osp.join(model_save_path, "net_B.pt"))
            # torch.save(self.netC.state_dict(), osp.join(model_save_path, "net_C.pt"))
            torch.save(self.netC_all.state_dict(), osp.join(model_save_path, "net_C_all.pt"))
            torch.save(self.netD.state_dict(), osp.join(model_save_path, "net_D.pt"))
            torch.save(self.netA.state_dict(), osp.join(model_save_path, "net_A.pt"))
            torch.save(self.ppnet.state_dict(), osp.join(model_save_path, "ppnet.pt"))
            torch.save(self.protModel_C.state_dict(), osp.join(model_save_path, "protModel_C.pt"))
            torch.save(self.protModel_D.state_dict(), osp.join(model_save_path, "protModel_D.pt"))

            torch.save(self.args, osp.join(model_save_path, "args.pt"))

        if centroids is not None:
            torch.save(centroids, osp.join(model_save_path, "centroids.pt"))
        if refined_lb:
            torch.save(refined_lb, osp.join(model_save_path, "refined_lb.pt"))

        return

    def resume_weights(self, args, load_cent=False, load_ref_lb=False, load_net=True, load_eval=True):
        if args.resume:
            if load_net:
                print("Resuming trained weights: ", args.resume)
                modelpath = args.resume + '/net_F.pt'
                self.netF.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/net_B.pt'
                self.netB.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/net_A.pt'
                self.netA.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/net_C_all.pt'
                self.netC_all.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/net_D.pt'
                self.netD.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/ppnet.pt'
                self.ppnet.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/protModel_C.pt'
                self.protModel_C.load_state_dict(torch.load(modelpath))
                modelpath = args.resume + '/protModel_D.pt'
                self.protModel_D.load_state_dict(torch.load(modelpath))

            centroids, refined_lb = None, None
            if load_cent:
                centroids = torch.load(osp.join(args.resume, "centroids.pt"))
            if load_ref_lb:
                refined_lb = torch.load(osp.join(args.resume, "refined_lb.pt"))
            if load_eval:
                checkpoint_path = osp.join(args.resume+"/../")
                self.eval_info.base_info = np.load(checkpoint_path + 'base_info.npy', allow_pickle=True).item()
                self.eval_info.prot_info = np.load(checkpoint_path + 'prot_info.npy', allow_pickle=True).item()
                self.eval_info.init_info = np.load(checkpoint_path + 'init_info.npy', allow_pickle=True).item()

            return centroids, refined_lb


###################################################################

    def initialize_optimizer(self,args=None):
        # baseModel
        backbone_param_group = []
        learning_rate = args.lr
        for k, v in self.netF.named_parameters():
            backbone_param_group += [{'params': v, 'lr': learning_rate * 0.1, 'weight_decay': 1e-3}]

        base_param_group = []
        learning_rate = args.lr
        base_param_group += backbone_param_group

        for k, v in self.netB.named_parameters():
            base_param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netA.named_parameters():
            base_param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netC_all.named_parameters():
            base_param_group += [{'params': v, 'lr': learning_rate}]
        for k, v in self.netD.named_parameters():
            base_param_group += [{'params': v, 'lr': learning_rate}]

        self.base_optimizer = optim.SGD(base_param_group)
        for param_group in self.base_optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
        self.base_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.base_optimizer, step_size=5, gamma=0.1)

        # protoModel
        # After warmup, train ppnet as baseModel
        proto_param_group = []
        # learning_rate = args.lr
        if args.ppnet_last_layer_type == 'sigmoid': # for sigmoid, need larger learning rate, because gradients are small
            learning_rate = args.lr
            ppnet_last_layer_lr = learning_rate*0.1
            DC_lr = args.lr#* 0.1
        else:# softmax should be larger, as original lr not good, tried, improve slow.
            learning_rate = args.lr
            ppnet_last_layer_lr = learning_rate*0.1
            DC_lr = args.lr#* 0.1

        proto_param_group += backbone_param_group
        for k, v in self.ppnet.add_on_layers.named_parameters():
            # print("prroto params: ", k)
            proto_param_group += [{'params': v, 'lr': learning_rate}]
        proto_param_group += [{'params': self.ppnet.prototype_vectors, 'lr': learning_rate}]
        for k, v in self.ppnet.last_layer.named_parameters():
            # pint("proto params: ", k)
            proto_param_group += [{'params': v, 'lr': ppnet_last_layer_lr}]
        for k, v in self.ppnet.last_layer_negative.named_parameters():
            # print("proto params: ", k)
            proto_param_group += [{'params': v, 'lr': ppnet_last_layer_lr}]

        # to avoid error, better remove those from proto_optimizer
        # for k, v in self.netC_all.named_parameters():
        for k, v in self.protModel_C.named_parameters():
            # print("proto params: ", k)
            proto_param_group += [{'params': v, 'lr': DC_lr}]
        # for k, v in self.netD.named_parameters():
        for k, v in self.protModel_D.named_parameters():
            # print("proto params: ", k)
            proto_param_group += [{'params': v, 'lr': DC_lr}]

        self.proto_optimizer = optim.Adam(proto_param_group)
        for param_group in self.proto_optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
        self.proto_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.proto_optimizer, step_size=5, gamma=0.1)

        # For warmup, set the learning rate larger
        proto_param_group = []
        if args.ppnet_last_layer_type == 'sigmoid': # for sigmoid, need larger learning rate, because gradients are small
            learning_rate = args.lr * 10
            ppnet_last_layer_lr = learning_rate  # or 0.1*
        else:
            learning_rate = args.lr * 10
            ppnet_last_layer_lr = learning_rate  # or 0.1*

        proto_param_group += [{'params': self.ppnet.add_on_layers.parameters(), 'lr': learning_rate},
                              {'params': self.ppnet.prototype_vectors, 'lr': learning_rate},
                              ]
        proto_param_group += [
            {'params': self.ppnet.last_layer.parameters(), 'lr': ppnet_last_layer_lr},
            {'params': self.ppnet.last_layer_negative.parameters(), 'lr': ppnet_last_layer_lr},
        ]

        self.proto_warm_optimizer = optim.Adam(proto_param_group)
        for param_group in self.proto_warm_optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
        self.proto_warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.proto_warm_optimizer, step_size=5, gamma=0.1)

        # for last-layers, slightly finetune, so set the learning rate small
        last_layer_optimizer_specs = [
            {'params': self.ppnet.last_layer.parameters(), 'lr': 0.1*learning_rate},
            {'params': self.ppnet.last_layer_negative.parameters(), 'lr': 0.1*learning_rate},
        ]
        self.proto_ll_optimizer = optim.Adam(last_layer_optimizer_specs)
        for param_group in self.proto_ll_optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
        self.proto_ll_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.proto_ll_optimizer, step_size=5, gamma=0.1)

    def set_training_mode(self, mode='pre_train', finetune_backbone=False, fix_DC=True, args=None):
        '''
        Args:
            mode: [pre_train, all, base, proto, proto_vec, proto_last, proto_joint, backbone]
            finetune_backbone:
            args:
        Returns:
            None
        Set:
            self.optimizer
            self.lr_scheduler
        '''

        for k, v in self.netF.named_parameters():
            v.requires_grad = finetune_backbone  # [True | False]

        if mode == 'pre_train':
            # base_optimizer only contains baseModel params, so no need to change ppnet status
            for k, v in self.netB.named_parameters():
                v.requires_grad = True
            for k, v in self.netA.named_parameters():
                v.requires_grad = True
            for k, v in self.netC_all.named_parameters():
                v.requires_grad = True
            for k, v in self.netD.named_parameters():
                v.requires_grad = True
            self.optimizer = self.base_optimizer
            self.scheduler = self.base_lr_scheduler
        elif mode == 'base':
            # base_optimizer only contains baseModel params, so no need to change ppnet status
            for k, v in self.netB.named_parameters():
                v.requires_grad = True
            for k, v in self.netA.named_parameters():
                v.requires_grad = True
            for k, v in self.netC_all.named_parameters():
                v.requires_grad = True
            for k, v in self.netD.named_parameters():
                v.requires_grad = True

            self.optimizer = self.base_optimizer
            self.scheduler = self.base_lr_scheduler

        elif mode == 'proto':  # whole protoNet with/without backbone
            # proto_optimizer contains D&C params, so need to change DC&ppnet status
            for k, v in self.ppnet.add_on_layers.named_parameters():
                v.requires_grad = True
            self.ppnet.prototype_vectors.requires_grad = True
            for k, v in self.ppnet.last_layer.named_parameters():
                v.requires_grad = True
            for k, v in self.ppnet.last_layer.named_parameters():
                v.requires_grad = True

            # for k, v in self.netC_all.named_parameters():
            for k, v in self.protModel_C.named_parameters():
                v.requires_grad = not fix_DC
            # for k, v in self.netD.named_parameters():
            for k, v in self.protModel_D.named_parameters():
                v.requires_grad = not fix_DC


            self.optimizer = self.proto_optimizer
            self.scheduler = self.proto_lr_scheduler

        elif mode == 'proto_warm':  # warm_up protoNet, without backbone
            for k, v in self.ppnet.add_on_layers.named_parameters():
                v.requires_grad = True
            self.ppnet.prototype_vectors.requires_grad = True
            for k, v in self.ppnet.last_layer.named_parameters():
                v.requires_grad = True
            for k, v in self.ppnet.last_layer_negative.named_parameters():
                v.requires_grad = True

            # for k, v in self.netC_all.named_parameters():
            for k, v in self.protModel_C.named_parameters():
                v.requires_grad = not fix_DC
            # for k, v in self.netD.named_parameters():
            for k, v in self.protModel_D.named_parameters():
                v.requires_grad = not fix_DC

            self.optimizer = self.proto_warm_optimizer
            self.scheduler = self.proto_warm_lr_scheduler

        elif mode == 'proto_last':  # last layer of protoNet, without backbone
            for k, v in self.ppnet.last_layer.named_parameters():
                v.requires_grad = True
            for k, v in self.ppnet.last_layer_negative.named_parameters():
                v.requires_grad = True
            self.optimizer = self.proto_ll_optimizer
            self.scheduler = self.proto_ll_lr_scheduler

        else:
            raise Exception("Training mode is not correctly defined!")
        return