import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.data_list import ImageSet
from models.model import XSROSDA


def train(args):
    # Prepare Attribute Vectors for each class
    attri_dict = {}
    attri_mat = torch.tensor(np.load(args.attri_path)).float().cuda()  # [total_cls_n, 85]
    for i in range(attri_mat.size()[0]):
        attri_dict[i] = attri_mat[i]  # [85,]

    # Prepare Data
    source_set = ImageSet(args.source_path, args, train=True, balanced=True)
    target_set = ImageSet(args.target_path, args, train=True)
    test_set = ImageSet(args.target_path, args, train=False)

    # Prepare training data
    source_loader = DataLoader(source_set, batch_size=args.batchsize, shuffle=True,
                               num_workers=4, drop_last=True)
    target_loader = DataLoader(target_set, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

    # For pseudo labels
    source_loader_estimate = DataLoader(source_set, batch_size=args.batchsize, shuffle=False,
                                       num_workers=4, drop_last=False)
    target_loader_estimate = DataLoader(target_set, batch_size=args.batchsize, shuffle=False,
                                       num_workers=4, drop_last=False)
    s_loader_len = len(source_loader)
    t_loader_len = len(target_loader)

    # Evaluation
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                               num_workers=4, drop_last=False)


    # Prepare model
    model = XSROSDA(args, s_loader_len, t_loader_len)
    model.initialize_optimizer(args)

    loaded_epoch = 'none'
    if not args.resume:
        print(" Initialize pseudo labels ------------------ ")
        centroids, refined_lb = model.estimate_pseudo_label(source_loader_estimate, target_loader_estimate)
        centroids = centroids.cuda()
        model.set_training_mode(mode='pre_train', finetune_backbone=args.finetune_backbone, fix_DC=False, args=args)
        for epoch in range(args.pretrain_epochs):
            # centroids: tensor [shared_class_n+K, 256]
            # refined_lb: list of pseudo labels(int) of target samples
            print('\n Pretrain Epoch {}----------------'.format(epoch))
            target_set_pseudo = ImageSet(args.target_path, args, train=True, pseudo_lb=refined_lb)
            target_loader = DataLoader(target_set_pseudo, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)
            model.pre_train(epoch, source_loader, target_loader, centroids, attri_dict, mode='pre-train', args=args)
            centroids, refined_lb = model.estimate_pseudo_label(source_loader_estimate, target_loader_estimate)
            centroids = centroids.cuda()
            model.save_net(epoch="pre_trained", centroids=centroids, refined_lb=refined_lb, save_net=True)
            model.eval_info.update_epoch_record(epoch='pre_trained', dataloader=test_loader, model=model,
                                          attri_mat=attri_mat.detach().cpu().numpy(), args=args)
    else:
        print("Loading pre-trained weights...")
        centroids, refined_lb = model.resume_weights(args, load_cent=True, load_ref_lb=True, load_net=True)
        loaded_epoch = model.eval_info.base_info['epochs'][-1]
        print("Loaded checkpoint - epoch: ", loaded_epoch)
        model.eval_info.update_epoch_record(epoch=loaded_epoch, dataloader=test_loader, model=model,
                                            attri_mat=attri_mat.detach().cpu().numpy(), args=args)

    # Copy pre-trained C&D params to protModel_C & D
    model.protModel_C.load_state_dict(model.netC_all.state_dict())
    model.protModel_D.load_state_dict(model.netD.state_dict())


    # Use large learning-rate to quickly warmup protoModel
    if (not args.resume) or ('pre_trained' in args.resume): # start warmup
        loaded_epoch = 'pre_trained'
        print("Start warming up protModel")
        for epoch in range(args.ppnet_warm_num):
            print('\n\nWarm up ppnet {}----------------'.format(epoch))
            target_set_pseudo = ImageSet(args.target_path, args, train=True, pseudo_lb=refined_lb)
            target_loader = DataLoader(target_set_pseudo, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

            model.set_training_mode(mode='proto_warm', finetune_backbone=args.finetune_backbone, fix_DC=True, args=args)
            model.train_ppnet(epoch, source_loader, target_loader, centroids, attri_dict, mode='proto', args=args)
            if (epoch + 1) % args.test_interval == 0:
                print("Evaluating... Epoch: {}/{}".format(epoch, args.ppnet_warm_num))
                model.eval_info.update_epoch_record(epoch='warmup', dataloader=test_loader, model=model,
                                                    attri_mat=attri_mat.detach().cpu().numpy(), args=args)
                model.save_net(epoch='warmup', centroids=centroids, refined_lb=refined_lb, save_net=True)
    elif 'warmup' in args.resume:
        loaded_epoch = 'warmup'
        print("Loaded ", args.resume, ". -- Skip warmup protModel!")
    else: # 'warmup' in args.resume: # start training
        print("Loaded ", args.resume, loaded_epoch, ". -- Skip warmup protModel!")

    start_epoch = -1 if type(loaded_epoch) == str else loaded_epoch
    early_stop_count = 0
    for epoch in range(start_epoch+1, args.epochs):
        print('\n\nEpoch {}----------------'.format(epoch))
        target_set_pseudo = ImageSet(args.target_path, args, train=True, pseudo_lb=refined_lb)
        target_loader = DataLoader(target_set_pseudo, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)

        model.set_training_mode(mode='base', finetune_backbone=args.finetune_backbone, fix_DC=False, args=args)
        model.train_base(epoch, source_loader, target_loader, centroids, attri_dict, mode='base', args=args)

        model.set_training_mode(mode='proto', finetune_backbone=args.finetune_backbone, fix_DC=False, args=args)
        model.train_ppnet(epoch, source_loader, target_loader, centroids, attri_dict, mode='proto', args=args)

        if (epoch + 1) % args.test_interval == 0:
            print("Evaluating... Epoch: {}/{}".format(epoch, args.epochs))
            model.eval_info.update_epoch_record(epoch=epoch, dataloader=test_loader, model=model,
                                                attri_mat=attri_mat.detach().cpu().numpy(), args=args)
            #---------------------------------------------------------------------------
            save_best_model(model=model, centroids=centroids, refined_lb=refined_lb, epoch=epoch, args=args)

        # Early stop
        # 1.1 best baseModel
        base_cur_res = max(model.eval_info.base_info['srosda']['D_dis']['H'][-1],
                      model.eval_info.base_info['srosda']['C_dis']['H'][-1])

        base_best_res = max(max(model.eval_info.base_info['srosda']['D_dis']['H'][-min((epoch + 1), 100):]),
                       max(model.eval_info.base_info['srosda']['C_dis']['H'][-min((epoch + 1), 100):]))
        # 1.2 best protModel
        prot_cur_res = max(model.eval_info.prot_info['srosda']['D_dis']['H'][-1],
                      model.eval_info.prot_info['srosda']['C_dis']['H'][-1])
        prot_best_res = max(max(model.eval_info.prot_info['srosda']['D_dis']['H'][-min((epoch + 1), 100):]),
                       max(model.eval_info.prot_info['srosda']['C_dis']['H'][-min((epoch + 1), 100):]))

        if (base_cur_res < base_best_res) and (prot_cur_res < prot_best_res):
            # Neither base/proto model achieve better performance
            if early_stop_count < 10:
                early_stop_count += 1
            else:
                print("Haven't improved performance for 10 epochs! Stop early!")
                break
        else: # once one branch achieve better, reset early_stop_count.
            early_stop_count = 0
        print("Early stop count = ", early_stop_count)

    return


def save_best_model(model=None, centroids=None, refined_lb=None, epoch=None, args=None):
    if type(epoch) == str: # pretrain / warmup peiriod, already stored net
        return
    elif epoch == 0:
        print("Storing first (0) epoch models.")
        model.save_net(epoch='srosda_best_baseModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        model.save_net(epoch='srosda_best_protModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        model.save_net(epoch='osda_best_baseModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        model.save_net(epoch='osda_best_protModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
    else: # epoch >= 1
        # 1. SR-OSDA best model (base & prot)
        # 1.1 best baseModel
        cur_res = max(model.eval_info.base_info['srosda']['D_dis']['H'][-1],
                      model.eval_info.base_info['srosda']['C_dis']['H'][-1])

        best_res = max(max(model.eval_info.base_info['srosda']['D_dis']['H'][-(epoch + 1):]),
                       max(model.eval_info.base_info['srosda']['C_dis']['H'][-(epoch + 1):]))
        if cur_res >= best_res:
            print("Storing best SR-OSDA baseModel: {} - SROSDA: current_protModel({:.4}) >= best_protModel({:.4})".format(
                    epoch, cur_res, best_res))
            model.save_net(epoch='srosda_best_baseModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        else:
            print("Not save. SROSDA: current_baseModel({:.4}) <= best_baseModel({:.4})".format(cur_res, best_res))

        # 1.2 best protModel
        cur_res = max(model.eval_info.prot_info['srosda']['D_dis']['H'][-1],
                      model.eval_info.prot_info['srosda']['C_dis']['H'][-1])
        best_res = max(max(model.eval_info.prot_info['srosda']['D_dis']['H'][-(epoch + 1):]),
                       max(model.eval_info.prot_info['srosda']['C_dis']['H'][-(epoch + 1):]))
        if cur_res >= best_res:
            print("Storing best SR-OSDA protModel: {} - SROSDA: current_protModel({:.4}) >= best_protModel({:.4})".format(
                epoch, cur_res, best_res))
            model.save_net(epoch='srosda_best_protModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        else:
            print("Not save. SROSDA: current_protModel({:.4}) < best_protModel({:.4})".format(cur_res, best_res))

        # 2. OSDA best model (base & prot)
        # 2.1 best baseModel
        cur_res = model.eval_info.base_info['osda']['OS'][-1]
        best_res = max(model.eval_info.base_info['osda']['OS'][-(epoch + 1):])
        if cur_res >= best_res:
            print("Storing best OSDA baseModel: {} - OSDA: current_protModel({:.4}) >= best_protModel({:.4})".format(
                epoch, cur_res, best_res))
            model.save_net(epoch='osda_best_baseModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        else:
            print("Not save. OSDA: current_baseModel({:.4}) < best_baseModel({:.4})".format(cur_res, best_res))

        # 2.2 best protModel
        cur_res = model.eval_info.prot_info['osda']['OS'][-1]
        best_res = max(model.eval_info.prot_info['osda']['OS'][-(epoch + 1):])
        if cur_res >= best_res:
            print("Storing best OSDA protModel: {} - OSDA: current_protModel({:.4}) >= best_protModel({:.4})".format(epoch, cur_res, best_res))
            model.save_net(epoch='osda_best_protModel', centroids=centroids, refined_lb=refined_lb, save_net=True)
        else:
            print("Not save. OSDA: current_protModel({:.4}) < best_protModel({:.4})".format(cur_res, best_res))

