import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
from scipy.spatial.distance import cdist as distance
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('always')

class EvalInfo():
    def __init__(self, args):
        self.init_info = {
            'osda': { # accuracy for open-set domain adaptation
                'OS*': [], # seen classes average accuracy
                'OS^': [], # unseen classes average accuracy
                'OS': [], # overall classes average accuracy (Ks+1)
                'H': [] # Harmonic mean classes average accuracy (2SH/S+H)
            },
        }
        self.base_info = {
            'epochs': [],
            'osda': { # accuracy for open-set domain adaptation
                'OS*': [], # seen classes average accuracy
                'OS^': [], # unseen classes average accuracy
                'OS': [], # overall classes average accuracy (Ks+1)
                'H': [] # Harmonic mean classes average accuracy (2SH/S+H)
            },
            'srosda': { # accuracy for semantic recovery osda
                'no_dis': {# no seen/unseen discriminator used
                    'S': [],
                    'U': [],
                    'H': []
                },
                'D_dis': {# use D as seen/unseen discriminator
                    'S': [],
                    'U': [],
                    'H': []
                },
                'C_dis': {# use C as seen/unseen discriminator
                    'S': [],
                    'U': [],
                    'H': []
                }

            },
            'att': {
                'Precision': [],  # list of precisions for all classes
                'Recall': [],
                'F1': [],
                'Avg_Precision': [],  # class-wise average for all classes
                'Avg_Recall': [],
                'Avg_F1': [],
                'Avg_sample_Precision': [],
                # sample-wise average precision for each sample, only for 1/positive dimension
                'Avg_sample_Recall': [],  # only for 1/positive dimension
                'Avg_sample_F1': []  # only for 1/positive dimension
            }
        }

        self.prot_info = {
            'epochs': [],
            'osda': { # accuracy for open-set domain adaptation
                'OS*': [], # seen classes average accuracy
                'OS^': [], # unseen classes average accuracy
                'OS': [], # overall classes average accuracy (Ks+1)
                'H': [] # Harmonic mean classes average accuracy (2SH/S+H)
            },
            'srosda': { # accuracy for semantic recovery osda
                'no_dis': {# no seen/unseen discriminator used
                    'S': [],
                    'U': [],
                    'H': []
                },
                'D_dis': {# use D as seen/unseen discriminator
                    'S': [],
                    'U': [],
                    'H': []
                },
                'C_dis': {# use C as seen/unseen discriminator
                    'S': [],
                    'U': [],
                    'H': []
                }

            },
            'att': {
                'Precision': [],  # list of precisions for all classes
                'Recall': [],
                'F1': [],
                'Avg_Precision': [],  # class-wise average for all classes
                'Avg_Recall': [],
                'Avg_F1': [],
                'Avg_sample_Precision': [],
                # sample-wise average precision for each sample, only for 1/positive dimension
                'Avg_sample_Recall': [],  # only for 1/positive dimension
                'Avg_sample_F1': []  # only for 1/positive dimension
            }
        }

        self.epoch_record = {
            'y_osda': [], # open-set Ks+1 ground-truth lbls
            'y_all': [], # all ground-truth lbls
            'att_gt': [],  # nt x att_dim (eg., nt x 85)

            # base
            'att_base_pred': [],  # nt x att_dim, (save probability)
            'y_base_pred': [],  # (nt, ), [1, ..., Ks+1]
            'y_dis_base_pred': [],  # (nt, ), {0, 1}, 0=seen, 1=unseen
            'att_sim_base': [], # nt x (Ks+Kt)

            # proto
            'att_prot_pred': [],  # nt x att_dim, (save probability)
            'y_prot_pred': [], # (nt, )
            'y_dis_prot_pred': [], # nt x 1 {0, 1}
            'att_sim_prot': [] # nt x (Ks+Kt)

        }

        self.output_path = '..'  # ./checkpoints
        self.task = '' # Awa2_102Painting_17
        self.epoch = -1
        self.sim_type = 'cosine' # [cosine, euclidean, hamming]
        self.args = args


    def reset_epoch_record(self, epoch):
        for k in self.epoch_record:
            self.epoch_record[k] = []
        self.epoch = epoch

    @torch.no_grad()
    def update_epoch_record(self, epoch, dataloader, model, attri_mat, args=None):
        self.reset_epoch_record(epoch=epoch)
        self.epoch_record = self.test_on_loader(dataloader, model, attri_mat=attri_mat, branch='both', args=args)
        self.calculate_metric(epoch)
        self.print_info(epoch)
        self.reset_epoch_record(epoch=epoch)
        self.save_info(epoch=epoch)
        self.plot(epoch=epoch)

    def calculate_metric(self, epoch):
        # 1. baseModel
        self.base_info['epochs'].append(epoch)
        # 1.1 osda
        matrix = confusion_matrix(self.epoch_record['y_all'], self.epoch_record['y_base_pred'])
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        os_seen = cls_wise_acc[:self.args.shared_class_num].mean()
        os_unseen = (cls_wise_acc[self.args.shared_class_num:, self.args.shared_class_num]
                     / cls_wise_acc[self.args.shared_class_num:].sum(axis=1)).mean()
        os = (os_seen * self.args.shared_class_num + os_unseen) / (self.args.shared_class_num + 1)
        self.base_info['osda']['OS*'].append(os_seen)
        self.base_info['osda']['OS^'].append(os_unseen)
        self.base_info['osda']['OS'].append(os)
        self.base_info['osda']['H'].append(2*os_seen*os_unseen / (os_seen + os_unseen))

        # 1.2 sr-osda
        # 1.2.1 no_dis
        y_srosda_pred = np.argmax(self.epoch_record['att_sim_base'], axis=1)
        matrix = confusion_matrix(self.epoch_record['y_all'], y_srosda_pred)
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.base_info['srosda']['no_dis']['S'].append(s)
        self.base_info['srosda']['no_dis']['U'].append(u)
        self.base_info['srosda']['no_dis']['H'].append(h)

        # 1.2.2 D_dis
        dis_pred = self.epoch_record['y_dis_base_pred']
        y_srosda_pred = []
        for i in range(len(self.epoch_record['att_sim_base'])):
            sim_i = self.epoch_record['att_sim_base'][i]
            if dis_pred[i] == 1: # seen
                y_srosda_pred.append(np.argmax(sim_i[:self.args.shared_class_num]))
            elif dis_pred[i] == 0: # unseen
                y_srosda_pred.append(np.argmax(sim_i[self.args.shared_class_num:]) + self.args.shared_class_num)
            else:
                raise Exception("Error with seen/unseen discrimination results.")

        matrix = confusion_matrix(self.epoch_record['y_all'], np.array(y_srosda_pred))
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.base_info['srosda']['D_dis']['S'].append(s)
        self.base_info['srosda']['D_dis']['U'].append(u)
        self.base_info['srosda']['D_dis']['H'].append(h)

        # 1.2.3 C_dis
        dis_pred = [1 if y < self.args.shared_class_num else 0 for y in self.epoch_record['y_base_pred']]
        y_srosda_pred = []
        for i in range(len(self.epoch_record['att_sim_base'])):
            sim_i = self.epoch_record['att_sim_base'][i]
            if dis_pred[i] == 1: # seen
                y_srosda_pred.append(np.argmax(sim_i[:self.args.shared_class_num]))
            elif dis_pred[i] == 0: # unseen
                y_srosda_pred.append(np.argmax(sim_i[self.args.shared_class_num:]) + self.args.shared_class_num)
            else:
                raise Exception("Error with seen/unseen discrimination results.")

        matrix = confusion_matrix(self.epoch_record['y_all'], np.array(y_srosda_pred))
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.base_info['srosda']['C_dis']['S'].append(s)
        self.base_info['srosda']['C_dis']['U'].append(u)
        self.base_info['srosda']['C_dis']['H'].append(h)

        # 1.3 att
        binary_att_pred = np.array(self.epoch_record['att_base_pred']) >= 0.5

        precision, recall, f1, sup = precision_recall_fscore_support(self.epoch_record['att_gt'], binary_att_pred)
        self.base_info['att']['Precision'].append(precision)
        self.base_info['att']['Recall'].append(recall)
        self.base_info['att']['F1'].append(f1)
        self.base_info['att']['Avg_Precision'].append(precision.mean())
        self.base_info['att']['Avg_Recall'].append(recall.mean())
        self.base_info['att']['Avg_F1'].append(f1.mean())

        precision, recall, f1, sup = precision_recall_fscore_support(self.epoch_record['att_gt'], binary_att_pred, average='samples')
        self.base_info['att']['Avg_sample_Precision'].append(precision)
        self.base_info['att']['Avg_sample_Recall'].append(recall)
        self.base_info['att']['Avg_sample_F1'].append(f1)

        # 2. protoModel
        self.prot_info['epochs'].append(epoch)
        # 2.1 osda
        matrix = confusion_matrix(self.epoch_record['y_osda'], self.epoch_record['y_prot_pred'])
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        os_seen = cls_wise_acc[:self.args.shared_class_num].mean()
        os_unseen = (cls_wise_acc[self.args.shared_class_num:, self.args.shared_class_num]
                     / cls_wise_acc[self.args.shared_class_num:].sum(axis=1)).mean()
        os = (os_seen * self.args.shared_class_num + os_unseen) / (self.args.shared_class_num + 1)
        self.prot_info['osda']['OS*'].append(os_seen)
        self.prot_info['osda']['OS^'].append(os_unseen)
        self.prot_info['osda']['OS'].append(os)
        self.prot_info['osda']['H'].append(2 * os_seen * os_unseen / (os_seen + os_unseen))


        # 2.2 sr-osda
        # 2.2.1 no_dis
        y_srosda_pred = np.argmax(self.epoch_record['att_sim_prot'], axis=1)
        matrix = confusion_matrix(self.epoch_record['y_all'], y_srosda_pred)
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.prot_info['srosda']['no_dis']['S'].append(s)
        self.prot_info['srosda']['no_dis']['U'].append(u)
        self.prot_info['srosda']['no_dis']['H'].append(h)

        # 2.2.2 D_dis
        dis_pred = self.epoch_record['y_dis_prot_pred']
        y_srosda_pred = []
        for i in range(len(self.epoch_record['att_sim_prot'])):
            sim_i = self.epoch_record['att_sim_prot'][i]
            if dis_pred[i] == 1:  # seen
                y_srosda_pred.append(np.argmax(sim_i[:self.args.shared_class_num]))
            elif dis_pred[i] == 0:  # unseen
                y_srosda_pred.append(np.argmax(sim_i[self.args.shared_class_num:]) + self.args.shared_class_num)
            else:
                raise Exception("Error with seen/unseen discrimination results.")

        matrix = confusion_matrix(self.epoch_record['y_all'], y_srosda_pred)
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.prot_info['srosda']['D_dis']['S'].append(s)
        self.prot_info['srosda']['D_dis']['U'].append(u)
        self.prot_info['srosda']['D_dis']['H'].append(h)

        # 2.2.3 C_dis
        dis_pred = [1 if y < self.args.shared_class_num else 0 for y in self.epoch_record['y_prot_pred']]
        y_srosda_pred = []
        for i in range(len(self.epoch_record['att_sim_prot'])):
            sim_i = self.epoch_record['att_sim_prot'][i]
            if dis_pred[i] == 1:  # seen
                y_srosda_pred.append(np.argmax(sim_i[:self.args.shared_class_num]))
            elif dis_pred[i] == 0:  # unseen
                y_srosda_pred.append(np.argmax(sim_i[self.args.shared_class_num:]) + self.args.shared_class_num)
            else:
                raise Exception("Error with seen/unseen discrimination results.")

        matrix = confusion_matrix(self.epoch_record['y_all'], y_srosda_pred)
        cls_wise_acc = matrix.diagonal() / matrix.sum(axis=1)
        s = cls_wise_acc[:self.args.shared_class_num].mean()
        u = cls_wise_acc[self.args.shared_class_num:].mean()
        h = 2 * s * u / (s + u)
        self.prot_info['srosda']['C_dis']['S'].append(s)
        self.prot_info['srosda']['C_dis']['U'].append(u)
        self.prot_info['srosda']['C_dis']['H'].append(h)

        # 2.3 att
        binary_att_pred = np.array(self.epoch_record['att_prot_pred']) >= 0.5
        precision, recall, f1, sup = precision_recall_fscore_support(self.epoch_record['att_gt'], binary_att_pred)
        self.prot_info['att']['Precision'].append(precision)
        self.prot_info['att']['Recall'].append(recall)
        self.prot_info['att']['F1'].append(f1)
        self.prot_info['att']['Avg_Precision'].append(precision.mean())
        self.prot_info['att']['Avg_Recall'].append(recall.mean())
        self.prot_info['att']['Avg_F1'].append(f1.mean())

        precision, recall, f1, sup = precision_recall_fscore_support(self.epoch_record['att_gt'], binary_att_pred,
                                                                     average='samples')
        self.prot_info['att']['Avg_sample_Precision'].append(precision)
        self.prot_info['att']['Avg_sample_Recall'].append(recall)
        self.prot_info['att']['Avg_sample_F1'].append(f1)

    def print_info(self, epoch):
        print("--------------------- {} ----------------------------".format(epoch))
        # print base
        print("baseModel info:")
        for task in self.base_info: # [epochs, osda, srosda, att]
            print("Task: ", task)
            if type(self.base_info[task]) == dict:
                for k in self.base_info[task].keys():
                    if k in ['Precision', 'Recall', 'F1']:
                        continue
                    if type(self.base_info[task][k]) == list:
                        print(k, [round(i, 4) for i in self.base_info[task][k][-3:]], ' | best: ',
                              round(np.max(self.base_info[task][k]), 4))
                    elif type(self.base_info[task][k]) == dict:
                        print(k)
                        for c in self.base_info[task][k].keys(): # [S, U, H]
                            if type(self.base_info[task][k][c]) == list:
                                print(c, [round(i, 4) for i in self.base_info[task][k][c][-3:]], ' | best: ',
                                      round(np.max(self.base_info[task][k][c]), 4))
                            else:
                                print(self.base_info[task][k][c])
                    else:
                        print(k, self.base_info[task][k])

            else:
                print(task, self.base_info[task])

        # print prot
        print("----------------------------------------------------------")
        print("protModel info:")
        for task in self.prot_info:  # [osda, srosda, att]
            print("Task: ", task)
            if type(self.prot_info[task]) == dict:
                for k in self.prot_info[task].keys():
                    if k in ['Precision', 'Recall', 'F1']:
                        continue
                    if type(self.prot_info[task][k]) == list:
                        print(k, [round(i, 4) for i in self.prot_info[task][k][-3:]], ' | best: ',
                              round(max(self.prot_info[task][k]), 4))
                    elif type(self.prot_info[task][k]) == dict:
                        print(k)
                        for c in self.prot_info[task][k].keys():
                            if type(self.prot_info[task][k][c]) == list:
                                print(c, [round(i, 4) for i in self.prot_info[task][k][c][-3:]], ' | best: ',
                                      round(max(self.prot_info[task][k][c]), 4))
                            else:
                                print(self.prot_info[task][k][c])
                    else:
                        print(k, self.prot_info[task][k])
            else:
                print(task, self.prot_info[task])

        print("-------------------------------------------------")


    def save_info(self, epoch=None):
        # save results
        path = self.args.save_folder
        np.save(file=path+'/base_info.npy', arr=self.base_info)
        np.save(file=path + '/init_info.npy', arr=self.init_info)
        np.save(file=path + '/prot_info.npy', arr=self.prot_info)

    def plot(self, epoch):
        # print(self.init_info)
        path = self.args.save_folder
        plot_all = True
        if plot_all:
            last_epoch = len(self.base_info['epochs'])
        else:
            last_epoch = self.base_info['epochs'][-1]
            last_epoch = last_epoch+1 if type(last_epoch)==int else 1
        # 1. plot osda (os*, os^, os, H) x (base, prot)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
        # 1.1 OS*
        ax1 = axes[0]
        ax1.plot(self.base_info['osda']['OS*'][-last_epoch:], c='C0', label='baseModel')
        ax1.axhline(y=self.base_info['osda']['OS*'][0], c='C0', linestyle='--', label='base_init')

        ax1.plot(self.prot_info['osda']['OS*'][-last_epoch:], c='C5', label='protModel')
        ax1.axhline(y=self.prot_info['osda']['OS*'][0], c='C5', linestyle='--', label='prot_init')

        ax1.legend()
        ax1.set_title("OS*")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)

        # 1.2 OS^
        ax2 = axes[1]
        ax2.plot(self.base_info['osda']['OS^'][-last_epoch:], c='C0', label='baseModel')
        ax2.axhline(y=self.base_info['osda']['OS^'][0], c='C0', linestyle='--', label='base_init')

        ax2.plot(self.prot_info['osda']['OS^'][-last_epoch:], c='C5', label='protModel')
        ax2.axhline(y=self.prot_info['osda']['OS^'][0], c='C5', linestyle='--', label='prot_init')

        ax2.legend()
        ax2.set_title("OS^")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)

        # 1.3 OS
        ax3 = axes[2]
        ax3.plot(self.base_info['osda']['OS'][-last_epoch:], c='C0', label='baseModel')
        ax3.axhline(y=self.base_info['osda']['OS'][0], c='C0', linestyle='--', label='base_init')

        ax3.plot(self.prot_info['osda']['OS'][-last_epoch:], c='C5', label='protModel')
        ax3.axhline(y=self.prot_info['osda']['OS'][0], c='C5', linestyle='--', label='prot_init')
        ax3.legend()
        ax3.set_title("OS")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1)

        # 1.4 H
        ax4 = axes[3]
        ax4.plot(self.base_info['osda']['H'][-last_epoch:], c='C0', label='baseModel')
        ax4.axhline(y=self.base_info['osda']['H'][0], c='C0', linestyle='--', label='base_init')

        ax4.plot(self.prot_info['osda']['H'][-last_epoch:], c='C5', label='protModel')
        ax4.axhline(y=self.prot_info['osda']['H'][0], c='C5', linestyle='--', label='prot_init')
        ax4.legend()
        ax4.set_title("H")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Accuracy")
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(path + '/osda.pdf')

        # 2. plot sr-osda (S, U, H) x (base, prot)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
        # 2.1 S
        ax1 = axes[0]
        ax1.plot(self.base_info['srosda']['no_dis']['S'][-last_epoch:], c='C0', label='baseModel-nodis')
        ax1.axhline(y=self.base_info['srosda']['no_dis']['S'][0], c='C0', linestyle='--', label='base_init-nodis')
        ax1.plot(self.base_info['srosda']['D_dis']['S'][-last_epoch:], c='C1', label='baseModel-Ddis')
        ax1.axhline(y=self.base_info['srosda']['D_dis']['S'][0], c='C1', linestyle='--', label='base_init-Ddis')
        ax1.plot(self.base_info['srosda']['C_dis']['S'][-last_epoch:], c='C2', label='baseModel-Cdis')
        ax1.axhline(y=self.base_info['srosda']['C_dis']['S'][0], c='C2', linestyle='--', label='base_init-Cdis')

        ax1.plot(self.prot_info['srosda']['no_dis']['S'][-last_epoch:], c='C5', label='protModel-nodis')
        ax1.axhline(y=self.prot_info['srosda']['no_dis']['S'][0], c='C5', linestyle='--', label='prot_init-nodis')
        ax1.plot(self.prot_info['srosda']['D_dis']['S'][-last_epoch:], c='C6', label='protModel-Ddis')
        ax1.axhline(y=self.prot_info['srosda']['D_dis']['S'][0], c='C6', linestyle='--', label='prot_init-Ddis')
        ax1.plot(self.prot_info['srosda']['C_dis']['S'][-last_epoch:], c='C7', label='protModel-Cdis')
        ax1.axhline(y=self.prot_info['srosda']['C_dis']['S'][0], c='C7', linestyle='--', label='prot_init-Cdis')

        ax1.legend(ncol=2)
        ax1.set_title("S")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1)

        # 2.2 U
        ax2 = axes[1]
        ax2.plot(self.base_info['srosda']['no_dis']['U'][-last_epoch:], c='C0', label='baseModel-nodis')
        ax2.axhline(y=self.base_info['srosda']['no_dis']['U'][0], c='C0', linestyle='--', label='base_init-nodis')
        ax2.plot(self.base_info['srosda']['D_dis']['U'][-last_epoch:], c='C1', label='baseModel-Ddis')
        ax2.axhline(y=self.base_info['srosda']['D_dis']['U'][0], c='C1', linestyle='--', label='base_init-Ddis')
        ax2.plot(self.base_info['srosda']['C_dis']['U'][-last_epoch:], c='C2', label='baseModel-Cdis')
        ax2.axhline(y=self.base_info['srosda']['C_dis']['U'][0], c='C2', linestyle='--', label='base_init-Cdis')

        ax2.plot(self.prot_info['srosda']['no_dis']['U'][-last_epoch:], c='C5', label='protModel-nodis')
        ax2.axhline(y=self.prot_info['srosda']['no_dis']['U'][0], c='C5', linestyle='--', label='prot_init-nodis')
        ax2.plot(self.prot_info['srosda']['D_dis']['U'][-last_epoch:], c='C6', label='protModel-Ddis')
        ax2.axhline(y=self.prot_info['srosda']['D_dis']['U'][0], c='C6', linestyle='--', label='prot_init-Ddis')
        ax2.plot(self.prot_info['srosda']['C_dis']['U'][-last_epoch:], c='C7', label='protModel-Cdis')
        ax2.axhline(y=self.prot_info['srosda']['C_dis']['U'][0], c='C7', linestyle='--', label='prot_init-Cdis')

        ax2.legend(ncol=2)
        ax2.set_title("U")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1)

        # 2.3 H
        ax3 = axes[2]
        ax3.plot(self.base_info['srosda']['no_dis']['H'][-last_epoch:], c='C0', label='baseModel-nodis')
        ax3.axhline(y=self.base_info['srosda']['no_dis']['H'][0], c='C0', linestyle='--', label='base_init-nodis')
        ax3.plot(self.base_info['srosda']['D_dis']['H'][-last_epoch:], c='C1', label='baseModel-Ddis')
        ax3.axhline(y=self.base_info['srosda']['D_dis']['H'][0], c='C1', linestyle='--', label='base_init-Ddis')
        ax3.plot(self.base_info['srosda']['C_dis']['H'][-last_epoch:], c='C2', label='baseModel-Cdis')
        ax3.axhline(y=self.base_info['srosda']['C_dis']['H'][0], c='C2', linestyle='--', label='base_init-Cdis')

        ax3.plot(self.prot_info['srosda']['no_dis']['H'][-last_epoch:], c='C5', label='protModel-nodis')
        ax3.axhline(y=self.prot_info['srosda']['no_dis']['H'][0], c='C5', linestyle='--', label='prot_init-nodis')
        ax3.plot(self.prot_info['srosda']['D_dis']['H'][-last_epoch:], c='C6', label='protModel-Ddis')
        ax3.axhline(y=self.prot_info['srosda']['D_dis']['H'][0], c='C6', linestyle='--', label='prot_init-Ddis')
        ax3.plot(self.prot_info['srosda']['C_dis']['H'][-last_epoch:], c='C7', label='protModel-Cdis')
        ax3.axhline(y=self.prot_info['srosda']['C_dis']['H'][0], c='C7', linestyle='--', label='prot_init-Cdis')

        ax3.legend(ncol=2)
        ax3.set_title("H")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(path + '/sr-osda.pdf')

        # 3. plot att
        # cls-wise: (precision, recall, F1, avg_pre, avg_rec, avg_f1)
        # sample-wise: (sample_avg_pre, sample_avg_rec, sample_avg_f1)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        # 3.1 cls-wise avg
        ax1 = axes[0]
        ax1.plot(self.base_info['att']['Avg_Precision'][-last_epoch:], c='C0', label='baseModel-P')
        ax1.axhline(y=self.base_info['att']['Avg_Precision'][0], c='C0', linestyle='--', label='base_P')
        ax1.plot(self.base_info['att']['Avg_Recall'][-last_epoch:], c='C1', label='baseModel-R')
        ax1.axhline(y=self.base_info['att']['Avg_Recall'][0], c='C1', linestyle='--', label='base_R')
        ax1.plot(self.base_info['att']['Avg_F1'][-last_epoch:], c='C2', label='baseModel-F1')
        ax1.axhline(y=self.base_info['att']['Avg_F1'][0], c='C2', linestyle='--', label='base_F1')

        ax1.plot(self.prot_info['att']['Avg_Precision'][-last_epoch:], c='C5', label='protModel-P')
        ax1.axhline(y=self.prot_info['att']['Avg_Precision'][0], c='C5', linestyle='--', label='prot_init-P')
        ax1.plot(self.prot_info['att']['Avg_Recall'][-last_epoch:], c='C6', label='protModel-R')
        ax1.axhline(y=self.prot_info['att']['Avg_Recall'][0], c='C6', linestyle='--', label='prot_init-R')
        ax1.plot(self.prot_info['att']['Avg_F1'][-last_epoch:], c='C7', label='protModel-F1')
        ax1.axhline(y=self.prot_info['att']['Avg_F1'][0], c='C7', linestyle='--', label='prot_init-F1')

        ax1.legend(ncol=2)
        ax1.set_title("Class-wise Avg")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("P/R/F1")
        ax1.set_ylim(0, 1)

        # 3.2 sample-wise avg
        ax2 = axes[1]
        ax2.plot(self.base_info['att']['Avg_sample_Precision'][-last_epoch:], c='C0', label='baseModel-P')
        ax2.axhline(y=self.base_info['att']['Avg_sample_Precision'][0], c='C0', linestyle='--', label='base_P')
        ax2.plot(self.base_info['att']['Avg_sample_Recall'][-last_epoch:], c='C1', label='baseModel-R')
        ax2.axhline(y=self.base_info['att']['Avg_sample_Recall'][0], c='C1', linestyle='--', label='base_R')
        ax2.plot(self.base_info['att']['Avg_sample_F1'][-last_epoch:], c='C2', label='baseModel-F1')
        ax2.axhline(y=self.base_info['att']['Avg_sample_F1'][0], c='C2', linestyle='--', label='base_F1')

        ax2.plot(self.prot_info['att']['Avg_sample_Precision'][-last_epoch:], c='C5', label='protModel-P')
        ax2.axhline(y=self.prot_info['att']['Avg_sample_Precision'][0], c='C5', linestyle='--', label='prot_init-P')
        ax2.plot(self.prot_info['att']['Avg_sample_Recall'][-last_epoch:], c='C6', label='protModel-R')
        ax2.axhline(y=self.prot_info['att']['Avg_sample_Recall'][0], c='C6', linestyle='--', label='prot_init-R')
        ax2.plot(self.prot_info['att']['Avg_sample_F1'][-last_epoch:], c='C7', label='protModel-F1')
        ax2.axhline(y=self.prot_info['att']['Avg_sample_F1'][0], c='C7', linestyle='--', label='prot_init-F1')

        ax2.legend(ncol=2)
        ax2.set_title("Sample-wise Avg")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("P/R/F1")
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(path + '/att.pdf')

        # 4. attributes element P/R/F1
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
        width = 0.4
        best_f1_epoch = np.argmax(self.base_info['att']['Avg_F1'])
        best_P_base = self.base_info['att']['Precision'][best_f1_epoch]
        best_R_base = self.base_info['att']['Recall'][best_f1_epoch]
        best_F1_base = self.base_info['att']['F1'][best_f1_epoch]

        best_f1_epoch = np.argmax(self.prot_info['att']['Avg_F1'])
        best_P_prot = self.prot_info['att']['Precision'][best_f1_epoch]
        best_R_prot = self.prot_info['att']['Recall'][best_f1_epoch]
        best_F1_prot = self.prot_info['att']['F1'][best_f1_epoch]

        # 4.1 Precision
        ax1 = axes[0]
        ax1.bar(np.arange(self.args.att_dim)-width/2, width=width, height=best_P_base, color='C0', label='baseModel')
        ax1.bar(np.arange(self.args.att_dim)+width/2, width=width, height=best_P_prot, color='C5', label='protModel')
        ax1.legend()
        ax1.set_title("Precision")
        ax1.set_xlabel("Attributes")
        ax1.set_ylabel("Precision")
        # ax1.set_ylim(0, 1)

        # 4.2 Recall
        ax2 = axes[1]
        ax2.bar(np.arange(self.args.att_dim)-width/2, width=width, height=best_R_base, color='C0', label='baseModel')
        ax2.bar(np.arange(self.args.att_dim)+width/2, width=width, height=best_R_prot, color='C5', label='protModel')
        ax2.legend()
        ax2.set_title("Recall")
        ax2.set_xlabel("Attributes")
        ax2.set_ylabel("Recall")
        # ax2.set_ylim(0, 1)

        # 4.3 F1
        ax3 = axes[2]
        ax3.bar(np.arange(self.args.att_dim)-width/2, width=width, height=best_F1_base, color='C0', label='baseModel')
        ax3.bar(np.arange(self.args.att_dim)+width/2, width=width, height=best_F1_prot, color='C5', label='protModel')
        ax3.legend()
        ax3.set_title("F1")
        ax3.set_xlabel("Attributes")
        ax3.set_ylabel("F1")
        # ax3.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(path + '/att_elements.pdf')

        print("Plot finished!")

    def test_on_loader(self, dataloader, model, attri_mat, branch='both', args=None):
        epoch_record = {
            'y_osda': [], # open-set Ks+1 ground-truth lbls
            'y_all': [], # all ground-truth lbls
            'att_gt': [],  # nt x att_dim (eg., nt x 85)

            # base
            'att_base_pred': [],  # nt x att_dim, (save probability)
            'y_base_pred': [],  # (nt, ), [1, ..., Ks+1]
            'y_dis_base_pred': [],  # (nt, ), {0, 1}, 0=seen, 1=unseen
            'att_sim_base': [],

            # proto
            'att_prot_pred': [],  # nt x att_dim, (save probability)
            'y_prot_pred': [],
            'y_dis_prot_pred': [],
            'att_sim_prot': []

        }

        print('\nTesting...')
        model.netF.eval()
        model.netB.eval()
        model.netA.eval()
        model.netC_all.eval()
        model.netD.eval()
        model.ppnet.eval()
        model.protModel_C.eval()
        model.protModel_D.eval()


        for batch_idx, batch in enumerate(dataloader):
            img = batch[0].cuda()
            lb = batch[1]  # Ks+1, open-set lbls
            ori_lb = batch[2]  # Ks+Kt, sr-osda lbls

            res_feat, res_featmap = model.netF(img)

            # 1. baseModel
            feat = model.netB(res_feat)
            attri_pred, _ = model.netA(feat)  # [bs, 85]

            feat_joint = torch.cat([feat, attri_pred], dim=1)  # [bs, 256+85]
            # 1.1 open-set classifier (Ks+1)
            logits = model.netC_all(feat_joint)  # [bs, K+1]
            pred_lb = logits.argmax(dim=1)


            # 1.2 sr-osda
            logits_dis = model.netD(feat_joint)  # [Nt, 2], 1 is shared, 0 is unknown
            pred_dis = logits_dis.argmax(dim=1)

            if self.sim_type == 'cosine':
                att_sim = 1 - distance(attri_pred.detach().cpu().numpy(), attri_mat, metric=self.sim_type) # nt x (Ks+Kt)
            else:
                raise Exception("The distance type is not defined!")


            # 2. protoModel
            t_att_logits, t_open_cls_logits, t_min_distances, t_conv_feats = model.ppnet(res_featmap)
            # ppnet_attri_pred = torch.sigmoid(t_att_logits)

            if self.args.ppnet_last_layer_type == 'softmax':
                all_pred_attri_logits = t_open_cls_logits # (ns+nt) x 85 x 2
                all_prob_attri = torch.softmax(all_pred_attri_logits, dim=2)
                ppnet_attri_pred = all_prob_attri[:, :, 1] # positive probability, 1 prob.
                # all_pred_att_onehot = torch.argmax(all_prob_attri, dim=2) # 0 means negative, 1 for positive
            elif self.args.ppnet_last_layer_type == 'sigmoid':
                all_pred_attri_logits = t_att_logits
                ppnet_attri_pred = torch.sigmoid(all_pred_attri_logits)
            else:
                raise Exception("The ppnet last layer type is not correctly defined!")

            ppnet_feat_joint = torch.cat([feat, ppnet_attri_pred], dim=1)
            # 2.1 open-set classifier (Ks+1)
            # ppnet_logits = model.netC_all(ppnet_feat_joint)  # [bs, K+1]
            ppnet_logits = model.protModel_C(ppnet_feat_joint)  # [bs, K+1]
            ppnet_pred_lb = ppnet_logits.argmax(dim=1)

            # 2.2 sr-osda
            # ppnet_logits_dis = model.netD(ppnet_feat_joint)  # [Nt, 2], 1 is shared, 0 is unknown
            ppnet_logits_dis = model.protModel_D(ppnet_feat_joint)  # [Nt, 2], 1 is shared, 0 is unknown

            ppnet_pred_dis = ppnet_logits_dis.argmax(dim=1)

            if self.sim_type == 'cosine':
                ppnet_att_sim = 1 - distance(ppnet_attri_pred.detach().cpu().numpy(), attri_mat, metric=self.sim_type) # nt x (Ks+Kt)
            else:
                raise Exception("The distance type is not defined!")


            # 3. record batch output
            epoch_record['y_osda'].extend(lb.detach().cpu().numpy())
            epoch_record['y_all'].extend(ori_lb.detach().cpu().numpy())
            epoch_record['att_gt'].extend(attri_mat[ori_lb])

            epoch_record['att_base_pred'].extend(attri_pred.detach().cpu().numpy()) # probability output, after sigmoid
            epoch_record['y_base_pred'].extend(pred_lb.detach().cpu().numpy()) # argmax
            epoch_record['y_dis_base_pred'].extend(pred_dis.detach().cpu().numpy()) # argmax, 0:seen 1:unseen
            epoch_record['att_sim_base'].extend(att_sim)

            epoch_record['att_prot_pred'].extend(ppnet_attri_pred.detach().cpu().numpy())
            epoch_record['y_prot_pred'].extend(ppnet_pred_lb.detach().cpu().numpy())
            epoch_record['y_dis_prot_pred'].extend(ppnet_pred_dis.detach().cpu().numpy())

            epoch_record['att_sim_prot'].extend(ppnet_att_sim)


        return epoch_record