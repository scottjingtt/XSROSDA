import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

class AverageMeter(object):

    def __init__(self, num_class):
        self.num_class = num_class

        self.avg = np.zeros(num_class)
        self.sum = np.zeros(num_class)
        self.count = np.zeros(num_class)

    def reset(self):

        self.avg = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)
        self.count = np.zeros(self.num_class)

    def update(self, gt_lb, pred_lb, n=1):
        for i, value in enumerate(gt_lb):
            # value: cls id; val[i]: if prediction correct: 1, else: 0
            self.sum[value] += pred_lb[i] * n  # total number of correct prediction
            self.count[value] += n # total number of test samples
            self.avg[value] = self.sum[value] / self.count[value]  # acc of each class; updat .avg averagey time .sum and .count are chagned.

class Logger(object):
    def __init__(self, name, value_name_list, save_path, plot_scale=None):
        self.name = name
        self.value_dict = {}
        for v in value_name_list:
            self.value_dict[v] = []

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.system('mkdir -p ' + self.save_path)

        self.plot_scale = plot_scale

    def add_record(self, value_list):
        cnt = 0
        for k in self.value_dict.keys():
            self.value_dict[k].append(value_list[cnt])
            cnt += 1

    def save_record(self):
        key_list = []
        for v in self.value_dict.keys():
            x_len = len(self.value_dict[v])
            key_list.append(v)

        color_list = ['#800000', '#469990', '#911eb4', '#bfef45']
        x = [i+1 for i in range(x_len)]

        fig = plt.figure(1)

        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        if self.plot_scale is not None:
            axes.set_ylim(self.plot_scale)
        else:
            axes.set_ylim([-2, 5])

        for i in range(len(key_list)):
            cur_key = key_list[i]
            plt.plot(x, self.value_dict[cur_key], color=color_list[i%5], linestyle='-', label=cur_key,
                     linewidth=2.3)

        plt.title(self.name)
        plt.legend(prop={'size': 12})

        plt.savefig(osp.join(self.save_path, self.name+'.png'))
        plt.close()

        torch.save(self.value_dict, osp.join(self.save_path, self.name+'.pt'))
