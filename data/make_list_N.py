"""This script is used for creating DomainNet, AwA2, and 3D2 datasets and Attribute files"""
import os
import csv
import numpy as np


# Create AwA2 full txt

root_path = '/home/scott/Work/Dataset/'
domain = 'Painting'

# Create DomainNet full txt

ori_dir = root_path + 'DomainNet/images/real' #'/media/room/date/DataSets/domain_net/painting'

tgt_txt_path = './data/DomainNet/' + domain + '_full.txt'

with open(tgt_txt_path, mode='w') as f:
    cls = 0
    for root, folders, _ in os.walk(ori_dir):
        for folder in folders:
            class_folder_path = os.path.join(root, folder)
            for _, _, imgs in os.walk(class_folder_path):
                for img in imgs:
                    cur_img_abs_path = os.path.join(class_folder_path, img)
                    line = cur_img_abs_path + ' ' + str(cls) + '\n'
                    f.write(line)

            cls += 1



# Create DomainNet-AwA2 shared class sample list
src_txt_path = './data/DomainNet/' + domain + '_full.txt'
tgt_txt_path_src = './data/DomainNet/' + domain + '_src_0-9.txt'
tgt_txt_path_tgt = './data/DomainNet/' + domain + '_tgt_0-16.txt'


shared_cls = ['bat', 'cow', 'dolphin', 'elephant', 'giraffe', 'horse', 'lion', 'mouse', 'panda', 'pig']  # 0-9
open_cls = ['rabbit', 'raccoon', 'rhinoceros', 'sheep', 'squirrel', 'tiger', 'zebra']  # 10-16

shared_cls_dict = {}
open_cls_dict = {}

for i, c in enumerate(shared_cls):
    shared_cls_dict[c] = str(i)

for i, c in enumerate(open_cls):
    open_cls_dict[c] = str(i+10)

f = open(src_txt_path, 'r')
lines = f.readlines()
f.close()

fs = open(tgt_txt_path_src, 'w')
ft = open(tgt_txt_path_tgt, 'w')

for l in lines:
    cls = l.split(' ')[0].split('/')[-2]
    if cls in shared_cls:
        wl = l.split(' ')[0] + ' ' + shared_cls_dict[cls] + '\n'
        fs.write(wl)
        ft.write(wl)
    elif cls in open_cls:
        wl = l.split(' ')[0] + ' ' + open_cls_dict[cls] + '\n'
        ft.write(wl)

fs.close()
ft.close()