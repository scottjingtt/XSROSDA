"""This script is used for creating DomainNet, AwA2, and 3D2 datasets and Attribute files"""
import os
import csv
import numpy as np


# Create AwA2 full txt

root_path = '/home/scott/Work/Dataset/'

ori_dir = root_path + 'Animals_with_Attributes2/images' #/media/room/date/DataSets/Animals_with_Attributes2/JPEGImages'

tgt_txt_path = './data/AwA2/AwA2_full.txt'

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
src_txt_path = './data/AwA2/AwA2_full.txt'
tgt_txt_path_src = './data/AwA2/AwA2_src_0-9.txt'
tgt_txt_path_tgt = './data/AwA2/AwA2_tgt_0-16.txt'


shared_cls = ['bat', 'cow', 'dolphin', 'elephant', 'giraffe', 'horse', 'lion', 'mouse', 'giant+panda', 'pig']  # 0-9
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



# Create I-AwA2 shared class sample list
src_txt_path = './data/AwA2/AwA2_full.txt'
tgt_txt_path_src = './data/AwA2/AwA2_src_0-39.txt'
tgt_txt_path_tgt = './data/AwA2/AwA2_tgt_0-49.txt'

share_cls_txt = './data/AwA2/shr_classes.txt'
open_cls_txt = './data/AwA2/unk_classes.txt'

shared_cls = []
open_cls = []

f = open(share_cls_txt, 'r')
lines = f.readlines()
for l in lines:
    shared_cls.append(l.strip())
f.close()

f = open(open_cls_txt, 'r')
lines = f.readlines()
for l in lines:
    open_cls.append(l.strip())
f.close()

shared_cls_dict = {}  # 40 knw class: 0--39
for i, c in enumerate(shared_cls):
    shared_cls_dict[c] = str(i)

open_cls_dict = {}  # 10 open class: 40--49
for i, c in enumerate(open_cls):
    open_cls_dict[c] = str(i+40)

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



# Create DomainNet full txt

ori_dir = root_path + 'DomainNet/images/real' #'/media/room/date/DataSets/domain_net/painting'

tgt_txt_path = './data/DomainNet/Real_full.txt'

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
src_txt_path = './data/DomainNet/Real_full.txt'
tgt_txt_path_src = './data/DomainNet/Real_src_0-9.txt'
tgt_txt_path_tgt = './data/DomainNet/Real_tgt_0-16.txt'


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


# Create DomainNet-AwA2 shared class Attributes file (17 class version)
attri_file = './data/AwA2/att_bi.csv'
attri_tgt = './data/AwA2/att_17.txt'

shared_cls = ['bat', 'cow', 'dolphin', 'elephant', 'giraffe', 'horse', 'lion', 'mouse', 'giant+panda', 'pig']  # 0-9
open_cls = ['rabbit', 'raccoon', 'rhinoceros', 'sheep', 'squirrel', 'tiger', 'zebra']  # 10-16

all_cls = shared_cls + open_cls

f1 = open(attri_tgt, 'w')

for c in all_cls:
    # If csvfile is a file object, it should be opened with newline=''
    with open(attri_file, mode='r', newline='') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row[''] == c:
                line = ''
                for k, v in row.items():
                    if k == '':
                        continue
                    else:
                        line += v
                        line += ' '
                line += '\n'
                f1.write(line)
                break

f1.close()


# Create I-AwA2 shared class Attributes file (50 class version)
attri_file = './data/AwA2/att_bi.csv'
attri_tgt = './data/AwA2/att_50.txt'

share_cls_txt = './data/AwA2/shr_classes.txt'
open_cls_txt = './data/AwA2/unk_classes.txt'

shared_cls = []
open_cls = []

f = open(share_cls_txt, 'r')
lines = f.readlines()
for l in lines:
    shared_cls.append(l.strip())
f.close()

f = open(open_cls_txt, 'r')
lines = f.readlines()
for l in lines:
    open_cls.append(l.strip())
f.close()

all_cls = shared_cls + open_cls  # 40 + 10 classes

f1 = open(attri_tgt, 'w')

for c in all_cls:
    with open(attri_file, mode='r', newline='') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            if row[''] == c:
                line = ''
                for k, v in row.items():
                    if k == '':
                        continue
                    else:
                        line += v
                        line += ' '
                line += '\n'
                f1.write(line)
                break

f1.close()



# Create 3D2 full txt
ori_dir = root_path + '3D2/images' #/media/room/date/DataSets/3D2'
tgt_txt_path = './data/3D2/3D2_full.txt'

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



# Create 3D2-AwA2's 3D2 source sample list
src_txt_path = './data/3D2/3D2_full.txt'
tgt_txt_path_src = './data/3D2/3D2_src_0-39.txt'

share_cls_txt = './data/AwA2/shr_classes.txt'

shared_cls = []

f = open(share_cls_txt, 'r')
lines = f.readlines()
for l in lines:
    shared_cls.append(l.strip())
f.close()


shared_cls_dict = {}  # 40 knw class: 0--39
for i, c in enumerate(shared_cls):
    shared_cls_dict[c] = str(i)

f = open(src_txt_path, 'r')
lines = f.readlines()
f.close()

fs = open(tgt_txt_path_src, 'w')

for l in lines:
    cls = l.split(' ')[0].split('/')[-2].split('-')[0]
    if cls in shared_cls:
        wl = l.split(' ')[0] + ' ' + shared_cls_dict[cls] + '\n'
        fs.write(wl)

fs.close()




# Create attributes .mat file
attri_src = './data/AwA2/att_50.txt'
out_f = './data/AwA2/att_50.npy'

num = 50
attri = np.zeros((num, 85))

f = open(attri_src, 'r')
lines = f.readlines()
f.close()

for i, l in enumerate(lines):
    li = l.strip().split(' ')
    for j, v in enumerate(li):
        attri[i, j] = float(v)

np.save(out_f, attri)


