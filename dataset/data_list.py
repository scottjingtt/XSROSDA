from PIL import Image
from torch.utils.data import Dataset
import os
import os.path
from torchvision import transforms
import random


def make_dataset(list_path):
    f = open(list_path, 'r')
    lines = f.readlines()
    f.close()

    img_list = []
    lb_list = []

    for l in lines:
        img_list.append(l.strip().split(' ')[0])
        lb_list.append(int(l.strip().split(' ')[1]))

    return img_list, lb_list


def rgb_loader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except:
        raise Exception("Error: " + path)

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageSet(Dataset):
    def __init__(self, list_path, args=None, train=False, pseudo_lb=None, balanced=False, mode='RGB'):
        self.args = args
        imgs, lb = make_dataset(list_path)

        if len(imgs) == 0:
            raise(RuntimeError("No images in the data_list"))

        self.imgs = imgs    # list of str path
        self.lb = lb        # list of int64

        self.shared_cls_num = args.shared_class_num
        self.total_cls_num = args.total_class_num

        # Ks + 1 open-set lbls
        self.modify_lb = [i if i < self.shared_cls_num else self.shared_cls_num for i in self.lb]

        self.pseudo_lb = pseudo_lb  # None or a list

        if train:
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            resize_size = 256
            crop_size = 224
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.balance_sample = balanced
        self.cls_num = len(list(set(self.lb)))
        self.class_dict = self._get_class_dict()

    def _get_class_dict(self):
        # return a dict = {'lbl': idx of images in the image list}
        class_dict = dict()
        for i, cur_lb in enumerate(self.lb):
            if not cur_lb in class_dict.keys():
                class_dict[cur_lb] = []
            class_dict[cur_lb].append(i)
        return class_dict


    def __getitem__(self, index):
        if self.balance_sample:
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)

        img_path, lb, ori_lb = self.imgs[index], self.modify_lb[index], self.lb[index]

        if self.pseudo_lb is not None:
            pseudo_lb = self.pseudo_lb[index]

        img_path = os.path.join(self.args.dataroot, '/'.join(img_path.split('/')[5:]))
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)


        if self.pseudo_lb is None:
            return img, lb, ori_lb
        else:
            return img, lb, ori_lb, pseudo_lb

    def __len__(self):
        return len(self.imgs)
