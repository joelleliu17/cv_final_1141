import os
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torchvision
import random
import collections
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision.transforms as transforms  
import pdb  


# preprocessing
transform = transforms.Compose([             
              transforms.ToTensor(), #Convert image to tensor. 
              transforms.Normalize(                      
              mean=[0.485, 0.456, 0.406],   # Subtract mean 
              std=[0.229, 0.224, 0.225]     # Divide by standard deviation   
              )])

def default_loader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_arr = np.array(img_pil)
    width, height, _ = img_arr.shape   # 記錄圖片長寬
    if width > height:
        img_pil = img_pil.rotate(-90, expand=True)  # 將圖片統一轉為橫的
    img_pil = img_pil.resize((270, 405)) # height=270, width=405 , ori: 270, 405
    img_tensor = transform(img_pil)
    return img_tensor



class ADAPT_WHITEBAIT(Dataset):
    def __init__(self, datapath, loader=default_loader):
        root = datapath    # dataset path

        label_name = list(os.listdir(root))
        label_name.sort()
        path = [ root + label_name[i] for i in range(len(label_name)) ] # 每個類別的root
        
        num_per_cls = []    # num of each class len=32 or 34
        data_path = []      # all the sample path_name list len=28190
        data_label = []     # all the sample label list len=28190
        data_sum = 0 # 總資料量
        for i in range(len(path)):
            # print(path[i])
            data = os.listdir(path[i]) # 每個類別資料夾內的data
            num_per_cls.append(len(data))
            data_path[data_sum:(data_sum+num_per_cls[i])] = [ list([path[i] + '/']*len(data))[j] + data[j] for j in range(len(data))] # 獲得每個類別每張img的path
            # print(data_path)
            data_label += len(data) * [i]
            data_sum += len(data)
            
        print('Num per cls: ', num_per_cls)
        print('Total num: ', len(data_label))

        self.images_ = data_path       
        self.target_ = data_label
        self.loader = loader
        self.num_classes = len(label_name)
        self.label_name = label_name
        

    def __getitem__(self, index):
        try:
            fn_train = self.images_[index]
            images_train = self.loader(fn_train)
            target_train = self.target_[index]
    
            return images_train, target_train
        except RecursionError:
            print(f"RecursionError at index {index}. Please check `self.loader` and `transform`.")
            raise
        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise

    def __len__(self):
        return len(self.images_)


    def get_gt_label(self):
        return self.target_valid_, self.target_test_      



################################################## CIFAR10 / CIFAR100 ##################################################

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        self.img_num_per_cls = img_num_per_cls
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        self.cls_num_list = cls_num_list
        return cls_num_list
    def get_group_list(self):
        g0, g1, g2, g3 = [], [], [], []
        for i in range(len(self.cls_num_list)):
            if self.cls_num_list[i] < 200:
                g3.append(i)
            elif self.cls_num_list[i] >= 200 and self.cls_num_list[i] < 1000:
                g2.append(i)
            elif self.cls_num_list[i] >= 1000 and self.cls_num_list[i] < 3000:
                g1.append(i)
            else:
                g0.append(i)
        return [g0, g1, g2, g3]

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100
    def get_group_list(self):
        g0, g1, g2, g3 = [], [], [], []
        for i in range(len(self.cls_num_list)):
            if self.cls_num_list[i] < 125:
                g3.append(i)
            elif self.cls_num_list[i] >= 125 and self.cls_num_list[i] < 250:
                g2.append(i)
            elif self.cls_num_list[i] >= 250 and self.cls_num_list[i] < 375:
                g1.append(i)
            else:
                g0.append(i)
        return [g0, g1, g2, g3]
