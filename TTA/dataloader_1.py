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
          #    transforms.RandomRotation(),
          #    transforms.CenterCrop(224),
              # transforms.Pad(padding=[68, 0, 68, 0], fill=0, padding_mode='constant'),   
              transforms.ToTensor(), #Convert image to tensor. 
              transforms.Normalize(                      
              mean=[0.485, 0.456, 0.406],   # Subtract mean 
              std=[0.229, 0.224, 0.225]     # Divide by standard deviation   
          #    mean=[0.5, 0.5, 0.5],   # Subtract mean 
          #    std=[0.5, 0.5, 0.5]     # Divide by standard deviation
              )])

def default_loader(path):
    img_pil =  Image.open(path)
    img_arr = np.array(img_pil)
    width, height, _ = img_arr.shape   # 記錄圖片長寬
    if width > height:
        img_pil = img_pil.rotate(-90, expand=True)  # 將圖片統一轉為橫的
    img_pil = img_pil.resize((270, 405)) # height=270, width=405 , ori: 270, 405
#    img_pil = img_pil.resize((299,299))  # inception v3 專用
    img_tensor = transform(img_pil)
    return img_tensor

# def default_loader(path):
#     img_pil =  Image.open(path)
#     img_arr = np.array(img_pil)
#     width, height, _ = img_arr.shape   # 記錄圖片長寬
#     if width > height:
#         img_pil = img_pil.rotate(-90, expand=True)  # 將圖片統一轉為直的
#     img_pil = img_pil.resize((406, 406)) # width=270, height=405
#     img_pil = ImageOps.expand(img_pil, border=(68, 0, 68, 0), fill=0)
#     #plt.imshow(img_pil)
#     #plt.savefig('./123.png')
# #    img_pil = img_pil.resize((299,299))  # inception v3 專用
#     img_tensor = transform(img_pil)
#     return img_tensor


class IMBALANCED_WHITEBAIT(Dataset):
    def __init__(self, dataset, loader=default_loader):
        if dataset == '28k':
            path = '/Users/User/Desktop/datasets/RawData_28k_rename/'    # dataset path
            # path = '../BBN_V3/datasets/RawData_28k_rename/'
            #  path = '/home/nvlab110/SDB1/nvlab110hao/whitebait/RawData_28k_rename/'
        elif dataset == '34k':
            path = './original34k_rename/'    # dataset path
        else:
            print('only "28k" and "34k" in whitebait dataset.')
        

        label_name = list(os.listdir(path))
        label_name.sort()
        path = [ path + label_name[i] for i in range(len(label_name)) ] # 每個類別的root
        
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
        
        train_num_per_cls = []
        img_train = []
        img_valid = []
        img_test = []
        label_train = []
        label_valid = []
        label_test = []
        num_train, num_valid, num_test, data_sum = 0, 0, 0, 0
       
        for i in range(len(num_per_cls)):  # 針對每一個類別
            data_x = data_path[data_sum:(data_sum+num_per_cls[i])]
            data_y = data_label[data_sum:(data_sum+num_per_cls[i])]
            
            # train_test_split 應解包為四個變數：訓練集、測試集和對應的標籤
            tv_img, test_img, tv_label, test_label = train_test_split(
                data_x, data_y, random_state=777, test_size=1/6
            )  # 分割為訓練/驗證集及測試集
        
            train_img, valid_img, train_label, valid_label = train_test_split(
                tv_img, tv_label, random_state=777, test_size=1/5
            )  # 再次分割為訓練及驗證集

            
            img_train[num_train:num_train+len(train_img)] = train_img
            label_train[num_train:num_train+len(train_label)] = train_label
            img_valid[num_valid:num_valid+len(valid_img)] = valid_img
            label_valid[num_valid:num_valid+len(valid_label)] = valid_label
            img_test[num_test:num_test+len(test_img)] = test_img
            label_test[num_test:num_test+len(test_label)] = test_label
            
            train_num_per_cls.append(len(train_label))
            data_sum += len(data_x)
            num_train += len(train_label)
            num_valid += len(valid_label)
            num_test += len(test_label)
        
        self.images_train_ = img_train
        self.target_train_ = label_train
        self.images_valid_ = img_valid
        self.target_valid_ = label_valid
        self.images_test_ = img_test       
        self.target_test_ = label_test
        self.loader = loader
        self.num_classes = len(label_name)
        self.train_num_per_cls = train_num_per_cls
        self.label_name = label_name
        
        # print(self.target_test_)
        self.class_weight = [ (1/(i*self.num_classes)) for i in num_per_cls] # 與BBN不同!!!!!
        # self.class_weight = [ round(max(train_num_per_cls)/i) for i in train_num_per_cls] # BBN
        # print(self.class_weight)
        self.sum_weight = sum(self.class_weight)


        self.class_dict = dict() # 同BBN
        for i in range(len(label_train)):
            cat_id = label_train[i]
            if not cat_id in self.class_dict:
                self.class_dict[cat_id] = []
            self.class_dict[cat_id].append(i)


        _, _, self.g = self.get_label_list(dataset)
        self.all_target_group_idx_train = []
        for i in tqdm(range(len(self.target_train_))):
            each_target_group_idx = [0,0,0,0] # [0, 0, 123, 0]
            for j in range(len(self.g)):
                if self.target_train_[i] in self.g[j]:
                    #self.all_target_group.append(j)
                    each_target_group_idx[j] = self.g[j].index(self.target_train_[i]) + 1 # 含other
                    self.all_target_group_idx_train.append(each_target_group_idx)
        
        self.all_target_group_idx_val = []
        for i in tqdm(range(len(self.target_valid_))):
            each_target_group_idx = [0,0,0,0] # [0, 0, 123, 0]
            for j in range(len(self.g)):
                if self.target_valid_[i] in self.g[j]:
                    #self.all_target_group.append(j)
                    each_target_group_idx[j] = self.g[j].index(self.target_valid_[i]) + 1 # 含other
                    self.all_target_group_idx_val.append(each_target_group_idx)
        
        self.all_target_group_idx_test = []
        for i in tqdm(range(len(self.target_test_))):
            each_target_group_idx = [0,0,0,0] # [0, 0, 123, 0]
            for j in range(len(self.g)):
                if self.target_test_[i] in self.g[j]:
                    #self.all_target_group.append(j)
                    each_target_group_idx[j] = self.g[j].index(self.target_test_[i]) + 1 # 含other
                    self.all_target_group_idx_test.append(each_target_group_idx)

        ######### 0810 only head shot
        
        # print(self.g)
        # self.images_train_head = []
        # self.target_train_head = []
        # self.all_target_group_idx_train_head = []
        # for i in range(len(self.target_train_)):
        #     each_target_group_idx = [0,0,0,0] # [0, 0, 123, 0]
        #     if self.target_train_[i] in self.g[0]:
        #         #self.all_target_group.append(j)
        #         self.images_train_head.append(self.images_train_[i])
        #         self.target_train_head.append(self.target_train_[i])
        #         each_target_group_idx[0] = self.g[0].index(self.target_train_[i]) + 1 # 含other
        #         self.all_target_group_idx_train_head.append(each_target_group_idx)
        # #breakpoint()
        print('Len: {}'.format(len(self.images_train_)))

    #def __getitem__(self, index):
        #fn_train = self.images_train_[index]
        #images_train = self.loader(fn_train)
        #target_train = self.target_train_[index]
        #target_group_idx = torch.tensor(self.all_target_group_idx_train[index])
    def __getitem__(self, index):
        try:
            fn_train = self.images_train_[index]
            images_train = self.loader(fn_train)
            target_train = self.target_train_[index]
            target_group_idx = torch.tensor(self.all_target_group_idx_train[index])
    
            return images_train, target_train, target_group_idx
        except RecursionError:
            print(f"RecursionError at index {index}. Please check `self.loader` and `transform`.")
            raise
        except Exception as e:
            print(f"Error at index {index}: {e}")
            raise


        ######### 0810 only head 
        # fn_train = self.images_train_head[index]
        # images_train = self.loader(fn_train)
        # target_train = self.target_train_head[index]
        # target_group_idx = torch.tensor(self.all_target_group_idx_train_head[index])

        #########

        ############################################## meta is here ##############################################
        meta = dict()
        # only g2 & g3
        ######## 學姐sample方法
        #_, _, g = self.get_label_list('28k')
        #sample_class = g[2] + g[3] # 類別數最少的兩個set
        #sample_seed = random.randint(0, len(sample_class)-1)
        #sample_indexes = self.class_dict[sample_class[sample_seed]]
        #sample_index = random.choice(sample_indexes)
        #sample_img, sample_label = self.images_train_[sample_index], self.target_train_[sample_index]
        ######## BBN sample 方法
        # print(self.sample_class_index_by_weight())
        sample_class = self.sample_class_index_by_weight()
        sample_indexes = self.class_dict[sample_class]
        sample_index = random.choice(sample_indexes)
        sample_img, sample_label = self.images_train_[sample_index], self.target_train_[sample_index]

        sample_img = self.loader(sample_img)
        meta['sample_image'] = sample_img
        meta['sample_label'] = sample_label
        meta['group_label'] = torch.tensor(self.all_target_group_idx_train[sample_index])
        """
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_train
        meta['sample_label'] = target_train
        """
        ############################################## meta is above ##############################################
        
        return images_train, target_train, target_group_idx, meta
    
    def __len__(self):
        return len(self.images_train_)
        #return len(self.images_train_head)

    def get_bsce_weight(self):
        num_list = [0] * self.num_classes
        print("Weight List has been produced")
        for label in self.target_train_:
            num_list[label] += 1

        return num_list
    
    def get_label_list(self, dataset): # 所有類別名稱, train set中各類別數量, 照各類別數量分group                                                   # how is the group when 4:1:1 ?
        if dataset == '28k':
            # path = './datasets/RawData_28k_1116/train/'
             path = '/home/nvlab110/SDB1/nvlab110hao/whitebait/RawData_28k_rename/'
        elif dataset == '34k':
            path = 'D:/00_Yume/original34k/train/'
        label_name = list(os.listdir(path))
        label_name.sort()
        path = [ path + label_name[i] for i in range(len(label_name)) ]
        num_per_cls = []    # num of each class len=32 or 34
        for i in range(len(path)):
            data = os.listdir(path[i])
            num_per_cls.append(len(data))
        self.num_per_cls = num_per_cls
        g0, g1, g2, g3 = [], [], [], []
        g0_num, g1_num, g2_num, g3_num = [], [], [], []
        num_per_group = [0,0,0,0]
        # for i in range(len(self.num_per_cls)): # 根據每個類別的數量將類別分組
        #     if self.num_per_cls[i] < 101:
        #         g3.append(i)
        #         num_per_group[3] += self.num_per_cls[i]
        #         g3_num.append(self.num_per_cls[i])
        #     elif self.num_per_cls[i] < 1001 and self.num_per_cls[i] >= 101:
        #         g2.append(i)
        #         g2_num.append(self.num_per_cls[i])
        #         num_per_group[2] += self.num_per_cls[i]
        #     elif self.num_per_cls[i] < 10001 and self.num_per_cls[i] >= 1001:
        #         g1.append(i)
        #         g1_num.append(self.num_per_cls[i])
        #         num_per_group[1] += self.num_per_cls[i]
        #     elif self.num_per_cls[i] >= 10001:
        #         g0.append(i)
        #         g0_num.append(self.num_per_cls[i])
        #         num_per_group[0] += self.num_per_cls[i]
        
        for i in range(len(self.train_num_per_cls)):
            if self.train_num_per_cls[i] <= 20:
                g3.append(i)
                g3_num.append(self.train_num_per_cls[i])
                num_per_group[3] += self.train_num_per_cls[i]
            elif self.train_num_per_cls[i] <= 100 and self.train_num_per_cls[i] >= 21:
                g2.append(i)
                g2_num.append(self.train_num_per_cls[i])
                num_per_group[2] += self.train_num_per_cls[i]
            elif self.train_num_per_cls[i] <= 1000 and self.train_num_per_cls[i] >= 101:
                g1.append(i)
                g1_num.append(self.train_num_per_cls[i])
                num_per_group[1] += self.train_num_per_cls[i]
            else:
                g0.append(i)
                g0_num.append(self.train_num_per_cls[i])
                num_per_group[0] += self.train_num_per_cls[i]
    
        print('{}, {}, {}, {}'.format(max(g0_num)/min(g0_num), max(g1_num)/min(g1_num), max(g2_num)/min(g2_num), max(g3_num)/min(g3_num)))
        g = [g0, g1, g2, g3]
        print('Num per group: ', num_per_group)
        print(g)
        return self.label_name, self.train_num_per_cls, g
    
    def get_majority_minority(self):
        #print(self.train_num_per_cls)
        split = [[], []]
        for i in range(len(self.train_num_per_cls)):
            if self.train_num_per_cls[i] > 100:
                split[0].append(i)
            else:
                split[1].append(i)
            
        return split

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i


class IMBALANCED_WHITEBAIT_VALID(IMBALANCED_WHITEBAIT):

    def __getitem__(self, index_valid):
        fn_valid = self.images_valid_[index_valid]
        images_valid = self.loader(fn_valid)
        target_valid = self.target_valid_[index_valid]
        target_group_idx = torch.tensor(self.all_target_group_idx_val[index_valid])
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_valid
        meta['sample_label'] = target_valid
        
        #return images_valid, target_valid, target_group_idx, meta
        return images_valid, target_valid, target_group_idx
        
    def __len__(self):
        return len(self.images_valid_)
    def get_gt_label(self):
        return self.target_valid_, self.target_test_
    
class IMBALANCED_WHITEBAIT_TEST(IMBALANCED_WHITEBAIT):
    
    def __getitem__(self, index_test):
        fn_test = self.images_test_[index_test]
        images_test = self.loader(fn_test)
        target_test = self.target_test_[index_test]
        target_group_idx = torch.tensor(self.all_target_group_idx_test[index_test])
        
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_test
        meta['sample_label'] = target_test
        #return images_test, target_test, target_group_idx, meta, fn_test.split('/')[-1]
        return images_test, target_test
    
    def __len__(self):
        return len(self.images_test_)
    
   

"""
class IMBALANCED_WHITEBAIT(Dataset):
    def __init__(self, dataset, dual_sample=T rue, dual_sampler_type='reverse', loader=default_loader):
        self.dual_sample = dual_sample
        self.dual_sampler_type = dual_sampler_type
        if dataset == '28k':
            path = 'D:/00_Yume/awData_28k_rename/'    # dataset path
        elif dataset == '34k':
            path = 'D:/00_Yume/original34k_rename/'    # dataset path
        else:
            print('only "28k" and "34k" in whitebait dataset.')
        
        label_name = list(os.listdir(path))
        path = [ path + label_name[i] for i in range(len(label_name)) ]     
        
        num_per_cls = []    # num of each class len=32 or 34
        data_path = []      # all the sample path_name list len=28190
        data_label = []     # all the sample label list len=28190
        data_sum = 0
        for i in range(len(path)):
            data = os.listdir(path[i])
            num_per_cls.append(len(data))
            data_path[data_sum:(data_sum+num_per_cls[i])] = [ list([path[i] + '/']*len(data))[j] + data[j] for j in range(len(data))]
            data_label += len(data) * [i]
            data_sum += len(data)
        
        train_num_per_cls = []
        img_train = []
        img_valid = []
        img_test = []
        label_train = []
        label_valid = []
        label_test = []
        num_train, num_valid, num_test, data_sum = 0, 0, 0, 0
        for i in range(len(num_per_cls)):
            data_x = data_path[data_sum:(data_sum+num_per_cls[i])]
            data_y = data_label[data_sum:(data_sum+num_per_cls[i])]
            
            tv_img, test_img, tv_label, test_label = train_test_split(data_x, data_y, random_state=777, test_size=1/6)
            train_img, valid_img, train_label, valid_label = train_test_split(tv_img, tv_label, random_state=777, test_size=1/5)
            
            img_train[num_train:num_train+len(train_img)] = train_img
            label_train[num_train:num_train+len(train_label)] = train_label
            img_valid[num_valid:num_valid+len(valid_img)] = valid_img
            label_valid[num_valid:num_valid+len(valid_label)] = valid_label
            img_test[num_test:num_test+len(test_img)] = test_img
            label_test[num_test:num_test+len(test_label)] = test_label
            
            train_num_per_cls.append(len(train_label))
            data_sum += len(data_x)
            num_train += len(train_label)
            num_valid += len(valid_label)
            num_test += len(test_label)
        
        if self.dual_sample:
            # class weight
            self.class_weight = [max(train_num_per_cls) / i for i in train_num_per_cls]
            self.sum_weight = sum(self.class_weight)
        
            # class dict
            self.class_dict = dict()
            for i in range(len(label_train)):
                cat_id = label_train[i]
                if not cat_id in self.class_dict:
                    self.class_dict[cat_id] = []
                self.class_dict[cat_id].append(i)

        self.images_train_ = img_train
        self.target_train_ = label_train
        self.images_valid_ = img_valid
        self.target_valid_ = label_valid
        self.images_test_ = img_test       
        self.target_test_ = label_test
        self.loader = loader
        self.num_classes = len(label_name)
        self.train_num_per_cls = train_num_per_cls
        self.label_name = label_name
    
    def __getitem__(self, index):
        fn_train = self.images_train_[index]
        images_train = self.loader(fn_train)
        target_train = self.target_train_[index]
        
        ############################################## meta is here ##############################################
        
        
        meta = dict()
        
        if self.dual_sample:
            if self.dual_sampler_type == 'reverse':
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sampler_type == 'balance':
                sample_class = random.randint(0, self.num_classes-1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.dual_sampler_type == 'uniform':
                sample_index = random.randint(0, self.__len__() - 1)
        
            sample_img, sample_label = self.images_train_[sample_index], self.target_train_[sample_index]
            sample_img = self.loader(sample_img)
        
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label
        
        
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_train
        meta['sample_label'] = target_train
        
        ############################################## meta is above ##############################################
        
        return images_train, target_train, meta
    
    def __len__(self):
        return len(self.images_train_)
    
    def get_label_list(self, dataset):                                                   # how is the group when 4:1:1 ?
        if dataset == '28k':
            path = 'D:/00_Yume/RawData_28k_1116/train/'
        elif dataset == '34k':
            path = 'D:/00_Yume/original34k/train/'
        label_name = list(os.listdir(path))
        path = [ path + label_name[i] for i in range(len(label_name)) ]
        num_per_cls = []    # num of each class len=32 or 34
        for i in range(len(path)):
            data = os.listdir(path[i])
            num_per_cls.append(len(data))
        self.num_per_cls = num_per_cls
        g0, g1, g2, g3 = [], [], [], []
        for i in range(len(self.num_per_cls)):
            if self.num_per_cls[i] < 101:
                g3.append(i)
            elif self.num_per_cls[i] < 1001 and self.num_per_cls[i] >= 101:
                g2.append(i)
            elif self.num_per_cls[i] < 10001 and self.num_per_cls[i] >= 1001:
                g1.append(i)
            elif self.num_per_cls[i] >= 10001:
                g0.append(i)
        g = [g0, g1, g2, g3]
        return self.label_name, self.train_num_per_cls, g
    
    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

class IMBALANCED_WHITEBAIT_VALID(IMBALANCED_WHITEBAIT):
    def __getitem__(self, index_valid):
        fn_valid = self.images_valid_[index_valid]
        images_valid = self.loader(fn_valid)
        target_valid = self.target_valid_[index_valid]
        
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_valid
        meta['sample_label'] = target_valid
        
        return images_valid, target_valid, meta
    def __len__(self):
        return len(self.images_valid_)
    def get_gt_label(self):
        return self.target_valid_, self.target_test_
    
class IMBALANCED_WHITEBAIT_TEST(IMBALANCED_WHITEBAIT):
    def __getitem__(self, index_test):
        fn_test = self.images_test_[index_test]
        images_test = self.loader(fn_test)
        target_test = self.target_test_[index_test]
        
        # bbn meta
        meta = dict()
        meta['sample_image'] = images_test
        meta['sample_label'] = target_test
        return images_test, target_test, meta
    def __len__(self):
        return len(self.images_test_)
"""
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

"""
if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
"""
"""
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])



import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# preprocessing
transform = transforms.Compose([            
          #    transforms.RandomRotation(),
          #    transforms.CenterCrop(224),
              transforms.ToTensor(), #Convert image to tensor. 
              transforms.Normalize(                      
              mean=[0.485, 0.456, 0.406],   # Subtract mean 
              std=[0.229, 0.224, 0.225]     # Divide by standard deviation     
          #    mean=[0.5, 0.5, 0.5],   # Subtract mean 
          #    std=[0.5, 0.5, 0.5]     # Divide by standard deviation
              )])

def default_loader(path):
    img_pil =  Image.open(path)
    img_arr = np.array(img_pil)
    width, height, _ = img_arr.shape   # 記錄圖片長寬
    if width > height:
        img_pil = img_pil.rotate(-90, expand=True)  # 將圖片統一轉為橫的
    img_pil = img_pil.resize((270, 405))
#    img_pil = img_pil.resize((299,299))  # inception v3 專用
    img_tensor = transform(img_pil)
    return img_tensor


class trainset(Dataset):
    def __init__(self, loader=default_loader):
        path_train = 'D:/00_Yume/RawData_28k_1116/train/'
        label = list(os.listdir(path_train))
        label_38_list = list(label)
        path_train = [ path_train + label[i] for i in range(len(label)) ]
    
        ptrain = []
        label_train_number = []
        k = 0
        k1 = []
        for i in range(len(path_train)):
            ptrain[k:(k+len(os.listdir(path_train[i])))] = os.listdir(path_train[i])
            label_train_number += len(os.listdir(path_train[i])) * [i]
            k += len(os.listdir(path_train[i]))
            k1.append(len(os.listdir(path_train[i])))
        #count_train = Counter(label_train_number)

        A = []
        k = 0
        for i in range(len(path_train)):
            A[k:k+k1[i]] = ( k1[i] * [path_train[i]] )
            k += k1[i]
        file_train = [ '/'.join([A[i], ptrain[i]]) for i in range(len(ptrain)) ]
        k2 = sorted(k1, reverse=True)
        #定義好 image 的路徑
        self.images = file_train
        self.target = label_train_number
        self.loader = loader
        self.num_classes = len(label)

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        meta = dict()
        meta['sample_image'] = img
        meta['sample_label'] = target
        return img,target,meta

    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return self.num_classes


class valset(Dataset):
    
    def __init__(self, loader=default_loader):
        path_val = 'D:/00_Yume/RawData_28k_1116/val/'
        label = list(os.listdir(path_val))
        path_val = [ path_val + label[i] for i in range(len(label)) ]
    
        pval = []
        label_val_number = []
        q = 0
        q1 = []
        for i in range(len(path_val)):
            pval[q:(q+len(os.listdir(path_val[i])))] = os.listdir(path_val[i])
            label_val_number += len(os.listdir(path_val[i])) * [i]
            q += len(os.listdir(path_val[i]))
            q1.append(len(os.listdir(path_val[i])))
        #count_val = Counter(label_val_number)

        C = []
        q = 0
        for i in range(len(path_val)):
            C[q:q+q1[i]] = ( q1[i] * [path_val[i]] )
            q += q1[i]
        file_val = [ '/'.join([C[i], pval[i]]) for i in range(len(pval)) ]
    
        #定義好 image 的路徑
        self.images = file_val
        self.target = label_val_number
        self.loader = loader
        self.num_classes = len(label)

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        meta = dict()
        meta['sample_image'] = img
        meta['sample_label'] = target
        return img,target,meta

    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return self.num_classes


class testset(Dataset):
    def __init__(self, loader=default_loader):
        path_test = 'D:/00_Yume/RawData_28k_1116/test/'
        label = list(os.listdir(path_test))
        file_test = os.listdir(path_test)
        path_test = [ path_test + label[i] for i in range(len(label)) ]
    
        ptest = []
        label_test_number = []
        t = 0
        t1 = []
        for i in range(len(path_test)):
            ptest[t:(t+len(os.listdir(path_test[i])))] = os.listdir(path_test[i])
            label_test_number += len(os.listdir(path_test[i])) * [i]
            t += len(os.listdir(path_test[i]))
            t1.append(len(os.listdir(path_test[i])))
        #count_test = Counter(label_test_number)

        B = []
        t = 0
        for i in range(len(path_test)):
            B[t:t+t1[i]] = ( t1[i] * [path_test[i]] )
            t += t1[i]
        file_test = [ '/'.join([B[i], ptest[i]]) for i in range(len(ptest)) ]
        #定義好 image 的路徑
        self.images = file_test
        self.target = label_test_number
        self.loader = loader
        self.num_classes = len(label)

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        meta = dict()
        meta['sample_image'] = img
        meta['sample_label'] = target
        return img,target,meta

    def __len__(self):
        return len(self.images)
    
    def get_num_classes(self):
        return self.num_classes
"""