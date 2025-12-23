
import torch
from torch.utils.data import DataLoader, Dataset

from methods.sotdd import compute_pairwise_distance
from methods.distance import DatasetDistance
# import os
# import pickle
# from otdd.pytorch.datasets import load_torchvision_data
from torch.utils.data.sampler import SubsetRandomSampler
import time
# from trainer import train, test_func, frozen_module
# from models.resnet import *


import pandas as pd
import numpy as np
import time
import os



def load_excel_feature(path, feature_dim=2048, label_col=2048, pred_col=None):
    
    df = pd.read_csv(path, header=None)

    features = df.iloc[:, :feature_dim].values
    labels = df.iloc[:, label_col].values.astype(int)

    if pred_col is not None:
        preds = df.iloc[:, pred_col].values.astype(int)
        return features, labels, preds

    return features, labels


class FeatureDataset(Dataset):
    def __init__(self, features, labels):

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# path_ori = 'fish/fish_test_features.csv'
# path_shift = 'fish/fish_new_features.csv'

# ori_features, ori_labels = load_excel_feature(path_ori)
# shift_features, shift_labels = load_excel_feature(path_shift)

# ori_dataset = FeatureDataset(ori_features, ori_labels)
# ori_dataloader = DataLoader(ori_dataset, batch_size=256, shuffle=True)

# shift_dataset = FeatureDataset(shift_features, shift_labels)
# shift_dataloader = DataLoader(shift_dataset, batch_size=256, shuffle=True)

# dataloaders = [ori_dataloader, shift_dataloader]

folder_path = 'cifar/lt_1_100'

train_file = 'cifar100_test_ep100.csv'
train_features, train_labels = load_excel_feature(os.path.join(folder_path, train_file))
train_dataset = FeatureDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_file = 'cifar100_test_ep100.csv'
test_features, test_labels = load_excel_feature(os.path.join(folder_path, test_file))
test_dataset = FeatureDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)



dataloaders = [train_loader, test_loader]

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',        ## Noise
          'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', ## Blur
          'snow', 'frost', 'fog', 'brightness',           ## Weather
          'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', ## Digital

          ## Extra 4 corruptions
          'speckle_noise', 'gaussian_blur', 'spatter', 'saturate' ]


for corr in corruptions:
    features, labels = load_excel_feature(os.path.join(folder_path, f'cifar100c_{corr}_5.csv'))
    
    dataset = FeatureDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    dataloaders.append(loader)




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def sotdd(dataloaders, save_dir):

    ### 計算不同的prediction，越多距離估算越準，相對越慢
    projection_list = [100, 500, 1000, 5000, 10000] 

    pairwise_dist = torch.zeros(len(dataloaders), len(dataloaders))
    print("Compute sOTDD...")
    print(f"Number of datasets: {len(dataloaders)}")

    for proj_id in projection_list:
        try:
            
            kwargs = {
                 "dimension": 2048,   ## 先把img size, 或輸入dimension數
                 # "num_channels": 3,
                 "num_moments": 5,
                 "use_conv": False,  ## True: 輸入為img; False: 輸入為feature
                 "precision": "float",
                 "p": 2,
                 "chunk": 1000
                 }
            
            start = time.time()
            list_pairwise_dist = compute_pairwise_distance(list_D=dataloaders, num_projections=proj_id, device=device, **kwargs)
            end = time.time()
            sotdd_time_taken = end - start
            
            t = 0
            for i in range(len(dataloaders)):
                for j in range(i+1, len(dataloaders)):                        
                    pairwise_dist[i, j] = list_pairwise_dist[t]
                    pairwise_dist[j, i] = list_pairwise_dist[t]
                    t += 1

            torch.save(pairwise_dist, f'{save_dir}/sotdd_{proj_id}_dist.pt')
            with open(f'{save_dir}/time_running.txt', 'a') as file:
                file.write(f"Time proccesing for sOTDD ({proj_id} projections): {sotdd_time_taken} \n")
            
            print(f'Projection nums: {proj_id}, Distance: {pairwise_dist}')
            print(f'Use time: {sotdd_time_taken}')
            
        except:
            with open(f'{save_dir}/time_running.txt', 'a') as file:
                file.write(f"Time proccesing for sOTDD ({proj_id} projections): None \n")


def otdd_exact(dataloaders, save_dir='result'):
    # OTDD
    dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
    print("Compute OTDD (exact)...")

    # try:
    start = time.time()
    # for i in range(len(dataloaders)):
    for i in range(1):
            for j in range(i+1, len(dataloaders)):
                dist = DatasetDistance(dataloaders[i], dataloaders[j], 
                                       inner_ot_method='exact', debiased_loss=True, p=2, entreg=1e-3, device=device)
                
                d = dist.distance().item()
                dict_OTDD[i][j] = d
                dict_OTDD[j][i] = d
    end = time.time()
    otdd_time_taken = end - start
    print(f'Distance: {dict_OTDD}')
    print(f'Use time: {otdd_time_taken}')

    torch.save(dict_OTDD, f'{save_dir}/exact_otdd_dist.pt')

    with open(f'{save_dir}/time_running.txt', 'a') as file:
            file.write(f"Time proccesing for OTDD (exact): {otdd_time_taken} \n")
    # except:
    #     return 'OTDD(exact) failed.'
    #     with open(f'{save_dir}/time_running.txt', 'a') as file:
    #         file.write(f"Time proccesing for OTDD (exact): None \n")


## weird
def otdd_gaussian(dataloaders, save_dir='result'):
    # OTDD
    dict_OTDD = torch.zeros(len(dataloaders), len(dataloaders))
    print("Compute OTDD (gaussian_approx, iter 20)...")
    

    start = time.time()
    # for i in range(len(dataloaders)):
    for i in range(1):    
            for j in range(i+1, len(dataloaders)):
                dist = DatasetDistance(dataloaders[i], dataloaders[j], inner_ot_method='gaussian_approx', 
                                           debiased_loss=True, p=2, sqrt_method='approximate', 
                                           nworkers_stats=0, sqrt_niters=20, entreg=1e-3, device=device)
                    
                d = dist.distance().item()
                dict_OTDD[i][j] = d
                dict_OTDD[j][i] = d

    end = time.time()
    otdd_time_taken = end - start
    print(otdd_time_taken)
    print(dict_OTDD)

    torch.save(dict_OTDD, f'{save_dir}/ga_otdd_dist.pt')
    with open(f'{save_dir}/time_running.txt', 'a') as file:
            file.write(f"Time proccesing for OTDD (gaussian_approx, iter 20): {otdd_time_taken} \n")


sotdd(dataloaders=dataloaders, save_dir='result')
otdd_exact(dataloaders=dataloaders, save_dir='result')
# otdd_gaussian(dataloaders=dataloaders, save_dir='result')
