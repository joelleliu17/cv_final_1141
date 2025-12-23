
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader_1 import IMBALANCED_WHITEBAIT
from dataloader_1 import IMBALANCED_WHITEBAIT_VALID
from dataloader_1 import IMBALANCED_WHITEBAIT_TEST
from dataloader_fish import ADAPT_WHITEBAIT


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'


import numpy as np
import random
import math
import pandas as pd
import torch.backends.cudnn as cudnn


from methods import tent, stamp, cotta, sar, deyo  #roid, 
from utils.conf import cfg

## 分類正確率
def gacc(target_test, predicted):
    test_g_per = [0] * 4  # 每组样本数量统计
    test_acc_per = [0] * 4
    # 更新每组样本数量和准确数量
    for gt, pred in zip(target_test, predicted):
        # 更新样本组成统计
        if gt in g[0]:
            test_g_per[0] += 1
            if gt == pred:
                test_acc_per[0] += 1
        elif gt in g[1]:
            test_g_per[1] += 1
            if gt == pred:
                test_acc_per[1] += 1
        elif gt in g[2]:
            test_g_per[2] += 1
            if gt == pred:
                test_acc_per[2] += 1
        elif gt in g[3]:
            test_g_per[3] += 1
            if gt == pred:
                test_acc_per[3] += 1
        else:
            print(f"Unexpected label: {gt}")

    test_acc_rate_per_group = [100 * acc / total if total > 0 else 0 for acc, total in zip(test_acc_per, test_g_per)]
    return test_g_per, test_acc_rate_per_group



SEED = 123

batch_size = 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_weight_path = 'cifar100_100_res50_para_100.pth'

methods = 'cotta'

dataset = 'cifar100'

############################ 載入 Dataset ############################
############################ fish data ############################

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


setup_seed(seed=666)

train_data = IMBALANCED_WHITEBAIT(dataset='28k')
valid_data = IMBALANCED_WHITEBAIT_VALID(dataset='28k')
test_data = IMBALANCED_WHITEBAIT_TEST(dataset='28k')

fish_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
fish_valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
fish_test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))

label, num_per_cls, g = train_data.get_label_list('28k') # num_per_cls: train
valid_gt_label, test_gt_label = valid_data.get_gt_label()
num_classes = len(label)
print(num_classes)


## 新的測試資料

fish_adapt_data = ADAPT_WHITEBAIT(datapath='C:/Users/User/Desktop/datasets/03new_fish/')

############################ cifat data ############################

# CIFAR-100-C 測試資料集載入（你需要事先把 CIFAR-100-C 下載好放在 cifar100c_root）
class CIFAR100C_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        # CIFAR-100-C corruption data 存成 numpy file, e.g. gaussian_noise.npy
        corrupted_file = os.path.join(root, f"{corruption}.npy")
        self.data = np.load(corrupted_file)
        self.data = self.data[(severity-1)*10000 : severity*10000]  # 取對應 severity 的 10000 張圖片
        self.transform = transform
        # 讀取標籤
        self.targets = np.load(os.path.join(root, 'labels.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        # CIFAR-100-C 裡的圖片是 uint8 numpy array，轉成 PIL Image
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target


cifar100c_root = 'C:/Users/User/Desktop/datasets/TTA/CIFAR-100-C'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
])

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',        ## Noise
          'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', ## Blur
          'snow', 'frost', 'fog', 'brightness',           ## Weather
          'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', ## Digital

          ## Extra 4 corruptions
          'speckle_noise', 'gaussian_blur', 'spatter', 'saturate' 
        ]

testc_dataset = CIFAR100C_Dataset(cifar100c_root, corruption=corruptions[0], severity=5, transform=transform_test)  


############################ model ############################


## 預訓練模型權重
criterion = nn.CrossEntropyLoss()


class ResNet101Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101Model, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 替換最後一層

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)  
        features = features.view(features.size(0), -1)  

        x = self.model.fc(features)

        if return_features:
            return x, features  # 返回倒數第二層特徵
        
        return x


class ResNet50Model(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Model, self).__init__()
        from torchvision.models import ResNet50_Weights
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x, return_features=False):
        # 手动走到 avgpool 之前
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        features = self.model.avgpool(x)
        features = features.view(features.size(0), -1)

        x = self.model.fc(features)

        if return_features:
            return x, features  # 返回预测和倒数第二层特征
        return x



############################ TTA method ############################

def setup_tent(model, cfg):
    model = tent.Tent.configure_model(model)
    params, param_names = tent.Tent.collect_params(model)
    optimizer = optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    tent_model = tent.Tent(model, optimizer,
                      steps=cfg.OPTIM.STEPS,
                      episodic=cfg.MODEL.EPISODIC,
                      )
    return tent_model, param_names


## 出事
def setup_stamp(base_model, cfg):
    from utils.sam import SAM

    params, param_names = stamp.STAMP.collect_params(base_model)
    # optimizer = setup_optimizer(params, cfg)
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, rho=0.05)
    model = stamp.STAMP(base_model, optimizer, cfg.STAMP.ALPHA, cfg.num_classes)
    return model

def setup_cotta(model, cfg):
    model = cotta.CoTTA.configure_model(model)
    params, param_names = cotta.CoTTA.collect_params(model)
    optimizer = optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    cotta_model = cotta.CoTTA(model, optimizer,
                        steps=cfg.OPTIM.STEPS,
                        episodic=cfg.MODEL.EPISODIC,
                        dataset_name=cfg.CORRUPTION.ID_DATASET,
                        mt_alpha=cfg.M_TEACHER.MOMENTUM,
                        rst_m=cfg.COTTA.RST,
                        ap=cfg.COTTA.AP)
    return cotta_model, param_names

def setup_sar(model, cfg):
    sar_model = sar.SAR(model, lr=cfg.OPTIM.LR, batch_size=cfg.TEST.BATCH_SIZE, steps=cfg.OPTIM.STEPS,
                    num_classes=cfg.num_classes, episodic=cfg.MODEL.EPISODIC, reset_constant=cfg.SAR.RESET_CONSTANT,
                    e_margin=math.log(cfg.num_classes) * (
                        0.40 if cfg.SAR.E_MARGIN_COE is None else cfg.SAR.E_MARGIN_COE))

    return sar_model



def adaptation_result(pretrained_model, adapt_data, methods, dataset, device):

    if methods == 'base':
        model = pretrained_model.eval()
    elif methods == 'tent':
        model, param_names = setup_tent(pretrained_model, cfg)
    elif methods == 'stamp':
        model = setup_stamp(pretrained_model, cfg)
    elif methods == 'cotta':
        model, param_names = setup_cotta(pretrained_model, cfg)
    elif methods == 'sar':
        model = setup_sar(pretrained_model, cfg)

    print(f'================= {methods} =================')

    adapt_loader = DataLoader(adapt_data, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    features=[]
    preds, targets=[], []

    with torch.no_grad():
        for images, labels in adapt_loader:
            images, labels = images.to(device), labels.to(device)  # Move input to the correct device

            if methods == 'base':
                outputs, penultimate_features = model(images, return_features=True)
            else:
                outputs, penultimate_features = model(images)

            features.append(penultimate_features.detach().cpu().numpy())

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            # 记录预测结果和真实标签
            targets.extend(labels.detach().cpu().numpy())
            preds.extend(predicted.detach().cpu().numpy())

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            print(f'Now: {len(preds)} /{len(adapt_data)}', end='\r')

    test_accuracy = 100 * (test_correct / test_total)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    if dataset=='fish':
        test_g_per, test_acc_rate_per_group = gacc(targets, preds)
        print(f"Test Group Composition (Samples Per Group): {test_g_per}")
        print(f"Test Accuracy Per Group: {test_acc_rate_per_group}")

    return features, targets, preds



if dataset == 'cifar100':
    cfg.num_classes = 100
    pretrained_model = ResNet50Model(num_classes=100)
elif dataset == 'fish':
    cfg.num_classes = num_classes
    pretrained_model = ResNet101Model(num_classes=num_classes)

pretrained_model.load_state_dict(torch.load(model_weight_path))
pretrained_model.eval()
pretrained_model.to(device)

features, targets, preds = adaptation_result(pretrained_model, fish_adapt_data, methods, dataset)



############# 輸出feature 存成 CSV 格式 #############
import pandas as pd

csv_features = np.vstack(features)  # 轉換為 2D 陣列
csv_labels = np.array(targets).reshape(-1, 1)  # 標籤
csv_predictions = np.array(preds).reshape(-1, 1)  # 預測標籤

data = np.hstack([csv_features, csv_labels, csv_predictions])  # 結合所有數據
df = pd.DataFrame(data)

filename = f"{methods}_adapt_features.csv"
df.to_csv(filename, index=False)
print(f"Features saved to {filename}.")
