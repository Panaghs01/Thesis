import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CustomDataset(Dataset):
    def __init__(self,path,mode,transform=None):
        self.mode = mode
        self.path = path
        self.csv_path = path + mode + "_metadata2.csv"
        self.csv = pd.read_csv(self.csv_path)
        if mode == 'train':
           self.csv = self.csv[self.csv['protected'] == 0] # Filter out protected individuals for 100% bias in data

        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        path = os.path.join(str(self.path) + self.mode + '/' + str(self.csv.iloc[index].iloc[0] + '.jpg'))
        image = Image.open(path).convert("RGB")  # Ensure image is RGB
        map = {'benign':0,
               'malignant':1,
               'non-neoplastic':2   
            }
        self.csv["three_partition_label"] = self.csv["three_partition_label"].map(map)
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label)
        
        protected = torch.tensor(self.csv.iloc[index].iloc[6])


        return image,label,protected
    

class Fitzpatrick(Dataset):
    def __init__(self,root_dir,img_size,data_name,is_train=True,split='train',to_tensor=True):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.data_name = data_name

        self.df = pd.read_csv(os.path.join(self.root_dir, self.data_name + '_' + split + '.csv'))

        if is_train:
            self.augm = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(0.75),
                transforms.RandomVerticalFlip(0.75),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.9)),
                transforms.ToTensor() if to_tensor else transforms.Lambda(lambda x: x)
            ])
        else:
            self.augm = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor() if to_tensor else transforms.Lambda(lambda x: x)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label_map = {'benign': 0,'non-neoplastic':2, 'malignant':1}

        name = self.df.iloc[index]['new_img_name']
        label = self.df.iloc[index]['three_partition_label']
        label = torch.tensor(label_map[label],dtype=torch.long)
        fitzpatrick = self.df.iloc[index]['fitzpatrick']
        img_path = os.path.join(self.root_dir, 'full_dataset', name)

        image = Image.open(img_path).convert("RGB")

        image = self.augm(image)

        return {'name':name, 'image': image, 'label': label, 'fitzpatrick': fitzpatrick}

