import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import os


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CustomDataset(Dataset):
    def __init__(self,csv,transform=None):
        self.csv = pd.read_csv(csv)
        self.transform = transform
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        path = os.path.join('archive/SkinCancer',str(self.csv.iloc[index].iloc[1] + '.jpg'))
        image = Image.open(path)
        label = self.csv.iloc[index].iloc[5]
        if self.transform:
            image = self.transform(image)
        if label == 'male':
            label = 0
        else:
            label = 1
            
        label = torch.tensor(label)
        
        return image,label
    
    
path = "archive/metadata.csv"
data = CustomDataset(path)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.ToPILImage()])
im , _ = data[0]

img = transform(im)
print(type(img))
img.show()








