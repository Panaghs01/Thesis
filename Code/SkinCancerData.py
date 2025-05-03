import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
        
        return image,label,label
    

def CreateLoader(path,transform,batch_size,train = True):
    data = CustomDataset(path,transform=transform)    
    loader = DataLoader(data,batch_size=batch_size, shuffle = train)
    return loader
    
    

# path = "archive/metadata.csv"
# data = CustomDataset(path)


# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.ToPILImage()])
# im , _ = data[0]

# img = transform(im)
# print(type(img))
# img.show()








