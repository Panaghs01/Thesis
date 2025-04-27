import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



class classifier(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(classifier,self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.fc2 = nn.Linear(100,num_classes)
        
    def forward(self,x):
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.softmax(x,dim=-1)
        
        return x
    
def train_classifier(model,classifier,epochs,optimizer,criterion,dataloader):
    for step in range(epochs):
        progress2 = tqdm(dataloader,desc = f"Epoch {step+1}", unit="batch")
        total_loss = 0
        for im,label,_ in progress2:
            with torch.no_grad():    
                h = model.encoder(im)
                h = model.pre_vq_conv(h)
    
            y = classifier(h)
    
            loss = criterion(y,label)
            
            total_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress2.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)
        
        print(f"Average loss for epoch {step+1}: {total_loss/len(dataloader)}")
    torch.save(classifier.state_dict(), "classifier.pth")
    
    

    
