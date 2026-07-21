import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.backends.cudnn as cudnn


class model(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(model,self).__init__()
        
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,num_classes)
        
    def forward(self,x):
        x = torch.flatten(x,1)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x


#--------------------------------------------------------------------#

def save_checkpoint(state,filename='checkpointf.tar'):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model,optimizer,epoch):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch - checkpoint['epoch']

def train_classifier(model,classifier,epochs,optimizer,criterion,dataloader,load=False,save_period=3):
    scaler = torch.amp.GradScaler()
    cudnn.benchmark = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    if load:
        checkpoint = torch.load("checkpointf.tar")
        epochs = load_checkpoint(checkpoint,model,optimizer,epochs)

    for epoch_idx in range(epochs):
        progress2 = tqdm(dataloader,desc = f"Epoch {epoch_idx+1}/{epochs}", unit="batch")
        total_loss = 0
        for im,label,_ in progress2:
            im = im.to(model.device)
            label = label.long().to(model.device)
            optimizer.zero_grad()
            with torch.no_grad():    
                h = model.encoder(im)
                h = model.pre_vq_conv(h)
            with torch.amp.autocast("cuda"):
                y = classifier(h).squeeze(1)
                loss = criterion(y,label)
            scaler.scale(loss).backward()

            # AMP: Unscales the gradients and performs optimization
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            progress2.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)
        scheduler.step(total_loss/len(dataloader))
        if (epoch_idx + 1) % save_period == 0:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':epoch_idx
            }
            save_checkpoint(checkpoint)
        print(f"Average loss for epoch {epoch_idx+1}: {total_loss/len(dataloader)}")

    torch.save(classifier.state_dict(), "classifier.pth")




