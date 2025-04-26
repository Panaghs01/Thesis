import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from PIL import Image

from six.moves import xrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import ColorMnist

"""https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=RelHBLryfjcK"""


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class classifier(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(classifier,self).__init__()
        self.fc1 = nn.Linear(in_channels, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,num_classes)
        
    def forward(self,x):
        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)

        x = F.softmax(x,dim=-1)
        
        return x

def adversarial_walk(f,h,a,quantizer,steps = 8):
    h_delta = h.clone().detach().requires_grad_(True)
    e = 1e-6
    for _ in range(steps):
        prediction = f(h_delta)

        entropy = -torch.special.entr(prediction).sum(dim=1).mean()
        print(entropy)
        gradient = -torch.autograd.grad(entropy, h_delta)[0]
        
        delta = (gradient - gradient.mean()) / gradient.std() + e        
        h_delta = h_delta + a*delta
        _,h_delta,_,_ = quantizer(h_delta)
        h_delta = h_delta.requires_grad_(True)


    return h_delta
    
class encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens):
        super(encoder,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = num_hiddens // 8,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1)
        self.bn1 = nn.BatchNorm2d(num_hiddens//8)
        
        self.conv2 = nn.Conv2d(in_channels = num_hiddens//8,
                               out_channels = num_hiddens//4,
                               kernel_size = 4,
                               stride = 2,
                               padding = 1)
        self.bn2 = nn.BatchNorm2d(num_hiddens//4)
        self.conv3 = nn.Conv2d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_hiddens//2)
        self.conv4 = nn.Conv2d(in_channels=num_hiddens//2,
                               out_channels=num_hiddens,
                               kernel_size=2,
                               stride = 1,
                               padding =1)
        self.bn4 = nn.BatchNorm2d(num_hiddens)
        
        self.conv5 = nn.Conv2d(in_channels=num_hiddens,
                               out_channels=64,
                               kernel_size=2,
                               stride = 1,
                               padding =1)
        self.bn5 = nn.BatchNorm2d(64)
        
    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)

        return x
    

class decoder(nn.Module):
    def __init__(self,in_channels,num_hiddens):
        super(decoder,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = num_hiddens,
                               kernel_size = 5,
                               stride = 2,
                               padding = 1)
        
        self.bn1 = nn.BatchNorm2d(num_hiddens)
        
        self.transpose1 = nn.ConvTranspose2d(in_channels = num_hiddens,
                                             out_channels = num_hiddens//2,
                                             kernel_size = 4,
                                             stride = 2,
                                             padding = 1)
        
        self.transpose2 = nn.ConvTranspose2d(in_channels = num_hiddens//2,
                                             out_channels= num_hiddens//4 ,
                                             kernel_size = 3,
                                             stride = 2,
                                             padding = 1)
        
        self.transpose3 = nn.ConvTranspose2d(in_channels = num_hiddens//4,
                                             out_channels= 3 ,
                                             kernel_size = 2,
                                             stride = 2,
                                             padding = 1)
        

    def forward(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.transpose1(x)

        x = F.relu(x)
        
        x = self.transpose2(x)

        x = F.relu(x)

        x = self.transpose3(x)

        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    
    
class model(nn.Module):
    def __init__(self,num_hiddens,num_embeddings,embedding_dim,commitment_cost):
        super(model,self).__init__()
        
        self.encoder = encoder(3, num_hiddens)
        
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride = 1)
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = decoder(embedding_dim, num_hiddens)
    
    def forward(self,x):
        x = self.encoder(x)
        loss , quantized , perplexity,_ = self.vq(x)
        
        x_recon = self.decoder(quantized)
        
        return loss,x_recon,perplexity
  
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28))
    ])


training_data = datasets.MNIST(root="data", train=True, download=True,
                                  transform = transform)

validation_data = datasets.MNIST(root="data", train=False, download=True,
                                  transform = transform)

colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", 256,1,num_workers=0)
colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", 32,1,train = False,num_workers=0)
    

ALPHA = 0.1
TRAIN = False
Train_f = False
batch_size = 256
epochs = 10
num_hiddens = 256
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 20
commitment_cost = 1
decay = 0.99
learning_rate = 1e-3
    




training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True) 
    
validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


    

model = model(num_hiddens,num_embeddings, embedding_dim, 
              commitment_cost)
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

if TRAIN:
    model.train()
    for epoch_idx in range(epochs):
    
        progress_bar = tqdm(colored_train,desc = f"Epoch {epoch_idx+1}", unit="batch")
        total_loss = 0
        for im,label,_ in progress_bar:
            optimizer.zero_grad()
    
            vq_loss, data_recon , perplexity = model(im)
            recon_loss = criterion(data_recon,im)
            
            loss = recon_loss + vq_loss
            
            total_loss += loss
            
            loss.backward()
            
            optimizer.step()
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)
        print(f"Average loss for epoch {epoch_idx+1}: {total_loss/len(colored_train)}")
    
    torch.save(model.state_dict(), "vqvae.pth")
else:
    model.load_state_dict(torch.load("vqvae.pth",weights_only=False))

# with torch.no_grad():
#     model.eval()
    
#     (im,_,_) = next(iter(colored_test))
    
#     image = im
    
#     imshow(make_grid(image))
    
#     loss,recon,perp = model(image)
    

#     image = image
#     recon = recon
#     imshow(make_grid(recon))

f = classifier(64*9*9, 10)

f_optimizer = optim.SGD(f.parameters(),lr = 1e-3)
f_criterion = nn.CrossEntropyLoss()
steps = 10

if Train_f:

    for step in range(steps):
        progress2 = tqdm(colored_train,desc = f"Epoch {step+1}", unit="batch")
        total_loss = 0
        for im,label,_ in progress2:
            with torch.no_grad():    
                h = model.encoder(im)
    
            y = f(h)
    
            loss = f_criterion(y,label)
            
            total_loss += loss
            
            f_optimizer.zero_grad()
            loss.backward()
            f_optimizer.step()
            
            progress2.set_postfix({"loss": f"{loss.item():.4f}"},refresh=True)
        
        print(f"Average loss for epoch {step+1}: {total_loss/len(colored_train)}")
    torch.save(f.state_dict(), "classifier.pth")


else:
    f.load_state_dict(torch.load("classifier.pth",weights_only=False))
    f.eval()
    model.eval()
    
    (im,_,_) = next(iter(colored_test))
    
    image = im
    
    imshow(make_grid(image))
    
    h = model.encoder(image)
    output = adversarial_walk(f, h, ALPHA,model.vq)
    recon = model.decoder(output)
    
    imshow(make_grid(recon))
    















