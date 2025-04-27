import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import ColorMnist
import vq_vae
import simple_classifier



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def adversarial_walk(f,h,a,model,steps = 10):
    h_delta = h.clone().detach().requires_grad_(True)

    e = 1e-6
    for _ in range(steps):

        prediction = f(h_delta)

        entropy = -torch.special.entr(prediction).sum(dim=1).mean()

        gradient = torch.autograd.grad(entropy, h_delta)[0]
        
        delta = (gradient - gradient.mean()) / gradient.std() + e        
        h_delta = h_delta + a*delta

        _,h_delta,perplexity,_ = model.vq(h_delta)
        h_delta = h_delta.requires_grad_(True)


    return h_delta,perplexity


  
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((28,28))
    ])


training_data = datasets.MNIST(root="data", train=True, download=True,
                                  transform = transform)

validation_data = datasets.MNIST(root="data", train=False, download=True,
                                  transform = transform)

batch_size = 256

colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=0)
colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=0)
    

ALPHA = 0.1
TRAIN = False
Train_f = True

epochs = 100

num_hiddens = 256
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 128
commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-4
    


training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True) 
    
validation_loader = DataLoader(validation_data,
                               batch_size=43,
                               shuffle=True,
                               pin_memory=True)


    

model = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost)
    
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

if TRAIN:
    vq_vae.train_model(model,epochs, optimizer, criterion, colored_train)

else:
    model.load_state_dict(torch.load("vqvae2.pth",weights_only=False))

f = simple_classifier.classifier(64*7*7, 10)

f_optimizer = optim.SGD(f.parameters(),lr = 1e-3)
f_criterion = nn.CrossEntropyLoss()
epochs_f = 100

if Train_f:
    simple_classifier.train_classifier(model,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, colored_train)


else:
    f.load_state_dict(torch.load("classifier2.pth",weights_only=False))
    f.eval()
    model.eval()
    
    (im,_,_) = next(iter(colored_test))
    
    image = im
    
    imshow(make_grid(image[:64]))
    
    h = model.encoder(image)
    h = model.pre_vq_conv(h)
    output,perplexity = adversarial_walk(f, h, ALPHA,model)
    recon = model.decoder(output)
    
    imshow(make_grid(recon[:64]))
