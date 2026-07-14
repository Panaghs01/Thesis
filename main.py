import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from argparse import ArgumentParser
import os
import utils

import ColorMnist
from models import vq_vae
from models import simple_classifier
import SkinCancerData
import resnet

def train():
    pass

def test():
    pass

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy(force = True)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def adversarial_walk(f,h,a,model,device,steps = 2):    #h = latent representations f = classifier
    h_delta = h.clone().detach().requires_grad_(True).to(device)

    for p in f.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        p.requires_grad_(False)

    e = 1e-12
    for i in range(steps): 
        prediction = f(h_delta)
        prediction = torch.softmax(prediction, dim=1)
        entropy = -torch.special.entr(prediction + e).sum(dim=1).mean()
        gradient = torch.autograd.grad(entropy, h_delta,create_graph=False)[0]


        delta = (gradient - gradient.mean()) / (gradient.std() + e)    

        h_delta = (h_delta + a*delta).detach().requires_grad_(True)
    _,h_delta,perplexity,_ = model.vq(h_delta)



    torch.cuda.empty_cache()
    return h_delta,perplexity

torch.set_float32_matmul_precision('high')
transform = transforms.Compose([
    transforms.ToTensor()
#    transforms.Resize((256,256))
    ])


batch_size = 48

#colored_train = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,num_workers=8)
#colored_test = ColorMnist.get_biased_mnist_dataloader("coloredmnist_data", batch_size,1,train = False,num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "Code/archive/"
skin_train,skin_test = SkinCancerData.CreateLoader(path, transform, batch_size)



ALPHA = 0.1
TRAIN = False
Train_f = False
LOAD_VQ = True
LOAD_F = False
INFER = True

epochs = 300

num_hiddens = 512
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 2056
commitment_cost = 0.35
decay = 0.99
learning_rate = 0.0001



vq = vq_vae.model(num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings, embedding_dim, 
              commitment_cost,device).to(device)

optimizer = optim.Adam(vq.parameters(), lr=learning_rate, amsgrad=False)
criterion = torch.nn.MSELoss()

to_PIL = transforms.ToPILImage()

if TRAIN:

    vq_vae.train_model(vq,epochs, optimizer, criterion, skin_train,load=LOAD_VQ)

elif INFER:
    vq.load_state_dict(torch.load("vqvae.pth",weights_only=False))
    
    (im,label,_) = next(iter(skin_train))

    
    image = im.to(device)
    label = label.to(device) 
    _,recon,_ = vq(image)
    
    images = make_grid(image[:32])
    outputs = make_grid(recon[:32])

    images = to_PIL(images)
    images.save('original2.png')
    outputs = to_PIL(outputs)
    outputs.save('recons.png')

f = simple_classifier.classifier(embedding_dim*64*64,2,device).to(device)

f_optimizer = optim.Adam(f.parameters(),lr = 1e-3)
f_criterion = nn.CrossEntropyLoss()
epochs_f = 50
torch.cuda.empty_cache()
if Train_f:

    simple_classifier.train_classifier(vq,f,
                                       epochs_f, f_optimizer,
                                       f_criterion, skin_train,load=LOAD_F)


elif INFER:
    f.load_state_dict(torch.load("classifier.pth",weights_only=False))
    f.eval()
    vq.eval()
    
    (im,label,_) = next(iter(skin_test))
    image = im.to(device)
    label = label.to(device)
    images = make_grid(image[32:64])

    h = vq.encoder(image)
    h = vq.pre_vq_conv(h)
    output,perplexity = adversarial_walk(f, h, ALPHA,vq,device)
    recon = vq.decoder(output)
    
    outputs = make_grid(recon[32:64])

    images = to_PIL(images)
    images.save('original.png')
    outputs = to_PIL(outputs)
    outputs.save('outputs.png')


if __name__ == '__main__':


    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='test', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=224, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_GCN', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8| base_GCN_with_fusion | base_GCN | mamba')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)
    parser.add_argument('--accumulation_steps', default=0, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
