import torch
import torch.functional as F
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, efficientnet_b0, densenet121
import os

from models import vq_vae, simple_classifier
from torch.optim import lr_scheduler

def define_net(args):

    img_size = args.img_size
    in_channels = args.vqvae_embedding_dim * (img_size // 4) * (img_size // 4) # Calculate in_channels based on the output of the vqvae encoder

    vqvae_ckpt = vq_vae.model(num_hiddens=args.vqvae_hiddens,
                                num_residual_layers=args.vqvae_residual_layers,
                                num_residual_hiddens=args.vqvae_residual_hiddens,
                                embedding_dim=args.vqvae_embedding_dim,
                                num_embeddings=args.vqvae_num_embeddings,
                                commitment_cost=args.vqvae_commitment_cost)

    classifier_ckpt = simple_classifier.model(in_channels=in_channels, num_classes=args.n_class)

    if args.train == 'classifier':
        if os.path.exists(f"{args.best_ckpts}/best_ckpt_vqvae.pt"):
            print("Loading pre-trained VQ-VAE encoder...")
            checkpoint = torch.load(f"{args.best_ckpts}/best_ckpt_vqvae.pt")
            vqvae_ckpt.load_state_dict(checkpoint['vqvae_state_dict'])
            classifier_ckpt = simple_classifier.model(in_channels=in_channels, num_classes=args.n_class)
        else:
            raise FileNotFoundError("Pre-trained VQ-VAE encoder not found. Please train the VQ-VAE first.")
    elif args.train == 'strong_classifier' or 'standard':
        if os.path.exists(f"{args.best_ckpts}/best_ckpt_classifier.pt") and os.path.exists(f"{args.best_ckpts}/best_ckpt_vqvae.pt"):
            print("Loading models...")
            checkpoint_vq = torch.load(f"{args.best_ckpts}/best_ckpt_vqvae.pt")
            checkpoint_class = torch.load(f"{args.best_ckpts}/best_ckpt_classifier.pt",weights_only=False)
            vqvae_ckpt.load_state_dict(checkpoint_vq['model_strong_state_dict'])
            classifier_ckpt.load_state_dict(checkpoint_class['model_strong_state_dict'])
        else:
            raise FileNotFoundError("Pre-trained models not found. Please train the VQ-VAE and simple classifier first.")
    else:

        classifier_ckpt = None


    return vqvae_ckpt,classifier_ckpt

def define_strong_net(args):
    if args.strong_classifier == 'base_resnet18':
        net = base_resnet18(n_classes=args.n_class)
    elif args.strong_classifier == 'base_efficientnet_b0':
        net = base_efficientnet_b0(n_classes=args.n_class)
    elif args.strong_classifier == 'base_densenet121':
        net = base_densenet121(n_classes=args.n_class)
    else:
        raise NotImplementedError(f"Unknown strong classifier: {args.strong_classifier}")
    
    return net

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=15,min_lr=1e-8)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented' % args.lr_policy)
    return scheduler

##################### Strong classifiers #####################

class Base_Grad_model(nn.Module):
    def __init__(self):
        super(Base_Grad_model, self).__init__()
        self.gradients = None
        self.features_conv = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self,x):
        return self.features_conv(x)

class base_resnet18(Base_Grad_model):
    def __init__(self,n_classes):
        super(base_resnet18, self).__init__()
        res = resnet18(weights=resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT))
        
        self.features_conv = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
            res.layer1,
            res.layer2,
            res.layer3,
            res.layer4
        )

        self.avgpool = res.avgpool

        in_features = res.fc.in_features
        self.classifier = nn.Linear(in_features, n_classes)
        self.gradients = None

    def forward(self, x):
        x.requires_grad_()

        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
        


class base_efficientnet_b0(Base_Grad_model):
    def __init__(self,n_classes):
        super(base_efficientnet_b0, self).__init__()
        eff = efficientnet_b0(weights=efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT))

        self.features_conv = eff.features
        self.avg_pool = eff.avgpool
        eff.classifier[1].out_features = n_classes
        self.classifier = eff.classifier
        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
        
        x = F.relu(x, inplace=False) # Use False to avoid gradient issues
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class base_densenet121(Base_Grad_model):
    def __init__(self,n_classes):
        super(base_densenet121, self).__init__()
        dn = densenet121(pretrained=True)
        self.features_conv = dn.features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        dn.classifier.out_features = n_classes
        self.classifier = dn.classifier
        self.gradients = None


    def forward(self, x):
        x = self.features_conv(x)

        h = x.register_hook(self.activations_hook)
        
        x = F.relu(x, inplace=False) # Use False to avoid gradient issues
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        return x