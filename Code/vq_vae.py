import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

"""https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=RelHBLryfjcK"""



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class encoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(encoder,self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                         out_channels=num_hiddens//2,
                         kernel_size=4,
                         stride=2, padding=1)
        
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
    def forward(self,x):

        x = self._conv_1(x)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._residual_stack(x)

        
        return x
    

class decoder(nn.Module):
    def __init__(self,in_channels,num_hiddens,num_residual_layers,num_residual_hiddens):
        super(decoder,self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        

    def forward(self,x):
       
        x = self._conv_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)

        
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
    def __init__(self,num_hiddens,num_residual_layers,num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost):
        super(model,self).__init__()
        
        self.encoder = encoder(3, num_hiddens,num_residual_layers,num_residual_hiddens)
        
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                     out_channels=embedding_dim,
                                     kernel_size=1,
                                     stride = 1)
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = decoder(embedding_dim, num_hiddens,num_residual_layers,num_residual_hiddens)
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss , quantized , perplexity,_ = self.vq(x)

        x_recon = self.decoder(quantized)
        
        return loss,x_recon,perplexity
    
    
def train_model(model,epochs,optimizer,criterion,dataloader):
    model.train()
    for epoch_idx in range(epochs):
    
        progress_bar = tqdm(dataloader,desc = f"Epoch {epoch_idx+1}", unit="batch")
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
        print(f"Average loss for epoch {epoch_idx+1}: {total_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), "vqvae.pth")
    
    
    
    
    
    