import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cos_sim(a, b):
    """
    added eps for numerical stability
    """
    a_f = torch.flatten(a,1).unsqueeze(1)
    b_f = torch.flatten(b,1)

    return  F.cosine_similarity(a_f, b_f, dim=-1)        
                    
class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta, device, args, mul_num=10, downsampling_num = 4):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim*mul_num
        self.beta = beta
        self.device = device
 #mul_num  skip-3: 8/ skip-4: 10
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.structure_bank = nn.Parameter(torch.randn(n_e,128,int(args.size//(2**downsampling_num)),int(args.size//(2**downsampling_num)),requires_grad=False),requires_grad=False)
        
    def get_closest(self, structure):
        # structure_bnk_tensor  = torch.stack(self.structure_bank,dim=0)[:,0]
        structure_similarity = cos_sim(structure , self.structure_bank)

        closest_idx = torch.argmax(structure_similarity, dim=1)
        style = self.embedding.weight[closest_idx]
        # structure_ = structure_bnk_tensor[closest_index]
        return style

    def forward(self, z, structures):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
      
        loss = torch.tensor(0.0, device=self.device)
        # for i, z in enumerate(zs):
        z_flattened = z.view(-1, self.e_dim)
        # print(z_flattened.shape)
        # print(self.embedding)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        min_encoding_idx = min_encoding_indices.squeeze(1)
        self.structure_bank[min_encoding_idx] = structures.clone().detach()
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss += torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # zs[i] = z_q
            
            

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # return loss, z_q, perplexity, min_encodings, min_encoding_indices
        return loss, z_q