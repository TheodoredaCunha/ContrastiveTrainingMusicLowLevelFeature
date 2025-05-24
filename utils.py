import torch
import torch.nn as nn
import glob
import os
import numpy as np
import torch.nn.functional as F
from lpips import LPIPS
import csv
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#lpips_model = LPIPS(net='alex').to(device)


def collate_fn(batch):

    return batch

def seed_torch(seed=0):
    '''
    This function is called to initialize a seed value
    PyTorch, Python Hash, and NumPy are seeded so that
    any operations utilizing randomly generated numbers
    will be reproducible

    The default seed is 0, but can be adjusted through the seed
    parameter
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)# if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

mse = nn.MSELoss(reduction='sum')
def loss_function(recon_x, x, mu, logvar, beta, mse_weight, lpips_weight):
    recon_loss = mse(recon_x, x) * mse_weight# + torch.sum(lpips_model(recon_x, x)) * lpips_weight
    #recon_loss = torch.sum(lpips_model(recon_x, x))
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # mean KL loss
    KLD =  (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * beta # sum KL loss
    return recon_loss, KLD

def vq_vae_loss(recon_x, x, quantized, embed, commitment_cost):
    e_latent_loss = F.mse_loss(quantized.detach(), embed)
    #q_latent_loss = F.mse_loss(quantized, embed.detach())
    vq_loss = commitment_cost * e_latent_loss

    recon_loss = F.mse_loss(recon_x, x)# * mse_weight + torch.sum(lpips_model(recon_x, x)) * lpips_weight

    return recon_loss, vq_loss

def vqvae_transformer_loss(recon_x, x):
    recon_loss = F.mse_loss(recon_x, x)

    return recon_loss
        
def get_beta(epoch, period, max_weight):
    '''
    Cyclic Beta value retrieval
    each cycle is completed in 10 epochs
    '''
    midpoint = period / 2
    a = epoch / period
    b = int(epoch // period)

    val = (a - b) * period

    if val >= midpoint:
        return max_weight
    
    else:
        return max_weight * val/period
    
def get_non_corrupt_files(path):

    full_filenames = glob.glob(os.path.join(path, '*.mp3'))
    corrupt = []

    # Read the CSV file and append each row to the list of strings
    with open('strings.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            corrupt.append(row[0])

    filtered = [a for a in full_filenames if a not in corrupt]

    random.shuffle(filtered)
    return filtered
