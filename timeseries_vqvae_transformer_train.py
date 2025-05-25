import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from utils import seed_torch, vqvae_transformer_loss, get_non_corrupt_files
from dataset import AudioDataset
from timeseries_vqvae_transformer import TransformerVQVAE

from tqdm import tqdm
import gc

def train():
    torch.cuda.empty_cache()
    gc.collect()

    seed_torch(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print('='*100)

    # Define hyperparameters
    batch_size = 32
    num_workers = 4
    commitment_cost = 0.25
    load_model  = None

    if load_model:
        saved_weights = torch.load(f'model/{load_model}')

    latent_dim = 64
    if load_model:
        latent_dim = saved_weights['latent dim']


    filenames = get_non_corrupt_files('../../data/fma full/fma_large/*')
    train_dataset = AudioDataset(filenames[:7000], return_time_series = False)
    valid_dataset = AudioDataset(filenames[7000:14000], return_time_series = False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory = False, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory = False, shuffle=False)

    vae = TransformerVQVAE().to(device)

    if load_model:
        vae.load_state_dict(saved_weights['model'])


    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # Training
    num_epochs = 900

    torch.autograd.set_detect_anomaly(True)

    print('Training VAE...')
    training_history = []
    for epoch in range(num_epochs):
        dataloader = None
        print('\n\n')
        for mode in ['train', 'valid']:

            if 'train':
                vae.train()
                dataloader = train_dataloader
            else:
                vae.eval()
                dataloader = valid_dataloader

            train_loss = 0
            avg_recon_loss = 0
            avg_vq_loss = 0

            for i, data in enumerate(tqdm(dataloader)):
                data = data.to(device)
                data = data.unsqueeze(1)
                optimizer.zero_grad()
                x_recon, _, vq_loss = vae(data)
                recon_loss = vqvae_transformer_loss(x_recon, data)

                loss = recon_loss + vq_loss

                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                avg_recon_loss += recon_loss.item()
                avg_vq_loss += vq_loss.item()

            avg_train_loss = train_loss / (len(dataloader))
            avg_recon_loss = avg_recon_loss / (len(dataloader))
            avg_vq_loss = avg_vq_loss / (len(dataloader))

            print('====> {} Epoch: {} Average loss: {:.4f} \nCommitment Cost {} \nAverage Reconstruction Loss: {:.4f} \nAverage VQ loss: {:.4f}'.\
            format(mode.upper(), epoch, avg_train_loss, commitment_cost,  avg_recon_loss, avg_vq_loss))
            training_history.append([epoch, avg_train_loss, avg_recon_loss, avg_vq_loss])
        
    torch.save({ 
        'optimizer': optimizer.state_dict(),
        'model': vae.state_dict(),
        'history': training_history,
        'latent dim': latent_dim
    }, 'vq_vae/model4.pth')



train()
