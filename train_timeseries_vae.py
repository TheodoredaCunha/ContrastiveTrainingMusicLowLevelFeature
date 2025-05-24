import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from utils import seed_torch, loss_function, get_non_corrupt_files
from dataset import AudioDataset
from timeseries_vae import VAE


from tqdm import tqdm

def train():
    torch.cuda.empty_cache()
    seed_torch(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print('='*100)
    # Define hyperparameters
    batch_size = 32
    num_workers = 8
    start_beta = 0.00001
    lpips_weight = 1000
    mse_weight = 1
    load_model  = None

    if load_model:
        saved_weights = torch.load(f'model/{load_model}')

    filenames = get_non_corrupt_files('../../data/fma full/fma_large/*')

    train_dataset = AudioDataset(filenames[:7000], return_time_series = False)
    valid_dataset = AudioDataset(filenames[7000:14000], return_time_series = False)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory = False, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers = num_workers, pin_memory = False, shuffle=False)

    mel_spec_shape = next(iter(valid_dataset)).unsqueeze(0).shape

    latent_dim = 128
    if load_model:
        latent_dim = saved_weights['latent dim']

    vae = VAE(mel_spec_shape, latent_dim, device).to(device)
    
    if load_model:
        vae.load_state_dict(saved_weights['model'])


    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    # Training
    num_epochs = 400

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
            avg_kl_loss = 0

            for _, data in enumerate(tqdm(dataloader)):
                data = data.to(device)
                data = data.unsqueeze(1)
                optimizer.zero_grad()
                recon_batch, mu, logvar = vae(data)
                recon_batch = recon_batch.view(-1, mel_spec_shape[0], mel_spec_shape[1], mel_spec_shape[2])

                recon_loss, kl_loss = loss_function(recon_batch, data, mu, logvar, start_beta, mse_weight, lpips_weight)
                loss = recon_loss + kl_loss

                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                avg_recon_loss += recon_loss.item()
                avg_kl_loss += kl_loss.item()
            scheduler.step()

            avg_train_loss = train_loss / (len(dataloader))
            avg_recon_loss = avg_recon_loss / (len(dataloader))
            avg_kl_loss = avg_kl_loss / (len(dataloader))

            print('====> {} Epoch: {} Average loss: {:.4f} \nKL beta: {} \nAverage Reconstruction Loss: {:.4f} \nAverage Reconstruction KL loss: {:.4f}'.\
            format(mode.upper(), epoch, avg_train_loss, start_beta,  avg_recon_loss, avg_kl_loss))
            training_history.append([epoch, avg_train_loss, avg_recon_loss, avg_kl_loss])
        
    torch.save({ 
        'optimizer': optimizer.state_dict(),
        'model': vae.state_dict(),
        'history': training_history,
        'latent dim': latent_dim
    }, 'timeseries_vae_model/timestamp_model_with_lpips_lpipsw1000_beta00001.pth')



train()