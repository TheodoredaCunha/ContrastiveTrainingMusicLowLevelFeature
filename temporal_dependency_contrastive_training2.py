import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from tqdm import tqdm 
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
from contrastive_utils import contrastive_loss, test_process_dl

from contrastive_dataset import ContrastiveDataet
from contrastive_model import SemanticSimilarityModel

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# mp.set_start_method('spawn')

def train(fold = 1):

    df = pd.read_json('../../data/MusicBench/MusicBench_train.json', lines=True)
    
    df = df.loc[:, ['location', 'main_caption']]
    df['location'] = df['location'].apply(lambda x: '../../data/MusicBench/datashare/' + x)

    df_train = df.iloc[0:52567] 
    df_valid = df.iloc[52567:52767] 


    print(len(df_train))
    print(len(df_valid))

    
    batch_size = 16
    
    # Create an instance of the dataset and its dataloader
    train_dataset = ContrastiveDataet(df_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = ContrastiveDataet(df_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # initialize model and optimizer
    model = SemanticSimilarityModel().to(device)
    # model.load_state_dict(torch.load('contrasitive_training_checkpoints/model_contrastive_bert_musicbench_epoch100.pth')['model_state_dict'])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    

    epochs = 100

    # Create a directory for saving checkpoints
    checkpoint_dir = 'contrasitive_training_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print('Training Model with NCE Loss...')
    for epoch in range(epochs):

        avg_loss_train = 0
        avg_loss_valid = 0

        print('=' * 50)
        print(f' Epoch {epoch}')

        for mode in ['train', 'valid']:

            if mode == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = valid_dataloader

            for i, batch in enumerate(tqdm(dataloader, desc=f'{mode}')):
                optimizer.zero_grad()

                data, input_ids, attention_masks = batch
                data, input_ids, attention_masks = data.to(device), input_ids.to(device), attention_masks.to(device)

                music_logits, text_logits = model(data, input_ids, attention_masks)
                # print(music_logits.shape)
                # print(text_logits.shape)
                loss = contrastive_loss(music_logits, text_logits)
                
                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                    avg_loss_train += loss.item()
                else:
                    avg_loss_valid += loss.item()
            
            if mode == 'train':
                avg_loss_train /= (len(dataloader))
            else:
                avg_loss_valid /= (len(dataloader))

        print(f'Average Training Loss: {avg_loss_train}')
        print(f'Average Validation Loss: {avg_loss_valid}')

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_contrastive_bert_musicbench2_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss_train,
                'valid_loss': avg_loss_valid
            }, checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
        

train()