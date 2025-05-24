import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertModel
from timeseries_vqvae_transformer import TransformerVQVAE



class SemanticSimilarityModel(nn.Module):
    def __init__(self):
        super(SemanticSimilarityModel, self).__init__()
        
        # Tranformer VQ-VAE and convolutional layers for time series music embeddings
        self.music_encoder = TransformerVQVAE()
        for param in self.music_encoder.parameters():
            param.requires_grad = False  # Freeze all parameters of music encoder

        self.conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3)) # output shape: (14 x 24 x 4)
        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3)) # output shape: (12 x 22 x 2)
        self.timeseries_fc = nn.Linear(141312, 512) 

        
        # BERT model and linear layer for text embeddings
        self.text_encoder = BertModel.from_pretrained('bert-base-cased')
        # self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        total_params = len(list(self.text_encoder.parameters()))

        # Iterate over each parameter with its index
        for idx, param in enumerate(self.text_encoder.parameters()):
            param.requires_grad = False
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, 512)

    def forward(self, time_series, input_ids, attention_mask):

        # Get time series embeddings
        melspec_embed = []
        for batch in time_series:
            current_batch = []
            for melspec in batch:
                z = self.music_encoder.encode(melspec.unsqueeze(0))
                z_q, *_ = self.music_encoder.vq(z)

                current_batch.append(z_q)
            current_batch = torch.cat(current_batch, dim = 0)
            melspec_embed.append(current_batch)
        
        
        
        melspec_embed = torch.stack(melspec_embed, dim=0)
        x_ts = melspec_embed.permute(0, 2, 1, 3, 4)
        x_ts = self.conv1(x_ts)
        x_ts = self.conv2(x_ts)
        batch_size, *_ = x_ts.shape
        x_ts = x_ts.view(batch_size, -1)
        x_ts = self.timeseries_fc(x_ts)


        # Get text embeddings
        text_embed = self.text_encoder(input_ids = input_ids, attention_mask = attention_mask)
        x_text = text_embed.last_hidden_state[:, 0, :]
        x_text = self.text_fc(x_text)
        
        return x_ts, x_text
