import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerVQVAE(nn.Module):
    def __init__(self, input_shape=(1, 128, 216), num_codebook_vectors=512, 
                 codebook_dim=64, num_layers=4, num_heads=8, hidden_dim=256):
        super(TransformerVQVAE, self).__init__()
        
        self.input_shape = input_shape
        self.num_codebook_vectors = num_codebook_vectors
        self.codebook_dim = codebook_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, codebook_dim, kernel_size=3, stride=2, padding=1)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=codebook_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_codebook_vectors, codebook_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(codebook_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        # Encode the input
        z = self.encoder(x)
        
        # Reshape for transformer
        z = z.permute(2, 3, 0, 1).contiguous()
        z = z.view(-1, z.shape[2], z.shape[3])
        
        # Apply transformer
        z = self.transformer(z)
        
        # Reshape back
        z = z.view(x.shape[2]//8, x.shape[3]//8, x.shape[0], self.codebook_dim)
        z = z.permute(2, 3, 0, 1).contiguous()
        
        return z

    def decode(self, z):
        # Decode the latent representation
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        z_q, indices, vq_loss = self.vq(z)
        x_recon = self.decode(z_q)
        return x_recon, indices, vq_loss

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost = 0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # Reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # Distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2ze
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Quantize and unflatten
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Compute loss for embedding
        e_latent_loss = F.mse_loss(z_q.detach(), z)
        q_latent_loss = F.mse_loss(z_q, z.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        z_q = z + (z_q - z).detach()

        # Reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, vq_loss