import torch
from torch import nn

class TSPTransformer(nn.Module):
    def __init__(self, num_cities, embedding_dim=128, num_layers=3, num_heads=4, feedforward_dim=512, dropout=0.1):
        super().__init__()

        self.num_cities = num_cities
        self.bos_token = num_cities

        self.token_embedding = nn.Embedding(
            num_cities + 1, 
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            num_cities + 1,
            embedding_dim
        )

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transfomer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )

        self.output = nn.Linear(
            embedding_dim,
            num_cities
        )
        

    def forward(self, tokens):
        batch_size, sequence_length = tokens.shape

        positions = torch.arrange(sequence_length, device=tokens.device)

        hidden = (self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :])

        causal_mask = torch.trui(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device),
            diagonal=1
        )

        hidden = self.transformer(
            hidden, 
            mask=causal_mask
        )

        return self.output(hidden)
        
