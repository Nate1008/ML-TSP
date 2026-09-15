import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TourDataset
from model import TSPTransformer

def train(model, data_loader, optimizer, device, verbose=False):
    model.train()

    total_loss = 0
    total_tokens = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        out = model(inputs)

        loss = F.cross_entropy(
            out.reshape(-1, model.num_cities),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / total_tokens



        

