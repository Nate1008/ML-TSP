import time
import math
from random import shuffle

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TourDataset
from model import TSPTransformer
from aux import tour_distance
from heuristics import two_opt

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

    if verbose:
        print(f"Loss: {(total_loss / total_tokens):.4f}")

    return total_loss / total_tokens

def make_loader(model, tours):
    tours_dataset = TourDataset(tours, model.num_cities)
    data_loader = DataLoader(tours_dataset, batch_size=128, shuffle=True)
    return data_loader

@torch.no_grad()
def generate_tours(model, batch_size, temperature, start_city=0):

    device = next(model.parameters()).device
    model.eval()

    tokens = torch.full((batch_size, 1), model.bos_token, dtype=torch.long, device=device)

    first_city = torch.full((batch_size, 1), start_city, dtype=torch.long, device=device)

    tours = [first_city]

    visited = torch.zeros(batch_size, model.num_cities, dtype=torch.bool, device=device)

    visited.scatter_(1, first_city, True)
    tokens = torch.cat([tokens, first_city], dim=1)

    for _ in range(1, model.num_cities):
        out = model(tokens)[:, -1, :]
        out = out / temperature
        out = out.masked_fill(visited, -torch.inf)

        prob = F.softmax(out, dim=-1)

        next_city = torch.multinomial(prob, num_samples=1)

        tours.append(next_city)
        visited.scatter_(1, next_city, True)
        tokens = torch.cat([tokens, next_city], dim=1)

    return torch.cat(tours, dim=1)

def create_initial_data(cities, population_size):
    n = len(cities)
    tours = []
    for _ in range(10 * population_size):
        order = shuffle([i for i in range(n)])
        order = two_opt(order)
        tours.append(order)

    return prune_tours(tours, cities, population_size)

def create_next_generation(old_tours, new_tours, cities, population_size):
    tours = [two_opt(new_tours[i]) for i in range(new_tours)] + old_tours;
    return prune_tours(tours, cities, population_size)

def evaluate(tours, cities):
    best, avg = torch.inf, 0
    for tour in tours:
        dist = tour_distance(tour, cities)
        avg += dist
        best = min(best, dist)

    return best, (avg / len(tours))

def step(epoch, model, optimizer, old_tours, cities, population_size, temperature, verbose=False):
    tours = None
    if epoch == 0:
        assert len(old_tours) == 0
        tours = create_initial_data(cities, population_size)
        if verbose:
            print("Initialized! Beginning Training Process!")
    else:
        new_tours = generate_tours(model, 10 * population_size, temperature)   
        
        if verbose:
            best, avg = evaluate(tours)
            print(f"Epoch: {epoch} | Best: {best} | Average: {avg}")
    
        tours = create_next_generation(old_tours, new_tours, cities, population_size)

    model = train(tours, make_loader(tours), optimizer, model.device, verbose)
    return model, tours
    
def prune_tours(tours, cities, population_size):
    for i in range(len(tours)):
        pos = tours[i].find(0)
        tours[i] = tours[i][pos:] + tours[i][:pos]

    unique = set(tuple(tours))

    ranked = sorted(unique, key=lambda tour: tour_distance(tour, cities))

    return [list(tour) for tour in ranked[:population_size]]


        

