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

K = 3

def train(epoch, model, data_loader, optimizer, device, verbose=False):   
    if verbose:
        print(f"Started Epoch {epoch}!")

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

def make_loader(num_cities, tours, batch_size):
    tours_dataset = TourDataset(tours, num_cities)
    return DataLoader(tours_dataset, batch_size=batch_size, shuffle=True)

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
    for _ in range(K * population_size):
        order = [i for i in range(n)]
        shuffle(order)
        order = two_opt(order, cities)
        tours.append(order)
        print(_)

    return prune_tours(tours, cities, population_size)

def create_next_generation(old_tours, new_tours, cities, population_size):
    tours = [two_opt(new_tours[i], cities) for i in range(new_tours)] + old_tours;
    return prune_tours(tours, cities, population_size)

def evaluate(tours, cities):
    best, avg = torch.inf, 0
    for tour in tours:
        dist = tour_distance(tour, cities)
        avg += dist
        best = min(best, dist)

    return best, (avg / len(tours))

def step(epoch, model, optimizer, old_tours, cities, population_size, temperature, device, verbose=False):
    tours = None
    if epoch == 0:
        assert len(old_tours) == 0
        tours = create_initial_data(cities, population_size)
        if verbose:
            print("Initialized! Beginning Training Process!")
    else:
        new_tours = generate_tours(model, K * population_size, temperature)   

        if verbose:
            print(f"Generated Tours using Model from Epoch {epoch - 1}")
        
        if verbose:
            best, avg = evaluate(tours)
            print(f"Model from Epoch: {epoch - 1} ===> Best: {best} | Average: {avg}")
    
        tours = create_next_generation(old_tours, new_tours, cities, population_size)

    train(epoch, model, make_loader(model.num_cities, tours, 128), optimizer, device, verbose)

    return model, tours
    
def prune_tours(tours, cities, population_size):
    unique = set()
    for i in range(len(tours)):
        tour = [int(city) for city in tours[i]]
        pos = tour.index(0)
        forwards = tour[pos:] + tour[:pos]

        backwards = [forwards[0], *reversed(forwards[1:])]

        unique.add(min(tuple(forwards), tuple(backwards)))

    ranked = sorted(unique, key=lambda tour: tour_distance(tour, cities))

    return [list(tour) for tour in ranked[:population_size]]


        

