import torch

from aux import tour_distance
from heuristics import two_opt, nearest_neighbour
from data import load_locations
from model import TSPTransformer
from dataset import TourDataset
from train import *

CAPITALS = "world_capitals.csv"
TOP150 = "world_cities_150.csv"

EARTH_RADIUS = 6371 # ~6371 KM 
EPOCHS = 30
POP_SIZE = 2000
TEMP = 0.9
VERBOSE = True
TEST_NUMBER = 1
PATH=f"./checkpoints/test{TEST_NUMBER}"


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

def save_checkpoint(path, model, optimizer, epoch, temperature):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "temperature": temperature
        },
        PATH+f"checkpoint_epoch_{epoch}.pt"
    )

names, cities = load_locations(CAPITALS)
N = len(cities)

device = get_device()
model = TSPTransformer(N).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


tours = []
for epoch in range(EPOCHS):
    model, tours = step(epoch, model, optimizer, tours, cities, POP_SIZE, TEMP, VERBOSE)
    save_checkpoint(PATH, model, optimizer, epoch, TEMP)

