from aux import tour_distance
from heuristics import two_opt, nearest_neighbour
from data import load_locations


CAPITALS = "world_capitals.csv"
TOP150 = "world_cities_150.csv"

EARTH_RADIUS = 6371 # ~6371 KM 

names, cities = load_locations(CAPITALS)

N = len(cities)

order = [i for i in range(N)]

print(tour_distance(order, cities))

order = two_opt(order, cities)

print(tour_distance(order, cities))

order = nearest_neighbour(cities)

print(tour_distance(order, cities))



