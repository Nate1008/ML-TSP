import math

def distance(p, q):
    plat, plon = map(math.radians, p)
    qlat, qlon = map(math.radians, q)
    plat, plon = p
    qlat, qlon = q

    dlat, dlon = qlat - plat, qlon - plon

    intermediate = (math.sin(dlat / 2) ** 2) + math.cos(plat) * math.cos(qlat) * (math.sin(dlon / 2) ** 2)
    intermediate = max(0.0, min(1.0, intermediate))

    angle = 2 * math.atan2(math.sqrt(intermediate), math.sqrt(1 - intermediate))
    return angle # distance is radius * angle of the sector

def tour_distance(order, cities):
    res = 0
    for i in range(0, len(order)):
        res += distance(cities[order[i - 1]], cities[order[i]])

    return res
