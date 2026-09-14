import math

def distance(p, q):
    plat, plon = p
    qlat, qlon = q

    dlat, dlon = qlat - plat, qlon - plon

    intermediate = (math.sin(dlat / 2) ** 2) + math.cos(plat) * math.cos(qlat) * (math.sin(dlon / 2) ** 2)

    angle = 2 * math.atan2(math.sqrt(intermediate), math.sqrt(1 - intermediate))
    return angle # distance is radius * angle of the sector

def total_distance(cities):
    res = 0
    for i in range(0, len(cities)):
        res += distance(cities[i - 1], cities[i])

    return res
