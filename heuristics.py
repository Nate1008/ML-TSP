from aux import distance, tour_distance
import math

D = None

EPS = 1e-9

def create_distance_matrix(cities):
    global D
    if (D != None):
        return D
    
    n = len(cities)
    D = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = distance(cities[i], cities[j])

    return D

def two_opt(order, cities):

    D = create_distance_matrix(cities)

    n = len(order)
    while True:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b, c, d = order[i], order[i + 1], order[j], order[(j + 1) % n]
                delta = - distance(cities[a], cities[b]) - distance(cities[c], cities[d]) + distance(cities[a], cities[c]) + distance(cities[b], cities[d])

                if (delta < -EPS):
                    # print("Improved?:", delta)
                    order = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                    improved = True

        # print(tour_distance(order, cities))

        if not improved:
            break

    return order

def nearest_neighbour(cities):
    n = len(cities)

    dist = create_distance_matrix(cities)
    
    cur = 0
    order = [cur]

    seen = set()
    seen.add(cur)

    for t in range(n - 1):
        nxt, best = -1, 2 * math.pi
        for i in range(n):
            if i in seen:
                continue

            if (best > dist[cur][i]):
                nxt = i
                best = dist[cur][nxt]

        order.append(nxt)
        seen.add(nxt)
        cur = nxt

    return order
