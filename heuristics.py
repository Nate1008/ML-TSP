from aux import distance
import math

def create_distance_matrix(cities):
    n = len(cities)
    dist = [[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    return dist

def two_opt(order, cities):

    D = create_distance_matrix(cities)

    n = len(order)
    while True:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b, c, d = order[i], i + 1, j, (j + 1) % n
                delta = - distance(cities[a], cities[b]) - distance(cities[c], cities[d]) + distance(cities[a], cities[c]) + distance(cities[b], cities[d])

                if (delta < 0):
                    order = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                    improved = True

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

            if (best < dist[cur][i]):
                nxt = i
                best = dist[cur][nxt]

        order.append(nxt)
        seen.add(nxt)
        cur = nxt
