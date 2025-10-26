import numpy as np
import queue
from queue import PriorityQueue

def init_cd(n: int)-> np.ndarray:
    p_cd = np.zeros(n, dtype=int)
    for i in range(n):
        p_cd[i] = i
    return p_cd

def union(rep_1: int, rep_2: int, p_cd: np.ndarray)-> int:
    root_1 = find(rep_1, p_cd)
    root_2 = find(rep_2, p_cd)
    if root_1 != root_2:
        p_cd[root_1] = root_2
    return p_cd

def find(ind: int, p_cd: np.ndarray)-> int:
    while p_cd[ind] != ind:
        ind = p_cd[ind]
    return ind

def create_pq(n: int, l_g: list)-> queue.PriorityQueue:
    pq = queue.PriorityQueue()
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(np.array(l_g[i]) - np.array(l_g[j]))
            pq.put((dist, (i, j)))
    return pq

def kruskal(n: int, l_g: list)-> tuple[int, list]:
    p_cd = init_cd(n)
    pq = create_pq(n, l_g)
    total_cost = 0.0
    edges_used = 0
    l_t = []

    while not pq.empty() and edges_used < n - 1:
        dist, (u, v) = pq.get()
        if find(u, p_cd) != find(v, p_cd):
            union(u, v, p_cd)
            l_t.append((u, v, dist))
            total_cost += dist
            edges_used += 1

    while not pq.empty():
        pq.get()

    if edges_used == max(0, n - 1):
        return (n, l_t)

    return None