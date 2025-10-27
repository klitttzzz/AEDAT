import numpy as np
import queue
from queue import PriorityQueue

def init_cd(n: int)-> np.ndarray:
    p_cd = np.zeros(n, dtype=int)
    for i in range(n):
        p_cd[i] = i
    return p_cd

def find(ind: int, p_cd: np.ndarray)-> int:
    while p_cd[ind] != ind:
        ind = p_cd[ind]
    return ind

def union(rep_1: int, rep_2: int, p_cd: np.ndarray)-> np.ndarray:
    root_1 = find(rep_1, p_cd)
    root_2 = find(rep_2, p_cd)
    if root_1 != root_2:
        p_cd[root_1] = root_2
    return p_cd

def create_pq(n: int, l_g: list)-> queue.PriorityQueue:
    pq = queue.PriorityQueue()
    for u, v, w in l_g:
        pq.put((w, (u, v)))
    return pq

def kruskal(n: int, l_g: list)-> tuple[int, list]:
    p_cd = init_cd(n)
    pq = create_pq(n, l_g)
    
    edges_used = 0
    l_t = []

    while not pq.empty() and edges_used < n - 1:
        dist, (u, v) = pq.get()
        if find(u, p_cd) != find(v, p_cd):
            union(u, v, p_cd)
            l_t.append((u, v, dist))
            edges_used += 1

    if edges_used == max(0, n - 1):
        return (n, l_t)

    return None

# --- Ejemplo de ejecución ---

n_ejemplo = 5 
lg_ejemplo = [
    (0, 1, 10),
    (0, 2, 6),
    (0, 3, 5),
    (1, 3, 15),
    (2, 3, 4),
    (0, 4, 2)
]

resultado_mst = kruskal(n_ejemplo, lg_ejemplo)

if resultado_mst:
    num_vertices, ramas = resultado_mst
    print("¡MST encontrado!")
    print(f"Vértices: {num_vertices}")
    print("Ramas del MST:")
    for rama in ramas:
        print(f"  {rama}")
else:
    print("No se pudo encontrar un MST (el grafo podría no ser conexo).")