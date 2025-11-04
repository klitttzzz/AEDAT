import numpy as np
import queue
from queue import PriorityQueue
import random
import statistics, time
import matplotlib.pyplot as plt
import itertools

Coord = tuple[int, int]
Wall = tuple[Coord, Coord]


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

def complete_graph(n_nodes: int, max_weight=50)-> tuple[int, list]:
    l_g = []
    for i in range (n_nodes):
        for j in range (i+1, n_nodes):
            w = random.randint(1, max_weight)
            l_g.append((i, j, w))
        
    return (n_nodes, l_g)

def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int)-> list:
    lst = [] 
    for i in range(n_nodes_ini, n_nodes_fin, step):
        tiempos = []
        for j in range(n_graphs):
            graph = complete_graph(i, 50)
            t1 = time.time()
            k = kruskal(graph[0], graph[1])
            t2 = time.time()
            tiempos.append(t2 - t1)
        lst.append((statistics.mean(tiempos)))
    return lst

def canonical_wall(a: Coord, b: Coord) -> Wall:
    """Devuelve la representación canónica (ordenada) de un muro."""
    return tuple(sorted((a, b)))

def create_maze(m: int, n: int) -> list[Wall]:
    """
    Genera un laberinto perfecto (sin ciclos, conexo) de tamaño m x n
    usando una adaptación de Kruskal y tus funciones de Conjuntos Disjuntos.
    """
    num_cells = m * n
    if num_cells == 0:
        return []

    # 3. Inicializar la estructura de conjuntos disjuntos
    p_cd = init_cd(num_cells)
    
    # 1. Generar todas las posibles paredes internas
    all_walls: list[Wall] = []
    for r in range(m):
        for c in range(n - 1): # Paredes verticales
            all_walls.append(canonical_wall((r, c), (r, c + 1)))
    for r in range(m - 1):
        for c in range(n): # Paredes horizontales
            all_walls.append(canonical_wall((r, c), (r + 1, c)))
            
    # 2. Desordenar las paredes aleatoriamente
    random.shuffle(all_walls)
    
    maze_walls_to_keep: list[Wall] = []
    num_components = num_cells

    # 4. Recorrer cada pared
    for wall in all_walls:
        # 5. Salir pronto si ya está todo conectado
        if num_components == 1:
            maze_walls_to_keep.append(wall) # Añadir las paredes restantes
            continue

        (r1, c1), (r2, c2) = wall
        
        # Convertir coordenadas (r, c) a índice 1D
        idx_a = r1 * n + c1
        idx_b = r2 * n + c2

        # • Si 𝑎 y 𝑏 pertenecen a subconjuntos distintos...
        root_a = find(idx_a, p_cd)
        root_b = find(idx_b, p_cd)
        
        if root_a != root_b:
            # ...unirlos (eliminando la pared)
            p_cd = union(root_a, root_b, p_cd) # Reasignar p_cd
            num_components -= 1
        else:
            # • Si ya pertenecen al mismo conjunto, conservar la pared
            maze_walls_to_keep.append(wall)
            
    return maze_walls_to_keep

def draw_maze(m: int, n: int, maze: list[Wall], wall_color: str = "black"):
    """
    Dibuja un laberinto representado como una lista de muros.
    (Función proporcionada en el enunciado)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    wall_set = {canonical_wall(a, b) for a, b in maze}
    
    # Dibujar borde exterior
    ax.plot([0, n], [0, 0], color=wall_color)
    ax.plot([0, n], [m, m], color=wall_color)
    ax.plot([0, 0], [0, m], color=wall_color)
    ax.plot([n, n], [0, m], color=wall_color)
    
    # Dibujar muros internos
    for (r1, c1), (r2, c2) in wall_set:
        if r1 == r2 and abs(c1 - c2) == 1:
            # Muro vertical
            x = min(c1, c2) + 1
            y0, y1 = r1, r1 + 1
            ax.plot([x, x], [y0, y1], color=wall_color)
        elif c1 == c2 and abs(r1 - r2) == 1:
            # Muro horizontal
            y = min(r1, r2) + 1
            x0, x1 = c1, c1 + 1
            ax.plot([x0, x1], [y, y], color=wall_color)
            
    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.invert_yaxis() # Fila 0 aparece arriba
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    return None

num_rows = 10
num_cols = 10

print(f"Generando laberinto de {num_rows}x{num_cols}...")
maze = create_maze(num_rows, num_cols)
print("¡Laberinto generado! Mostrando...")

draw_maze(num_rows, num_cols, maze)

def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray: 
    m = np.random.randint(1, w_max+1, (n_nodes, n_nodes)) 
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0) 
    return m

def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    i = 0
    k = node_ini
    min_node = None
    num_min = None
    camino = []
    camino.append(node_ini)
    cd_l = init_cd(len(dist_m[0]))
    
    while(i < len(dist_m[0])-1):
        num_min = min(n for n in dist_m[k] if n != 0)
        for j in range(dist_m[k]):
            if(num_min == dist_m[k][j]):
                min_node = j
                
        if(find(min_node, cd_l) == find(camino[i], cd_l)):
            dist_m[k][min_node] = 0
        else:
            union(min_node, k)
            i = i+1
            camino.append(min_node)
            k = min_node
            
    camino.append(node_ini)
    
    return camino
    
def len_circuit(circuit: list, dist_m: np.ndarray)-> int:
    sum = 0
    for i in range(len(circuit) - 1):
        sum = sum + dist_m[circuit[i]][dist_m[circuit[i+1]]]
        
    return sum   
    
def repeated_greedy_tsp(dist_m: np.ndarray)-> list:
def exhaustive_tsp(dist_m: np.ndarray)-> list: