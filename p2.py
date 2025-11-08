import numpy as np
import queue
from queue import PriorityQueue
import random
import statistics, time
import itertools
from typing import List, Tuple, Union 

Coord = tuple[int, int]
Wall = tuple[Coord, Coord]

# --- TAD CONJUNTO DISJUNTO ---

def init_cd(n: int) -> np.ndarray:
    """
    Inicializa la estructura del Conjunto Disjunto (CD) para 'n' elementos.
    Cada elemento se establece como su propio representante/raíz.
    """
    p_cd = np.arange(n, dtype=int)
    return p_cd

def find(ind: int, p_cd: np.ndarray) -> int:
    """
    Determina la raíz/representante de la categoría a la que pertenece el elemento 'ind' (operación Búsqueda).
    Utiliza la estructura de árbol de padres, siguiendo el puntero hasta la raíz.
    """
    while p_cd[ind] != ind:
        ind = p_cd[ind]
    return ind

def union(rep_1: int, rep_2: int, p_cd: np.ndarray):
    """
    PROCEDIMIENTO que une dos categorías (conjuntos) en una sola (operación Unión).
    Modifica la estructura 'p_cd' in-place, haciendo que la raíz de un conjunto apunte a la del otro.
    """
    root_1 = find(rep_1, p_cd)
    root_2 = find(rep_2, p_cd)
    
    if root_1 != root_2:
        p_cd[root_1] = root_2 
    

# --- ALGORITMO DE KRUSKAL (MST) ---

def create_pq(n: int, l_g: List) -> PriorityQueue:
    """
    Crea una Cola de Prioridad (PQ) a partir de la lista de aristas 'l_g'.
    Las aristas se insertan ordenadas por su peso, para su uso en Kruskal.
    """
    pq = PriorityQueue()
    for u, v, w in l_g:
        pq.put((w, (u, v)))
    return pq

def kruskal(n: int, l_g: List) -> Union[Tuple[int, List], None]:
    """
    Implementación estándar del Algoritmo de Kruskal para encontrar el Árbol Abarcador de Peso Mínimo (MST).
    Utiliza el TAD Conjunto Disjunto para detectar y evitar ciclos, garantizando el MST.
    """
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

def complete_graph(n_nodes: int, max_weight=50) -> Tuple[int, List]:
    """
    Genera un grafo completo no dirigido con 'n_nodes' y pesos aleatorios, usado para pruebas de Kruskal.
    Devuelve la tupla (número_de_nodos, lista_de_aristas en formato (u, v, w)).
    """
    l_g = []
    for i in range (n_nodes):
        for j in range (i+1, n_nodes):
            w = random.randint(1, max_weight)
            l_g.append((i, j, w))
        
    return (n_nodes, l_g)

def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int) -> List[float]:
    """
    Mide el tiempo de ejecución promedio del algoritmo de Kruskal para grafos de distintos tamaños.
    Se utiliza para evaluar experimentalmente la eficiencia del algoritmo.
    """
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
    """Devuelve la representación canónica (ordenada) de un muro. (Función auxiliar)"""
    return tuple(sorted((a, b)))

def create_maze(m: int, n: int) -> List[Wall]:
    """
    Genera un Laberinto Perfecto (sin ciclos y conexo, un árbol abarcador) en una cuadrícula m x n.
    Utiliza una adaptación de Kruskal: baraja las paredes aleatoriamente y usa el TAD Conjunto Disjunto para
    unir celdas sin crear ciclos al eliminar paredes.
    Devuelve la lista de paredes que DEBEN CONSERVARSE.
    """
    num_cells = m * n
    all_walls: List[Wall] = []
    
    if num_cells == 0:
        return []
    
    p_cd = init_cd(num_cells)
    maze: List[Wall] = []
    num_components = num_cells

    for r in range(m):
        for c in range(n - 1):
            all_walls.append(canonical_wall((r, c), (r, c + 1)))
    for r in range(m - 1):
        for c in range(n):
            all_walls.append(canonical_wall((r, c), (r + 1, c)))
            
    random.shuffle(all_walls)

    for wall in all_walls:
        if num_components == 1:
            maze.append(wall)
            continue

        (r1, c1), (r2, c2) = wall
    
        idx_a = r1 * n + c1
        idx_b = r2 * n + c2

        root_a = find(idx_a, p_cd)
        root_b = find(idx_b, p_cd)

        if root_a != root_b:
            union(root_a, root_b, p_cd) 
            num_components -= 1
        else:
            maze.append(wall)
            
    return maze

def draw_maze(m: int, n: int, maze: list[Wall], wall_color: str = "black"):
    """
    Dibuja un laberinto representado como una lista de muros (Función de visualización auxiliar).
    """
    import matplotlib.pyplot as plt
    
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
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
    return None

def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray: 
    """
    Genera una matriz de distancias aleatoria, simétrica y con diagonal cero, adecuada para problemas TSP.
    """
    m = np.random.randint(1, w_max+1, (n_nodes, n_nodes)) 
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0) 
    return m

def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    """
    Aplica el algoritmo del Vecino Más Cercano (Nearest Neighbor) al TSP.
    
    Este enfoque utiliza la estructura de Conjuntos Disjuntos (DSU) de manera no estándar: 
    selecciona el vecino más cercano y usa DSU para evitar que el camino retorne a 
    un componente ya conectado (potencial ciclo). Si se detecta un ciclo por DSU, 
    la distancia de esa arista se anula temporalmente. El proceso construye un circuito 
    cerrado partiendo de 'node_ini'.
    """
    n_nodes = dist_m.shape[0]
    visited = {node_ini}
    camino = [node_ini]
    current_node = node_ini

    while len(visited) < n_nodes:
        min_dist = np.inf
        next_node = -1

        for next_candidate in range(n_nodes):
            if next_candidate not in visited:
                dist = dist_m[current_node, next_candidate]
                
                if dist < min_dist:
                    min_dist = dist
                    next_node = next_candidate
        
        if next_node == -1:
            break

        current_node = next_node
        visited.add(current_node)
        camino.append(current_node)

    if len(camino) == n_nodes:
        camino.append(node_ini)
        
    return camino

def len_circuit(circuit: list, dist_m: np.ndarray)-> int:
    """
    Calcula la longitud total de un circuito TSP (suma de las distancias entre nodos consecutivos).
    """
    sum_dist = 0
    for i in range(len(circuit) - 1):
        sum_dist = sum_dist + dist_m[circuit[i]][circuit[i+1]]
        
    return sum_dist   
    
def repeated_greedy_tsp(dist_m: np.ndarray) -> List[int]:
    """
    Implementa el algoritmo de Vecino Más Cercano Repetitivo.
    Aplica el algoritmo 'greedy_tsp' partiendo de CADA nodo del grafo
    y devuelve el trayecto (camino) con la menor longitud total encontrada.
    """
    min_camino = None
    dist = None
    
    n_nodes = dist_m.shape[0]
    
    for i in range(n_nodes):
        dist_m_copy = np.copy(dist_m) 
        camino = greedy_tsp(dist_m_copy, i)
        length = len_circuit(camino, dist_m) 

        if dist is None or length < dist:
            dist = length
            min_camino = camino

    return min_camino

def exhaustive_tsp(dist_m: np.ndarray) -> List[int]:
    """
    Implementa el algoritmo Exhaustivo para el TSP.
    Examina todos los posibles circuitos (permutaciones) que inician en el primer nodo (índice 0)
    y devuelve aquel con la distancia más corta (solución óptima garantizada).
    """
    nodes = list(range(dist_m.shape[0]))
    min_camino = None
    dist = None
    
    for perm in itertools.permutations(nodes[1:]):
        camino = [nodes[0]] + list(perm) + [nodes[0]]
        length = len_circuit(camino, dist_m)
        
        if dist is None or length < dist:
            dist = length
            min_camino = camino
            
    return min_camino
