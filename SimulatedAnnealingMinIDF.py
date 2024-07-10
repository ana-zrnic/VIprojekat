import networkx as nx
import matplotlib.pyplot as plt
import random
import math
import time
from GreedyMinIDF import *


def flip_vertices(idf, k):
    new_idf = idf.copy()
    for v in k:
        if new_idf[v] == 0:
            new_idf[v] = 1
        elif new_idf[v] == 1:
            new_idf[v] = random.choice([0, 2])  # Flip to either 0 or 2
        elif new_idf[v] == 2:
            new_idf[v] = 1  # Flip 2 to 1
    return new_idf


'''def is_dominating(G, idf):
    dominated = set()
    for v, label in idf.items():
        if label == 2:
            dominated.add(v)
            dominated.update(G.neighbors(v))
        elif label == 1:
            dominated.add(v)
    
    return dominated == set(G.nodes)'''

def extract_c1_c2(G):
    C1 = set()
    C2 = set()

    for node, data in G.nodes(data=True):
        if data['weight'] == 1:
            C1.add(node)
        elif data['weight'] == 2:
            C2.add(node)

    return C1, C2


def simulated_annealing_idf(G, initial_weights, T_init, alpha=0.95, beta=1e-6, gamma=0.1, phi=2*10**4, k=2, max_time=480):
    start_time = time.time()
    x_best = initial_weights.copy()
    x = x_best.copy()
    T = T_init

    #print(f"Initial solution: {x_best}")
    #print(f"Initial temperature: {T_init}")

    while time.time() - start_time < max_time:  # Run for max_time
        flip_vertices_list = random.sample(list(G.nodes), k)
        x_prime = flip_vertices(x, flip_vertices_list)

        #print(f"From: {x}")
        #print(f"Selected vertices to flip: {flip_vertices_list}")
        #print(f"x_prime: {x_prime}")

        C1, C2 = extract_c1_c2(G)
        nx.set_node_attributes(G, x_prime, 'weight')

        if not potential_function(G, (C1, C2)) == len(G.nodes()):
            #print(f"Flipped solution is not valid, skipping")
            nx.set_node_attributes(G, x, 'weight')  # Revert changes
            continue
        #else:
            #print(f"Flipped solution is valid, proceeding********************")

        current_weight = sum(x_prime.values())
        best_weight = sum(x.values())
        #print(f"Current weight: {current_weight}, Best weight: {best_weight}")

        if current_weight < best_weight:
            x = x_prime
            if current_weight < sum(x_best.values()):
                x_best = x_prime
                #print(f"New best solution: {x_best}")
        else:
            d = current_weight - best_weight
            if random.random() < math.exp(-d / T):
                x = x_prime
                #print(f"Accepted worse solution due to probability, new x: {x}")

        T *= alpha
        #print(f"Updated temperature: {T}")

        if T < 1e-10:
            T = T_init * beta
            #print(f"Temperature reset to: {T}")
    
    nx.set_node_attributes(G, x_best, 'weight')  # Finalize the best solution
    return x_best

def initial_temperature():
    return -2 / math.log(0.03)

def sa_minidf(G):
    T_init = initial_temperature()
    current_weight = greedy_minidf(G)
    #print("Best Greedy IDF solution weight:", current_weight)
    current_solution = nx.get_node_attributes(G, 'weight')
    best_sa_solution = simulated_annealing_idf(G,current_solution, T_init, alpha=0.95, beta=1e-6, gamma=0.1, phi=2*10**4, k=2, max_time=120)
    return sum(best_sa_solution.values())

def plot_graph(G, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    vertices = [1, 2, 3, 4, 5,6,7,8]
    edges = [(1, 2), (1, 8), (2, 8), (2, 4), (2,6), (2,3), (4, 8), (6, 8), (7, 8), (7,6), (3,4), (6,4), (5,4), (5,6)]
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    #plot_graph(G, "Graph for MinIDF Solution")

    T_init = initial_temperature()

    # Run randomized greedy phase for 2 minutes
    greedy_start_time = time.time()
    best_greedy_solution = None
    best_greedy_weight = float('inf')
    
    '''while time.time() - greedy_start_time < 120:
        current_solution,current_weight = greedy_minidf(G)
        #current_weight = sum(current_solution.values())
        if current_weight < best_greedy_weight:
            best_greedy_solution = current_solution
            best_greedy_weight = current_weight'''
    current_weight = greedy_minidf(G)
    #current_solution = {node:data['weight'] for node, data in G.nodes(data=True)}
    current_solution = nx.get_node_attributes(G, 'weight')

    print("Best Greedy IDF solution:", current_solution)
    print("Best Greedy IDF solution weight:", current_weight)

    # Run simulated annealing phase for 8 minutes
    best_sa_solution = simulated_annealing_idf(G,current_solution, T_init, alpha=0.95, beta=1e-6, gamma=0.1, phi=2*10**4, k=2, max_time=480)

    print("Best SA IDF solution:", {node:data['weight'] for node, data in G.nodes(data=True)})
    print("Best SA IDF solution weight:", sum(best_sa_solution.values()))
