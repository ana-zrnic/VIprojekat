import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from GreedyMinIDF import *

def initialize_pheromone(G):
    pheromone = {v: 0.5 for v in G.nodes()}
    return pheromone

def construct_solution(G, pheromone, drate=0.8):
    for v in G.nodes():
        G.nodes[v]['weight'] = -1  # Initially, all vertices are unlabeled
    V_prime = list(G.nodes())
    
    while V_prime:
        u = choose_vertex(G, pheromone, V_prime, drate)
        G.nodes[u]['weight'] = 2
        for neighbor in G.neighbors(u):
            if G.nodes[neighbor]['weight'] == -1:  # Only update if the neighbor is unlabeled
                G.nodes[neighbor]['weight'] = 0
        V_prime = [v for v in V_prime if G.nodes[v]['weight'] == -1]  # Update V_prime to only include unlabeled vertices

def choose_vertex(G, pheromone, V_prime, drate):
    r = random.random()
    if r <= drate:
        # Select the vertex that maximizes f(u) = deg(u) * tau_u
        selected_vertex = max(V_prime, key=lambda v: G.degree[v] * pheromone[v])
    else:
        # Roulette wheel selection
        f_values = {v: G.degree[v] * pheromone[v] for v in V_prime}
        total_f = sum(f_values.values())
        
        if total_f == 0:
            # If total_f is zero, assign equal probabilities to all vertices
            probabilities = {v: 1.0 / len(V_prime) for v in V_prime}
        else:
            probabilities = {v: f / total_f for v, f in f_values.items()}
        
        selected_vertex = roulette_wheel_selection(probabilities)
    
    return selected_vertex

def roulette_wheel_selection(probabilities):
    rand = random.random()
    cumulative_prob = 0.0
    for v, prob in probabilities.items():
        cumulative_prob += prob
        if rand < cumulative_prob:
            return v

def extend_solution(G):
    V01 = [v for v in G.nodes() if G.nodes[v]['weight'] in [0, 1]]
    iters = int(0.1 * len(V01))
    
    for _ in range(iters):
        if not V01:
            break
        u = random.choice(V01)
        if G.nodes[u]['weight'] == 0:
            G.nodes[u]['weight'] = 1
        else:
            G.nodes[u]['weight'] = 2
        V01.remove(u)


def reduce_solution(G):
    V_prime = sorted(G.nodes(), key=lambda v: G.degree[v])
    
    for v in V_prime:
        if G.nodes[v]['weight'] == 2:
            init_label = G.nodes[v]['weight']
            G.nodes[v]['weight'] = 1  # First try reducing to 1
            
            # Translate node weights into C1 and C2
            C1 = {u for u in G.nodes() if G.nodes[u]['weight'] == 1}
            C2 = {u for u in G.nodes() if G.nodes[u]['weight'] == 2}

            if potential_function(G, (C1, C2)) != len(G.nodes()):
                G.nodes[v]['weight'] = 0  # Then try reducing to 0
                if potential_function(G, (C1, C2)) != len(G.nodes()):
                    G.nodes[v]['weight'] = init_label  # Revert if neither reduction is feasible

def calculate_d(n, k, k_max, d_min, d_max):
    return d_min + ((k - 1) / (k_max - 1)) * (d_max - d_min)

def destroy_solution(G, d):
    n = len(G.nodes())
    num_to_unlabel = math.ceil(n * d)
    V_prime = [v for v in G.nodes() if G.nodes[v]['weight'] in [0, 1]]
    
    for _ in range(num_to_unlabel):
        if not V_prime:
            break
        u = random.choice(V_prime)
        G.nodes[u]['weight'] = -1
        V_prime.remove(u)

def random_variable_neighborhood_search(G, pheromone, d_min, d_max, drate, k_max, max_noimpr, max_itr):
    n = len(G.nodes())
    best_solution = {v: G.nodes[v]['weight'] for v in G.nodes()}
    best_weight = sum(G.nodes[v]['weight'] for v in G.nodes())
    cnoimpr = 0

    for k in range(1, max_itr + 1):
        d = calculate_d(n, k, k_max, d_min, d_max)
        destroy_solution(G, d)
        construct_solution(G, pheromone, drate)
        extend_solution(G)
        reduce_solution(G)

        C1 = {u for u in G.nodes() if G.nodes[u]['weight'] == 1}
        C2 = {u for u in G.nodes() if G.nodes[u]['weight'] == 2}

        if potential_function(G, (C1, C2)) == len(G.nodes()):
            new_weight = sum(G.nodes[v]['weight'] for v in G.nodes())
            if new_weight < best_weight:
                best_solution = {v: G.nodes[v]['weight'] for v in G.nodes()}
                best_weight = new_weight
                cnoimpr = 0
            else:
                cnoimpr += 1

        if cnoimpr >= max_noimpr:
            break

    for v in G.nodes():
        G.nodes[v]['weight'] = best_solution[v]


def aco_mdr(G):
    pheromone = initialize_pheromone(G)
    curr_best_sol = float('inf')
    best_sol = float('inf')
    best_solution = None
    tau_min = 0.1
    tau_max = 1.0
    Q = 1.0  # Constant for pheromone update
    d_min = 0.1  # Minimum value of d
    d_max = 0.5  # Maximum value of d
    k_max = 100  # Maximum number of iterations
    max_noimpr = 10  # Maximum number of iterations without improvement
    max_itr = 100  # Maximum number of iterations
    drate = 0.8  # drate parameter

    for k in range(1, k_max + 1):
        construct_solution(G, pheromone, drate)
        extend_solution(G)
        reduce_solution(G)
        random_variable_neighborhood_search(G, pheromone, d_min, d_max, drate, k_max, max_noimpr, max_itr)

        sol_sum = sum(G.nodes[v]['weight'] for v in G.nodes())
        if sol_sum < curr_best_sol:
            curr_best_sol = sol_sum
            if curr_best_sol < best_sol:
                best_sol = curr_best_sol
                best_solution = {v: G.nodes[v]['weight'] for v in G.nodes()}
        
        update_pheromone(pheromone, best_solution, curr_best_sol, best_sol, 0.1, tau_min, tau_max, Q)

    return sum(best_solution.values())

def update_pheromone(pheromone, best_solution, K_curr_best_sol, K_best_sol, evaporation_rate, tau_min, tau_max, Q):
    for key in pheromone:
        pheromone[key] *= (1 - evaporation_rate)
    
    for v in best_solution:
        pheromone[v] += Q / K_curr_best_sol
    
    for v in best_solution:
        pheromone[v] += Q / K_best_sol
    
    conv_fact = compute_convergence(pheromone, tau_min, tau_max)
    if conv_fact > 0.9:  # Example threshold for convergence
        reinitialize_pheromones(pheromone, 0.5)

def reinitialize_pheromones(pheromone, initial_value):
    for key in pheromone:
        pheromone[key] = initial_value

def compute_convergence(pheromone, tau_min, tau_max):
    n = len(pheromone)
    sum_max_diff = sum(max(tau_max - pheromone[v], pheromone[v] - tau_min) for v in pheromone)
    conv_fact = 2 * (sum_max_diff / (n * (tau_max + tau_min))) - 1
    return conv_fact



def plot_graph(G, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Example usage
    #G = nx.erdos_renyi_graph(10, 0.3)
    G=nx.Graph()
    #G.add_nodes_from([1, 2, 3, 4, 5,6,7,8])
    #G.add_edges_from([(1, 2), (2,3), (3, 8), (3, 4), (4,5), (2,3), (5,6), (6, 7), (7, 8)])
    G.add_nodes_from([1, 2, 3, 4, 5,6,7,8])
    G.add_edges_from([(1, 2), (1, 8), (2, 8), (2, 4), (2,6), (2,3), (4, 8), (6, 8), (7, 8), (7,6), (3,4), (6,4), (5,4), (5,6)])
    
    
    solution = aco_mdr(G)
    print("Best ACO MDR solution:", {node:data['weight'] for node, data in G.nodes(data=True)})
    print("Solution weight:", solution)

    plot_graph(G, "Graph for ACO MDR Solution")

 
