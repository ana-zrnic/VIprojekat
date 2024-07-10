import networkx as nx
import random
import numpy as np

def generate_grid_graph(m, n):
    graph = nx.grid_2d_graph(m, n)
    return nx.convert_node_labels_to_integers(graph) #inace su cvorovi oznaceni kao koordinate/tuple, ideja za konverziju preuzeta sa: https://stackoverflow.com/questions/37104997/changing-pos-while-changing-node-labels-in-grid-2d-graph-networkx

def generate_bipartite_graph(num_nodes_1, num_nodes_2, probability):
    graph = nx.bipartite.random_graph(num_nodes_1, num_nodes_2, probability)
    return graph

def generate_random_graph(num_nodes, probability):
    graph = nx.gnp_random_graph(num_nodes, probability)
    return graph

def generate_planar_graph(num_nodes):
    # Using random geometric graph as a proxy for a planar graph
    radius = np.sqrt(2 * np.log(num_nodes) / num_nodes)
    graph = nx.random_geometric_graph(num_nodes, radius)
    return nx.convert_node_labels_to_integers(graph)

def generate_barabasi_albert_graph(num_nodes, num_edges_to_attach): #izbor specificnog generatora objasnjeno na:https://www.geeksforgeeks.org/barabasi-albert-graph-scale-free-models/
    graph = nx.barabasi_albert_graph(num_nodes, num_edges_to_attach)
    return graph

def generate_recursive_tree(num_nodes):
    graph = nx.full_rary_tree(2, num_nodes)
    return graph

def create_list_of_random_graphs():
    # Generate a list of random graphs for testing
    graphs = []
    graph_types = [
        ("grid", generate_grid_graph, 15),
        ("bipartite", generate_bipartite_graph, 15),
        ("random", generate_random_graph, 15),
        ("planar", generate_planar_graph, 3),
        ("barabasi_albert", generate_barabasi_albert_graph, 3),
        ("recursive_tree", generate_recursive_tree, 3),
    ]
    
    sizes = [
        ("small", 10, 30),  # Small graphs with nodes between 5 and 10
        ("medium", 31, 50),  # Medium graphs with nodes between 11 and 20
        ("large", 51, 70)   # Large graphs with nodes between 21 and 30
    ]

    for graph_type, generator, count in graph_types:
        for size_label, min_nodes, max_nodes in sizes:
            instances_per_size = count // 3
            for _ in range(instances_per_size):  # Generate two graphs for each type and size
                num_nodes = np.random.randint(min_nodes, max_nodes)
                
                if graph_type == "grid":
                    m = np.random.randint(2, num_nodes // 2)
                    n = num_nodes // m
                    graph = generator(m, n)
                elif graph_type == "bipartite":
                    num_nodes_1 = num_nodes // 2
                    num_nodes_2 = num_nodes - num_nodes_1
                    probability = 0.3  # Example probability for edge creation
                    graph = generator(num_nodes_1, num_nodes_2, probability)
                elif graph_type == "random":
                    probability = 0.3  # Example probability for edge creation
                    graph = generator(num_nodes, probability)
                elif graph_type == "planar":
                    graph = generator(num_nodes)
                elif graph_type == "barabasi_albert":
                    num_edges_to_attach = np.random.randint(1, 5)  # Example attachment
                    graph = generator(num_nodes, num_edges_to_attach)
                elif graph_type == "recursive_tree":
                    graph = generator(num_nodes)
                
                graphs.append((graph, graph_type, size_label))
    
    return graphs



