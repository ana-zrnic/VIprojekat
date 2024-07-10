import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def domination_degree(G, C):
    DdegC = {v: 0 for v in G.nodes()}
    for v in G.nodes():
        if v in C[0] or v in C[1]: #slucaj da v pripada C, tada on nije vrijednosti 0 i totalno je dom
            DdegC[v] = 1
        elif any(u in C[1] for u in G.neighbors(v)): #v ne pripada C (v=0), provjeri ima li bar jedno susjeda v=2, ideja za any() : https://stackoverflow.com/questions/1342601/pythonic-way-of-checking-if-a-condition-holds-for-any-element-of-a-list
            DdegC[v] = 1
        elif sum(1 for u in G.neighbors(v) if u in C[0]) >= 2: #v ne pripada C (v=0), provjeri ima li dva ili vise susjeda v=1
            DdegC[v] = 1
        elif sum(1 for u in G.neighbors(v) if u in C[0]) == 1: #v ne pripada C (v=0), provjeri ima li tacno jednog susjeda v=1, parcijalna dom
            DdegC[v] = 0.5
    return DdegC

def potential_function(G, C):
    DdegC = domination_degree(G, C)
    return sum(DdegC.values())

def greedy_minidf(G):
    C1, C2 = set(), set()
    while potential_function(G, (C1, C2)) < len(G.nodes()):
        best_v = max(G.nodes(), key=lambda v: potential_function(G, (C1 | {v} if v not in C1 else C1, C2 | {v} if v not in C2 else C2))) #ideja za koristenje max i key=lambda v sa linka: https://stackoverflow.com/questions/18296755/python-max-function-using-key-and-lambda-expression 
        if best_v in C1:
            C1.remove(best_v)
            C2.add(best_v)
        else:
            C1.add(best_v)

    nx.set_node_attributes(G, 0, 'weight')
    for v in C1:
        G.nodes[v]['weight'] = 1
    for v in C2:
        G.nodes[v]['weight'] = 2

    sum_weights = sum(1 for v in C1) + sum(2 for v in C2)
    #sum_weights = sum(nx.get_node_attributes(G, 'weight').values())
    return sum_weights



def plot_graph(G, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Example usage
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5,6,7,8])
    G.add_edges_from([(1, 2), (1, 8), (2, 8), (2, 4), (2,6), (2,3), (4, 8), (6, 8), (7, 8), (7,6), (3,4), (6,4), (5,4), (5,6)])

    sum_weights = greedy_minidf(G)
    minidf_solution = {node:data['weight'] for node, data in G.nodes(data=True)}
    print("MinIDF solution:", minidf_solution)
    print("Sum of all node weights:", sum_weights)

    plot_graph(G, "Graph for MinIDF Solution")