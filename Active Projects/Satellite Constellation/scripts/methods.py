import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

def rand_connected_graph(num_nodes: int) -> nx.classes.graph.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
    num_additional_edges = 5
    for _ in range(num_additional_edges):
        node1 = np.random.randint(0, num_nodes - 1)
        node2 = np.random.randint(0, num_nodes - 1)
        if node1 != node2:
            G.add_edge(node1, node2)

    return G

def plot_graph(G: nx.classes.graph.Graph) -> None:
    nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')