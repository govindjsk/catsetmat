import networkx as nx
# from src.node2vec import *
import numpy as np
from tqdm.autonotebook import tqdm

def add_weight(G, u, v):
    if 'weight' not in G[u][v]:
        G[u][v]['weight'] = 1
    else:
        G[u][v]['weight'] += 1 

def read_graph(nodelist, hyperedge_list):
    '''
    Transfer the hyperedge to pairwise edge & Reads the input network in networkx.
    '''
    G = nx.Graph()
    # tot = sum(num)
    G.add_nodes_from(nodelist)
    for ee in tqdm(hyperedge_list):
        e = ee
        edges_to_add = []
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                edges_to_add.append((e[i], e[j]))
        G.add_edges_from(edges_to_add)
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                add_weight(G, e[i], e[j])

    G = G.to_undirected()

    return G

def read_graph_cross(node_list,U,V):
    G = nx.Graph()
    # tot = sum(num)
    G.add_nodes_from(node_list)
    for i,u in enumerate(U):
        edges_to_add=[]
        v=V[i]
        for x in u:
            for j in v:
                edges_to_add.append((x,j,1))
        G.add_weighted_edges_from(edges_to_add)
        # for i in range(len(e)):
        #     for j in range(i + 1, len(e)):
        #         add_weight(G, e[i], e[j])

    G = G.to_undirected()

    return G


def toint(hyperedge_list):
    return np.array([h.astype('int') for h in hyperedge_list])
