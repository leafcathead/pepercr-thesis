import networkx as nx
from itertools import permutations
from collections import Counter
import random
import pandas as pd



def cons_algorithm(G_old : nx.DiGraph):
    # Get all sources in the graph (Nodes with in_degree = 0)
    G = G_old.copy()
    topological_sort = []

    if not nx.is_directed_acyclic_graph(G):
        raise TypeError("Graph should be a DAG, this graph contains a cycle!")

    while (G.order() > 0):
        sources = [node for node in G.nodes if G.in_degree(node) == 0]
        # Calculate Weights for random choice. Weight is based on minimum other nodes could be visited from that node, including itself.
        weights = [1 + G.out_degree(node) for node in sources]
        tmp = sum(weights)
        weights = list(map(lambda x: x/tmp, weights))
        topo_elem = random.choices(sources, weights)[0]
        # Erase all edges from our selected source
        topological_sort.append(topo_elem)
        G.remove_edges_from(list(map(lambda x: (topo_elem, x), list(G.successors(topo_elem)))))
        # Remove selected node from the Graph
        G.remove_node(topo_elem)
        # Repeat
    return topological_sort

def cons_algorithm_alt(G_old : nx.DiGraph):
    # Get all sources in the graph (Nodes with in_degree = 0)
    topological_sort = []
    G = G_old.copy()

    if not nx.is_directed_acyclic_graph(G):
        raise TypeError("Graph should be a DAG, this graph contains a cycle!")

    while (G.order() > 0):
        sources = [node for node in G.nodes if G.in_degree(node) == 0]
        # print(f'Sources: {sources}')
        # Calculate Weights for random choice. Weight is based on minimum other nodes could be visited from that node, including itself.
        # weights = [1 + G.out_degree(node) for node in sources]
        # tmp = sum(weights)
        # weights = list(map(lambda x: x/tmp, weights))
        # print(f'Weights: {weights}')
        topo_elem = random.choices(sources)[0]
        # print(f'Selected Node: {topo_elem}')
        # Erase all edges from our selected source
        topological_sort.append(topo_elem)
        G.remove_edges_from(list(map(lambda x: (topo_elem, x), list(G.successors(topo_elem)))))
        # print(f'Edges: {G.edges}')
        # Remove selected node from the Graph
        G.remove_node(topo_elem)
        # Repeat
    return topological_sort


def transform_topo_sort(lst):
    return '|'.join(lst)

def main():
    nodes = ["A", "B", "C", "D"]
    topo_sorts = []
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([("A","C")])
    G.add_edges_from([("C","D")])


    for i in range(0,4):
        topo_sorts.append(cons_algorithm(G))
    print(topo_sorts)
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_4.csv", index=False)
    topo_sorts = []
    print("4 Test Done")

    for i in range(0,10):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_10.csv", index=False)
    topo_sorts = []
    print("10 Test Done")

    for i in range(0,350):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_350.csv", index=False)
    topo_sorts = []
    print("350 Test Done")

    for i in range(0,1000):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_1000.csv", index=False)
    topo_sorts = []
    print("1000 Test Done")

    for i in range(0,10000):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_10000.csv", index=False)
    topo_sorts = []
    print("10000 Test Done")

    for i in range(0,50000):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_50000.csv", index=False)
    topo_sorts = []
    print("50000 Test Done")

    for i in range(0,100000):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_100000.csv", index=False)
    topo_sorts = []
    print("100000 Test Done")

    for i in range(0,1000000):
        topo_sorts.append(cons_algorithm(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_1000000.csv", index=False)
    topo_sorts = []
    print("1000000 Test Done")


    print("---------------------------------")

    for i in range(0,4):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_4_alt.csv", index=False)
    topo_sorts = []
    print("4 Test Done - Alt")

    for i in range(0,10):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_10_alt.csv", index=False)
    topo_sorts = []
    print("10 Test Done - Alt")

    for i in range(0,350):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_350_alt.csv", index=False)
    topo_sorts = []
    print("350 Test Done - Alt")

    for i in range(0,1000):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_1000_alt.csv", index=False)
    topo_sorts = []
    print("1000 Test Done - Alt")

    for i in range(0,10000):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_10000_alt.csv", index=False)
    topo_sorts = []
    print("10000 Test Done - Alt")

    for i in range(0,50000):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_50000_alt.csv", index=False)
    topo_sorts = []
    print("50000 Test Done - Alt")


    for i in range(0,100000):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_100000_alt.csv", index=False)
    topo_sorts = []
    print("100000 Test Done - Alt")

    for i in range(0,1000000):
        topo_sorts.append(cons_algorithm_alt(G))
    df = pd.DataFrame(list(map(transform_topo_sort, topo_sorts)), columns=["Sort"])
    df.to_csv("LLN_Test_1000000_alt.csv", index=False)
    topo_sorts = []
    print("1000000 Test Done - Alt")

    print("All Test Complete")



if __name__ == "__main__":
    main()