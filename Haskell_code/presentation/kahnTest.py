import networkx as nx
from itertools import permutations
from collections import Counter
import random


def cons_algorithm(G : nx.DiGraph):
    # Get all sources in the graph (Nodes with in_degree = 0)
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

def cons_algorithm_alt(G : nx.DiGraph):
    # Get all sources in the graph (Nodes with in_degree = 0)
    topological_sort = []

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






def main():


    nodes = ["A", "B", "C", "D"]
    print(f"Vertices: {nodes}")

    # G = nx.DiGraph()
    # G.add_nodes_from(nodes)
    # G.add_edges_from([("A","C")])
    # print(cons_algorithm(G))

    my_permutations = list(permutations(nodes))
    print(f"# of Permutations: {len(my_permutations)}")
    topo_sorts = []

    for permutation in my_permutations:
        G = nx.DiGraph()
        G.add_nodes_from(permutation)
        G.add_edges_from([("A","C")])
        G.add_edges_from([("C","D")])
        topo_sorts.append(nx.topological_sort(G))

    list_of_tuples = [tuple(lst) for lst in topo_sorts]

    list_counts = Counter(list_of_tuples)

    # print("All Topological Sorts:")
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    # G.add_edges_from([("B","E")])
    # G.add_edges_from([("E","C")])
    G.add_edges_from([("A","C")])
    G.add_edges_from([("C","D")])
    print(f'Edges: {G.edges}')
    # # print(list(nx.all_topological_sorts(G)))
    print(f"# of All Possible Sorts: {len(list(nx.all_topological_sorts(G)))}")

    print("------------------------")
    print("Regular Kahn's")
    # # Print the counts
    print(f'Number of Unique Permutations Returned: {len(list_counts.items())}')
    for item, count in list_counts.items():
        print(f"{item}: {count} | Probability: {count/len(my_permutations)}")


    print("------------------------")
    print("Probabilistic Kahn's")
    topo_sorts = []
    for permutation in my_permutations:
        G = nx.DiGraph()
        G.add_nodes_from(permutation)
        G.add_edges_from([("A","C")])
        G.add_edges_from([("C","D")])
        topo_sorts.append(cons_algorithm(G))

    list_of_tuples = [tuple(lst) for lst in topo_sorts]

    list_counts = Counter(list_of_tuples)
    print(f'Number of Unique Permutations Returned: {len(list_counts.items())}')
    for item, count in list_counts.items():
        print(f"{item}: {count} | Probability: {count/len(my_permutations)}")


    print("------------------------")
    print("Probabilistic Kahn's Alt")
    topo_sorts = []
    for permutation in my_permutations:
        G = nx.DiGraph()
        G.add_nodes_from(permutation)
        G.add_edges_from([("A","C")])
        G.add_edges_from([("C","D")])
        topo_sorts.append(cons_algorithm_alt(G))

    list_of_tuples = [tuple(lst) for lst in topo_sorts]

    list_counts = Counter(list_of_tuples)
    print(f'Number of Unique Permutations Returned: {len(list_counts.items())}')
    for item, count in list_counts.items():
        print(f"{item}: {count} | Probability: {count/len(my_permutations)}")

if __name__ == "__main__":
    main()