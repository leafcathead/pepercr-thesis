import networkx as nx
import random
import numpy as np


vertices = ["liberate_case","spec_constr","rule_check2","late_specialise","triple_combo","late_dmd_anal","strict_anal","rule_check3","add_caller","add_late", "my_good_optimization", "my_neutral_optimization", "my_bad_optimization", "static_args","presimplify","specialise","full_laziness_1","simpl3","float_in_1","call_arity","strictness","exitification","full_laziness_2","cse","final","rule_check1"]


def generate_all_possible_rules():
    all_rules = []
    combined_list = vertices
    for opt_A in combined_list:
        for opt_B in combined_list:
            if opt_A != opt_B:
                all_rules.append((opt_A, opt_B))
    return all_rules

def generate_all_possible_valid_rules():
    # Uses the movable optimization list to create possible pairs. Does not touch the invalid list.
    all_rules = []
    for opt_A in vertices:
        for opt_B in vertices:
            if opt_A != opt_B:
                all_rules.append((opt_A, opt_B))
    return all_rules

def get_default_rules():
    return list(set(generate_all_possible_rules()) - set(generate_all_possible_valid_rules()))

def main():
    all_rules = generate_all_possible_rules()

    test_rules = [("liberate_case", "specialise"), ("rule_check2", "strictness"), ("triple_combo", "simpl3"), ("late_dmd_anal", "final"), ("strict_anal", "exitification"), ("add_caller", "rule_check1"), ("add_late", "static_args"), ("my_good_optimization", "full_laziness_1"), ("my_neutral_optimization", "call_arity"), ("my_bad_optimization", "float_in_1")]

    print(f'TEST RULES: {test_rules}')

    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(test_rules)
    sort_1 = list(nx.topological_sort(G))

    random.shuffle(vertices)
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(test_rules)
    sort_2 = list(nx.topological_sort(G))

    random.shuffle(vertices)
    G = nx.DiGraph()
    G.add_nodes_from(vertices)
    G.add_edges_from(test_rules)
    sort_3 = list(nx.topological_sort(G))

    print(f'Sort 1: {sort_1}')
    print(f'Sort 1: {sort_2}')
    print(f'Sort 1: {sort_3}')


if __name__ == "__main__":
    main()
