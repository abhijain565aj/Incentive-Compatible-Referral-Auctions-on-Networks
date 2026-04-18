
from collections import defaultdict
import networkx as nx


def all_nodes(seller_net, reports):
    nodes = set(seller_net)
    for u, vs in reports.items():
        nodes.add(u)
        nodes.update(vs)
    return nodes


def truthful_owner(seller_net, reports):
    return {n: n for n in all_nodes(seller_net, reports)}


def utility_single_unit(allocations, payments, owner, true_values):
    """Utility aggregated at the real-buyer level under unit demand."""
    utility = defaultdict(float)
    won = defaultdict(bool)
    for node, allocated in allocations.items():
        if allocated:
            real = owner[node]
            if not won[real]:
                utility[real] += true_values[real]
                won[real] = True
    for node, payment in payments.items():
        utility[owner[node]] -= payment
    return utility


def real_welfare(allocations, owner, true_values):
    winners = {owner[n] for n, a in allocations.items() if a}
    return sum(true_values[w] for w in winners)


def fake_winner_count(allocations):
    return sum(1 for n, a in allocations.items() if a and n.startswith("syb"))


def to_networkx_graph(seller_net, reports):
    g = nx.Graph()
    g.add_node("s")
    for n in seller_net:
        g.add_edge("s", n)
    for u, vs in reports.items():
        for v in vs:
            g.add_edge(u, v)
    return g
