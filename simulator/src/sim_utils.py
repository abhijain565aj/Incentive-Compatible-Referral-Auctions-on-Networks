
import random
import networkx as nx


def generate_graph(n=20, model="tree", seller_deg=3, p=0.18, m=2, seed=None):
    rng = random.Random(seed)

    if model == "tree":
        g = nx.random_labeled_tree(n, seed=seed)
    elif model == "er":
        while True:
            g = nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))
            if nx.is_connected(g):
                break
    elif model == "ba":
        g = nx.barabasi_albert_graph(n, max(1, m), seed=seed)
    else:
        raise ValueError(f"Unknown model: {model}")

    nodes = [f"b{i}" for i in range(n)]
    mapping = {i: nodes[i] for i in range(n)}
    g = nx.relabel_nodes(g, mapping)

    seller_neighbors = rng.sample(nodes, min(seller_deg, n))
    reports = {u: sorted(list(g.neighbors(u))) for u in g.nodes()}
    bids = {u: rng.randint(1, 100) for u in g.nodes()}
    return seller_neighbors, reports, bids, g


def pick_attacker(g):
    candidates = [n for n, d in g.degree() if d >= 2]
    if not candidates:
        return max(g.degree, key=lambda x: x[1])[0]
    bc = nx.betweenness_centrality(g)
    return max(candidates, key=lambda n: (bc[n], g.degree[n], n))
