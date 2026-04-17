
"""
Adapted from the MIT-licensed repository:
https://github.com/bcirillo99/Homogeneous-Multi-unit-Auction-over-a-Social-Network

Original repository implements VCG, GIDM, MUDAN, and MUDAR over a social network.
This file keeps the GIDM logic only, repackaged as a local module with cleaner names
and minor documentation edits. See LICENSE-THIRD-PARTY.txt in the project root.
"""

from collections import deque 
import copy
import networkx as nx
from .bidder import Bidder


class CriticalTree:
    def __init__(self, n_opt, parents, root):
        self.n_opt = n_opt
        self.parents = parents
        self.root = root
        self.tree = nx.DiGraph()
        self.weights = {}

    def create(self):
        for n in self.n_opt:
            parent_chain = self.parents[n]
            chain = [n] + parent_chain
            if len(chain) > 1:
                for i in range(1, len(chain)):
                    self.tree.add_edge(chain[i], chain[i - 1])
            self.tree.add_edge(self.root, chain[-1])

        for n in list(self.tree.nodes()):
            if n != self.root:
                self.weights[n] = 0

        for n in self.n_opt:
            while n != self.root:
                self.weights[n] += 1
                n = self.predecessor(n)

    def update_weights(self, winner, children):
        childs = children[winner]
        if not childs:
            return

        bidders = set(childs).intersection(set(self.n_opt))
        if not bidders:
            return

        min_val = float("inf")
        bidder = None
        for b in bidders:
            if b.bid < min_val and self.weights[b] > 0:
                min_val = b.bid
                bidder = b
        if bidder is None:
            return

        bidder_copy = copy.deepcopy(bidder)
        while bidder != winner:
            self.weights[bidder] -= 1
            bidder = self.predecessor(bidder)

        self.update_weights(bidder_copy, children)

    def predecessor(self, node):
        return list(self.tree.predecessors(node))[0]

    def __getitem__(self, bidder):
        return list(self.tree[bidder])

    def get_children(self, node):
        return list(self.tree.successors(node))

    def sum_weights(self, nodes):
        return sum(self.weights[n] for n in nodes)

    def nodes(self):
        return list(self.tree.nodes())


def efficient_allocation(k, bidders, bids):
    bidder_list = [Bidder(bidder, bids[bidder]) for bidder in bidders]
    bidder_list.sort(reverse=True)
    winners = bidder_list[:k]
    social_welfare = sum(w.bid for w in winners)
    return social_welfare, winners


def information_diffusion(graph, bidder, seller):
    level = deque([seller])
    visited = [seller]
    while len(level) > 0:
        n = level.popleft()
        for c in graph[n]:
            if (c not in visited) and (c != bidder):
                level.append(c)
                visited.append(c)
    return visited


def depth_dict(graph, root):
    level = deque([root])
    visited = [root]
    depth = {root: 0}
    while len(level) > 0:
        n = level.popleft()
        for c in graph[n]:
            if c not in visited:
                depth[c] = depth[n] + 1
                level.append(c)
                visited.append(c)
    return depth


def sort_critical_sequence(crit_seq, graph, root):
    crit_seq_new = {}
    depth = depth_dict(graph, root)
    for key, values in crit_seq.items():
        new_values = []
        for val in values:
            new_values.append((depth[val], val))
        new_values.sort(key=lambda tup: tup[0], reverse=True)
        crit_seq_new[key] = [val for _, val in new_values]
    return crit_seq_new


def critical_sequence(graph, seller):
    articulation_points = [n for n in nx.articulation_points(graph) if n != seller]
    nodes = {n for n in graph.nodes() if n != seller}
    crit_seq = {n: [] for n in nodes}
    empty = len(articulation_points) == 0
    for n in articulation_points:
        reachable = set(information_diffusion(graph, n, seller))
        unreachable = nodes - reachable
        unreachable.remove(n)
        for u in unreachable:
            crit_seq[u].append(n)
    if not empty:
        crit_seq = sort_critical_sequence(crit_seq, graph, seller)
    return crit_seq


def critical_children_from(parents):
    children = {key: [] for key in parents.keys()}
    for key, values in parents.items():
        for val in values:
            children[val].append(key)
    return children


def compute_ck(children, parents, bidder, k):
    ck = children[bidder]
    ck.sort(reverse=True)
    if len(ck) >= k:
        ck = ck[:k]
        ck = set(ck)
        parent_children = set()
        child_of_parent_children = set()
        for cki in ck:
            parent_children = parent_children.union(set(parents[cki]).intersection(children[bidder]))
        for pc in parent_children:
            child_of_parent_children = child_of_parent_children.union(set(children[pc]).intersection(children[bidder]))
        ck = ck.union(parent_children).union(child_of_parent_children)
    return set(ck)


def gidm(k, seller_net, reports, bids):
    allocations = {n: False for n in bids.keys()}
    payments = {n: 0.0 for n in bids.keys()}

    graph = nx.Graph()
    seller = Bidder("seller")

    for bidder in seller_net:
        graph.add_edge(seller, Bidder(bidder, bids[bidder]))

    for key in reports.keys():
        bi = Bidder(key, bids[key])
        for value in reports[key]:
            bj = Bidder(value, bids[value])
            graph.add_edge(bi, bj)

    _, n_opt = efficient_allocation(k, list(bids.keys()), bids)
    nodes = {n for n in graph.nodes() if n != seller}
    parents = critical_sequence(graph, seller)
    children = critical_children_from(parents)

    tree = CriticalTree(n_opt, parents, seller)
    tree.create()
    queue = deque(tree[seller])

    while len(queue) > 0:
        i = queue.pop()
        ck = compute_ck(children, parents, i, k)
        di = {i}.union(children[i])
        n_di = nodes - di
        n_ck = nodes - ck
        sw_di, _ = efficient_allocation(k, [b.name for b in n_di], bids)
        sw_ck, alloc_n_ck = efficient_allocation(k, [b.name for b in n_ck], bids)

        if i in n_opt:
            allocations[i.name] = True
            payments[i.name] = sw_di - sw_ck + bids[i.name]
        else:
            if i in alloc_n_ck:
                tree.update_weights(i, children)
                allocations[i.name] = True
                payments[i.name] = sw_di - sw_ck + bids[i.name]
            else:
                allocations[i.name] = False
                payments[i.name] = sw_di - sw_ck

        for n in tree[i]:
            if tree.weights[n] > 0:
                queue.append(n)

    return allocations, payments


def local_k_vickrey(k, seller_net, bids):
    allocations = {n: False for n in bids}
    payments = {n: 0.0 for n in bids}
    seller_bidders = sorted(seller_net, key=lambda x: bids[x], reverse=True)
    winners = seller_bidders[:k]
    threshold = bids[seller_bidders[k]] if len(seller_bidders) > k else 0.0
    for w in winners:
        allocations[w] = True
        payments[w] = threshold
    return allocations, payments


def seller_revenue(payments):
    return sum(payments.values())
