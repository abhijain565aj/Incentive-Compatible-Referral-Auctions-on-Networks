
"""
SC-GIDM: Sybil-cluster reduction followed by GIDM.

Key modelling choice:
- each cluster has a designated certified non-Sybil root;
- the cluster bid is the root bid (not the maximum Sybil bid);
- external edges are obtained by contracting identities inside a cluster.

This is the mathematically clean mechanism analysed in the report.
"""

from collections import defaultdict
from .gidm_adapted import gidm, seller_revenue


def contract_to_clusters(seller_net, reports, bids, owner, root_of_owner=None):
    reps = sorted(set(owner.values()))
    if root_of_owner is None:
        root_of_owner = {rep: rep for rep in reps}

    cluster_bids = {rep: bids[root_of_owner[rep]] for rep in reps}

    seller_clusters = []
    seen = set()
    for n in seller_net:
        rep = owner[n]
        if rep not in seen:
            seller_clusters.append(rep)
            seen.add(rep)

    cluster_reports = defaultdict(set)
    for u, vs in reports.items():
        ru = owner[u]
        for v in vs:
            rv = owner[v]
            if ru != rv:
                cluster_reports[ru].add(rv)

    cluster_reports = {u: sorted(vs) for u, vs in cluster_reports.items() if vs}
    return seller_clusters, cluster_reports, cluster_bids, root_of_owner


def sc_gidm(k, seller_net, reports, bids, owner, root_of_owner=None):
    cs, cr, cb, root_of_owner = contract_to_clusters(seller_net, reports, bids, owner, root_of_owner)
    alloc_c, pay_c = gidm(k, cs, cr, cb)

    allocations = {n: False for n in bids}
    payments = {n: 0.0 for n in bids}

    for rep, allocated in alloc_c.items():
        root = root_of_owner[rep]
        if allocated:
            allocations[root] = True
        payments[root] = pay_c[rep]

    return allocations, payments


def sc_revenue(k, seller_net, reports, bids, owner, root_of_owner=None):
    _, payments = sc_gidm(k, seller_net, reports, bids, owner, root_of_owner)
    return seller_revenue(payments)
