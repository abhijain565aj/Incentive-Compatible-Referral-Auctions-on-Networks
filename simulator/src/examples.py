from __future__ import annotations

from typing import Dict, List, Tuple

Node = str


def user_gidm_truthful() -> Tuple[Node, Dict[Node, List[Node]], Dict[Node, float], int, str]:
    seller = 'S'
    reports = {
        'S': ['A', 'B', 'Z'],
        'A': ['D'],
        'B': ['D', 'U', 'V'],
        'Z': ['C', 'U', 'V'],
        'C': ['E', 'F', 'G'],
        'D': ['H', 'I', 'J'],
        'H': ['I'],
        'J': ['I'],
        'E': ['K'],
        'F': ['K', 'L'],
        'I': ['M'],
        'K': ['Y'],
        'M': ['O'],
        'Y': ['P', 'Q'],
        'P': ['Q'],
        'U': [], 'V': [], 'G': [], 'L': [], 'O': [], 'Q': []
    }
    bids = {
        'A': 7, 'B': 4, 'Z': 1,
        'U': 11, 'V': 12, 'C': 2,
        'D': 14, 'E': 8, 'F': 9, 'G': 17,
        'H': 19, 'I': 6, 'J': 5, 'K': 18, 'L': 10,
        'M': 15, 'Y': 20, 'O': 3, 'P': 11, 'Q': 1,
        'S': 0,
    }
    return seller, reports, bids, 5, 'C'


def user_gidm_attacked() -> Tuple[Node, Dict[Node, List[Node]], Dict[Node, float], int, str]:
    seller, reports, bids, k, attacker = user_gidm_truthful()
    reports = {u: list(vs) for u, vs in reports.items()}
    bids = dict(bids)
    reports['C'].append("C'")
    reports["C'"] = []
    bids["C'"] = 18.9
    return seller, reports, bids, k, attacker
