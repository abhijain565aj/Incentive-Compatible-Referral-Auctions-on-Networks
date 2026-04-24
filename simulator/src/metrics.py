from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Set

Node = str


def real_utility(
    winners: Iterable[Node],
    payments: Mapping[Node, float],
    owner: Mapping[Node, Node],
    true_values: Mapping[Node, float],
) -> Dict[Node, float]:
    real_agents = set(owner.values())
    win_owner: Set[Node] = {owner[w] for w in winners}
    out: Dict[Node, float] = {}
    for a in real_agents:
        val = true_values.get(a, 0.0) if a in win_owner else 0.0
        pay = sum(payments.get(i, 0.0) for i, o in owner.items() if o == a)
        out[a] = val - pay
    return out


def real_welfare(winners: Iterable[Node], owner: Mapping[Node, Node], true_values: Mapping[Node, float]) -> float:
    seen = {owner[w] for w in winners}
    return float(sum(true_values[a] for a in seen))


def fake_winners(winners: Iterable[Node], owner: Mapping[Node, Node]) -> int:
    return sum(1 for w in winners if owner[w] != w)


def reward_mass_to_fake_nodes(payments: Mapping[Node, float], owner: Mapping[Node, Node]) -> float:
    return float(sum(-p for i, p in payments.items() if p < 0 and owner[i] != i))


def reward_mass_to_real_nodes(payments: Mapping[Node, float], owner: Mapping[Node, Node]) -> float:
    return float(sum(-p for i, p in payments.items() if p < 0 and owner[i] == i))


def attacker_reward_mass(payments: Mapping[Node, float], owner: Mapping[Node, Node], attacker: Node) -> float:
    return float(sum(-p for i, p in payments.items() if p < 0 and owner[i] == attacker))


def seller_revenue(payments: Mapping[Node, float]) -> float:
    return float(sum(payments.values()))
