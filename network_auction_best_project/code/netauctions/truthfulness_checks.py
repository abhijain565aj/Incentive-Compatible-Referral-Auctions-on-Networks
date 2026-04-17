from __future__ import annotations

import argparse
from .generators import path_tree
from .mechanisms import participation_cost_idm, idm


def check_pc_idm(max_value: int = 40) -> None:
    tree = path_tree(4)
    base_values = {1: 10.0, 2: 20.0, 3: 30.0, 4: 25.0}
    costs = {1: 2.0, 2: 3.0, 3: 4.0, 4: 1.0}
    truthful = participation_cost_idm(tree, base_values, costs)
    for i in tree.buyers:
        truth_u = truthful.utility(base_values, costs, i)
        for bid in range(max_value + 1):
            reported_values = dict(base_values)
            # PC-IDM asks for value report and subtracts public cost.
            reported_values[i] = float(bid)
            outcome = participation_cost_idm(tree, reported_values, costs)
            # Utility must be evaluated using true value, not reported value.
            p = (outcome.payments or {}).get(i, 0.0)
            alloc = 1.0 if outcome.winner == i else 0.0
            dev_u = alloc * base_values[i] - p - costs[i]
            if dev_u > truth_u + 1e-8:
                raise AssertionError((i, bid, truth_u, dev_u, truthful, outcome))
    print("PC-IDM passed exhaustive value-report checks on the toy path instance.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-value", type=int, default=40)
    args = parser.parse_args()
    check_pc_idm(args.max_value)


if __name__ == "__main__":
    main()
