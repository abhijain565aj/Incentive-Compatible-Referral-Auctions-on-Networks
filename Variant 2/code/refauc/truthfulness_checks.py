from __future__ import annotations

import argparse
from typing import Dict, Tuple


def ea_sp_outcome(reports: Dict[int, float], E: Dict[int, float], lam: float) -> Tuple[int | None, float]:
    score = {i: reports[i] - lam * E[i] for i in reports}
    w = max(score, key=lambda i: score[i])
    if score[w] < 0:
        return None, 0.0
    second = max([0.0] + [score[j] for j in reports if j != w])
    payment = second + lam * E[w]
    return w, payment


def utility(true_value: float, agent: int, winner: int | None, payment: float) -> float:
    return true_value - payment if winner == agent else 0.0


def exhaustive_check(max_value: int = 20, lam: float = 1.0) -> bool:
    """Ex-post DSIC check for the scalar transformed EA second price auction.

    We enumerate two branches, true values and reports on an integer grid. The check is not a proof,
    but catches implementation errors and illustrates the threshold-property theorem in the report.
    """
    E = {1: 3.0, 2: 7.0}
    for v1 in range(max_value + 1):
        for v2 in range(max_value + 1):
            truthful_reports = {1: float(v1), 2: float(v2)}
            w_truth, p_truth = ea_sp_outcome(truthful_reports, E, lam)
            u_truth = utility(float(v1), 1, w_truth, p_truth)
            for r1 in range(max_value + 1):
                w_dev, p_dev = ea_sp_outcome({1: float(r1), 2: float(v2)}, E, lam)
                u_dev = utility(float(v1), 1, w_dev, p_dev)
                if u_dev > u_truth + 1e-9:
                    print("Counterexample", {"v1": v1, "v2": v2, "r1": r1,
                                             "u_truth": u_truth, "u_dev": u_dev})
                    return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-value", type=int, default=30)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.0)
    args = parser.parse_args()
    ok = exhaustive_check(args.max_value, args.lam)
    print("EA-SP scalar truthfulness check:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
