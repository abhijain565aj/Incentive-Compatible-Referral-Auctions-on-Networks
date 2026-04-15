from __future__ import annotations

import argparse


def dgm_reward(depth: int, total_depth: int, budget: float = 1.0, alpha: float = 0.5) -> float:
    # Normalized geometric reward along a winning path.
    weights = [alpha ** k for k in range(total_depth + 1)]
    return budget * weights[depth] / sum(weights)


def mean_penalized_reward(depth: int, total_depth: int, neighbor_reports: list[float],
                          budget: float = 1.0, alpha: float = 0.5, mu: float = 0.2) -> float:
    base = dgm_reward(depth, total_depth, budget, alpha)
    penalty = mu * (sum(neighbor_reports) / max(1, len(neighbor_reports)))
    return max(0.0, base - penalty)


def sybil_counterexample(mu: float = 0.2) -> dict:
    """Toy demonstration: a mean-neighbor penalty is itself manipulable.

    An agent inserts low-report sybils as apparent neighbors, reducing the local mean penalty and
    increasing reward. This supports the report's decision not to make Variant 3 the main theorem.
    """
    honest_neighbors = [1.0, 1.0, 1.0]
    with_sybils = honest_neighbors + [0.0, 0.0, 0.0, 0.0]
    honest = mean_penalized_reward(1, 4, honest_neighbors, mu=mu)
    attacked = mean_penalized_reward(1, 4, with_sybils, mu=mu)
    return {"honest_reward": honest, "after_sybil_reward": attacked, "gain": attacked - honest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.2)
    args = parser.parse_args()
    print(sybil_counterexample(args.mu))


if __name__ == "__main__":
    main()
