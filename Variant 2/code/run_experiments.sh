#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m refauc.truthfulness_checks --max-value 30
python -m refauc.mean_externality_routing
python -m refauc.experiments --out ../results --n 30 50 --eta 0 0.5 1 2 3 --runs 100 --sigma 10
