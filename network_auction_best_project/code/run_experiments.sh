#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
/usr/bin/python3 -m netauctions.truthfulness_checks --max-value 40
/usr/bin/python3 -m netauctions.experiments --out ../results --runs 20 --n 20 40 60 --max-sybils 8
