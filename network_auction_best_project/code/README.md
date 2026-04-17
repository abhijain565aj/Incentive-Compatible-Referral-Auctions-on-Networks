# Network Referral Auction Simulator

This simulator accompanies the final report **Incentive-Compatible Referral Auctions on Networks: Cost, Sybil, Multi-Item and Multi-Seller Extensions**.

It is intentionally lightweight and uses only the Python standard library.

## Run everything

```bash
cd code
./run_experiments.sh
```

This runs:

1. an exhaustive truthfulness sanity check for the proposed participation-cost IDM on a toy path;
2. Monte-Carlo simulations for baseline IDM, Network-VCG, Local Vickrey, PC-IDM, and Tax-IDM;
3. sybil stress tests on a path;
4. multi-unit and public-quality heterogeneous-item benchmarks;
5. a fixed-partition multi-seller benchmark.

Outputs are written to `../results/`:

- `runs.csv`: every simulated run;
- `summary.csv`: grouped means and standard errors;
- `*.svg`: lightweight plot artifacts.

The report figures under `../report/figures/` are small vector PDFs generated from `summary.csv` by:

```bash
python3 -m netauctions.pdfplots
```

## Code structure

- `netauctions/network.py`: rooted-tree representation and sybil insertion.
- `netauctions/generators.py`: random/path/binary trees, non-i.i.d. values, participation costs.
- `netauctions/mechanisms.py`: Local Vickrey, Network VCG, IDM, PC-IDM, Tax-IDM, multi-unit VCG, heterogeneous-item VCG, and fixed-partition multi-seller PC-IDM.
- `netauctions/experiments.py`: experiment runner and CSV/SVG output.
- `netauctions/truthfulness_checks.py`: brute-force DSIC sanity check on a small instance.
- `netauctions/pdfplots.py`: pure-Python PDF plot generator for the LaTeX report.

## Research caveat

The theorem-level contribution is PC-IDM under public participation costs on rooted-tree diffusion. The multi-item, divisible, sybil-tax, and multi-seller components are experimental stress-test baselines or restricted-domain mechanisms; the report states these boundaries explicitly.
