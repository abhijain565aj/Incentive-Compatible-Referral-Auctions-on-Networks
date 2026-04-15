# Referral Auction Simulator

A modular Python simulator for single-item auctions on social/referral networks.

Implemented mechanisms:

- `central_vickrey`: all buyers known to the seller, second-price benchmark.
- `local_vickrey`: seller sells only to direct neighbours; no diffusion incentive.
- `network_vcg`: VCG extension on the invitation graph; efficient but can create negative revenue.
- `idm`: Information Diffusion Mechanism from Li et al.
- `param_referral`: configurable sandbox mechanism for non-i.i.d. and topology stress tests.

## Quick start

```bash
cd simulator
python -m pip install -r requirements.txt
python run_experiments.py
python plot_results.py
pytest -q
```

## Where to extend

- Add a mechanism in `refauc/mechanisms/` and register it in `refauc/mechanisms/__init__.py`.
- Add topology or valuation models in `refauc/generators.py`.
- Use `param_referral_auction(..., reserve_by_node=..., referral_share=...)` for experiments with non-i.i.d. priors.

## Sign convention

Payments are from buyer to seller. Negative payment means the mechanism rewards that buyer.
Seller revenue is the sum of all payments.
