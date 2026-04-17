from refauc.generators import line_instance, star_instance
from refauc.mechanisms import central_vickrey, local_vickrey, network_vcg, information_diffusion_mechanism
from refauc.mechanisms.sybil_resistant_referral import sybil_resistant_referral_auction
from refauc.experiments import summarize


def test_line_vcg_can_have_negative_revenue():
    inst = line_instance(5, seed=1)
    for i in inst.buyers():
        inst.valuations[i] = 0.0
    inst.valuations[5] = 1.0
    res = network_vcg(inst)
    assert res.winner == 5
    assert res.revenue < 0


def test_idm_nonnegative_revenue_on_line():
    inst = line_instance(5, seed=1)
    for i in inst.buyers():
        inst.valuations[i] = 0.0
    inst.valuations[5] = 1.0
    res = information_diffusion_mechanism(inst)
    assert res.revenue >= -1e-9


def test_star_local_equals_central_participants():
    inst = star_instance(6, seed=2)
    a = central_vickrey(inst)
    b = local_vickrey(inst)
    assert a.winner == b.winner
    assert abs(a.revenue - b.revenue) < 1e-9


def test_sybil_resistant_referral_nonnegative_revenue_on_line():
    inst = line_instance(6, seed=10)
    for i in inst.buyers():
        inst.valuations[i] = 0.0
    inst.valuations[6] = 1.0
    res = sybil_resistant_referral_auction(inst)
    assert res.winner is not None
    assert res.revenue >= -1e-9


def test_summary_contains_budget_balance_rate():
    rows = [
        {
            "mechanism": "m",
            "topology": "line",
            "n": 10,
            "valuation_mode": "uniform",
            "diffusion_strategy": "full",
            "revenue": 1.0,
            "welfare": 1.2,
            "welfare_sum_utilities": 0.8,
            "welfare_product_utilities": 0.08,
            "welfare_log_product_utilities": -2.5,
            "n_participants": 10,
        },
        {
            "mechanism": "m",
            "topology": "line",
            "n": 10,
            "valuation_mode": "uniform",
            "diffusion_strategy": "full",
            "revenue": -0.1,
            "welfare": 0.9,
            "welfare_sum_utilities": 0.4,
            "welfare_product_utilities": 0.02,
            "welfare_log_product_utilities": -3.2,
            "n_participants": 10,
        },
    ]
    out = summarize(rows)
    assert len(out) == 1
    assert abs(out[0]["budget_balance_rate"] - 0.5) < 1e-9
    assert "welfare_sum_utilities_mean" in out[0]
    assert "welfare_product_utilities_mean" in out[0]
    assert "welfare_log_product_utilities_mean" in out[0]


def test_row_contains_utility_welfare_metrics():
    inst = star_instance(6, seed=2)
    res = information_diffusion_mechanism(inst)
    row = res.as_row(inst)
    assert "welfare_sum_utilities" in row
    assert "welfare_product_utilities" in row
    assert "welfare_log_product_utilities" in row
