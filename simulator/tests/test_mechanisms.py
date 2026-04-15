from refauc.generators import line_instance, star_instance
from refauc.mechanisms import central_vickrey, local_vickrey, network_vcg, information_diffusion_mechanism


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
