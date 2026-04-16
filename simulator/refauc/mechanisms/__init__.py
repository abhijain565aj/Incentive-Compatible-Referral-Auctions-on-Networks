from .benchmarks import central_vickrey, local_vickrey
from .network_vcg import network_vcg
from .idm import information_diffusion_mechanism
from .param_referral import parametric_referral_auction
from .sybil_resistant_referral import sybil_resistant_referral_auction

ALL_MECHANISMS = {
    "central_vickrey": central_vickrey,
    "local_vickrey": local_vickrey,
    "network_vcg": network_vcg,
    "idm": information_diffusion_mechanism,
    "param_referral": parametric_referral_auction,
    "sybil_resistant_referral": sybil_resistant_referral_auction,
}
