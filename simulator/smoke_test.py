from src.examples import user_gidm_truthful, user_gidm_attacked
from src.attacks import truthful_owner
from src.gidm_tree import gidm_from_graph
from src.sc_gidm import sc_gidm
import random

seller, reports, bids, K, _ = user_gidm_truthful()
out = gidm_from_graph(seller, reports, bids, K)
assert len(out.winners) <= K

seller, reports, bids, K, _ = user_gidm_attacked()
out2 = gidm_from_graph(seller, reports, bids, K)
out3 = sc_gidm(seller, reports, bids, K, random.Random(3))
assert len(out2.winners) <= K
assert len(out3.winners) <= K
print('smoke test passed')
