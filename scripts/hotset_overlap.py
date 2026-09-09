import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import itertools, numpy as np
from memoe import RoutingTrace

doms = ["prose", "math", "code", "chat"]
tr = {d: RoutingTrace.load(str(ROOT / "results" / "traces" / f"olmoe_{d}.npz")) for d in doms}
cnt = {d: tr[d].counts() for d in doms}
L, E = cnt["prose"].shape

for frac in (0.25, 0.50):
    k = int(frac * E)
    print(f"\nhot-set overlap, top {frac:.0%} of experts per layer")
    for a, b in itertools.combinations(doms, 2):
        ov = np.mean([len(set(np.argsort(cnt[a][l])[::-1][:k]) &
                          set(np.argsort(cnt[b][l])[::-1][:k])) / k
                      for l in range(L)])
        print(f"  {a:6s} vs {b:6s}: {ov:.1%}   (chance = {frac:.0%})")
