"""MEMoE test suite. Run: python -m pytest -q"""
import sys, math
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from memoe import *
from memoe.moe import MoEModel
from memoe.memory import Tier, MemorySystem, GPU
from memoe.skew import fit_zipf_s, _gini

GB = 1024 ** 3
MB = 1024 ** 2

# ---------------------------------------------------------------- configs
def test_all_model_configs_load():
    for n in ["olmoe", "mixtral_8x7b", "qwen3_235b", "deepseek_v3"]:
        m = load_model(n)
        assert m.n_experts > 0 and m.top_k <= m.n_experts
        assert m.n_moe_layers <= m.n_layers


def test_bom_prefixed_yaml_still_parses(tmp_path):
    """Regression: PowerShell 5.1 -Encoding UTF8 writes a BOM."""
    p = tmp_path / "bom.yaml"
    p.write_bytes(b"\xef\xbb\xbfname: bom_test\nd_model: 8\n")
    d = load_yaml(p)
    assert "name" in d and d["name"] == "bom_test"
    assert not any(k.startswith("\ufeff") for k in d)


def test_empty_yaml_raises_clearly(tmp_path):
    p = tmp_path / "empty.yaml"; p.write_text("")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


def test_unknown_model_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        MoEModel.from_dict({"name": "x", "d_model": 8, "d_ff_expert": 8,
                            "n_experts": 2, "top_k": 1, "n_layers": 1,
                            "n_moe_layers": 1, "typo_field": 3})


# ------------------------------------------------------- analytical model
@pytest.mark.parametrize("cfg,expect_B,tol", [
    ("olmoe", 6.9, 0.4), ("mixtral_8x7b", 46.7, 1.0),
    ("qwen3_235b", 235.0, 6.0), ("deepseek_v3", 671.0, 5.0),
])
def test_param_counts_match_published(cfg, expect_B, tol):
    """The analytical model must reproduce published parameter counts."""
    m = load_model(cfg)
    assert abs(m.total_params / 1e9 - expect_B) < tol


def test_expert_bytes_formula():
    m = load_model("olmoe")
    assert m.expert_params == 3 * 2048 * 1024
    assert m.expert_bytes == m.expert_params * 2
    assert math.isclose(m.expert_bytes / MB, 12.0)


def test_expert_pool_dominates_footprint():
    for n in ["olmoe", "mixtral_8x7b", "qwen3_235b", "deepseek_v3"]:
        assert load_model(n).expert_fraction > 0.9


def test_active_far_below_total():
    m = load_model("deepseek_v3")
    assert m.active_params_per_token < 0.10 * m.total_params


# ------------------------------------------------------------- memory
def test_memory_capacity_multiplier():
    mem = load_memory("hbm_cxl_local")
    assert mem.has_cxl
    assert mem.capacity_multiplier > 3.0
    assert load_memory("hbm_only").capacity_multiplier == 1.0


def test_pooled_cxl_is_flagged_as_modeled():
    """Pre-production pooling must never be presented as measured."""
    assert load_memory("hbm_cxl_pooled").is_modeled
    assert not load_memory("hbm_cxl_local").is_modeled


def test_cxl_bandwidth_within_measured_range():
    for n in ["hbm_cxl_local", "hbm_cxl_local_pessimistic"]:
        bw = load_memory(n).cxl.bandwidth_gbs
        assert 18.0 <= bw <= 52.0, "CXL bandwidth must stay in the measured range"


# -------------------------------------------------------------- traces
def test_trace_shape_and_distinct_topk():
    tr = synth_trace(512, 4, 32, 8, zipf_s=1.0, seed=1)
    assert tr.ids.shape == (512, 4, 8)
    for t in range(0, 512, 37):
        for l in range(4):
            assert len(set(tr.ids[t, l].tolist())) == 8


def test_counts_conserve_tokens():
    tr = synth_trace(256, 3, 16, 4, seed=2)
    c = tr.counts()
    assert (c.sum(axis=1) == 256 * 4).all()


def test_distinct_grows_and_saturates_with_batch():
    tr = synth_trace(4096, 2, 64, 8, zipf_s=1.0, seed=3)
    d = [tr.distinct_per_batch(b) for b in (1, 8, 64, 512)]
    assert d[0] == 8.0
    assert d == sorted(d)
    assert d[-1] <= 64


def test_uniform_routing_has_low_skew():
    u = synth_trace(4096, 2, 32, 4, zipf_s=0.0, seed=4)
    z = synth_trace(4096, 2, 32, 4, zipf_s=1.2, seed=4)
    assert skew_stats(u.counts())["aggregate"]["gini"] < \
           skew_stats(z.counts())["aggregate"]["gini"]


def test_trace_roundtrip(tmp_path):
    tr = synth_trace(128, 2, 8, 2, seed=5)
    tr.save(tmp_path / "t.npz")
    tr2 = RoutingTrace.load(tmp_path / "t.npz")
    assert np.array_equal(tr.ids, tr2.ids) and tr2.n_experts == 8


# ---------------------------------------------------------------- skew
def test_fit_zipf_recovers_exponent():
    n = 512
    counts = 1e6 / np.arange(1, n + 1) ** 1.0
    assert abs(fit_zipf_s(counts) - 1.0) < 0.15


def test_gini_bounds():
    assert _gini(np.ones(100)) < 1e-9
    assert _gini(np.array([0.0] * 99 + [100.0])) > 0.95


def test_coverage_curve_monotone_and_bounded():
    tr = synth_trace(2048, 3, 64, 8, zipf_s=1.0, seed=6)
    f, c = coverage_curve(tr.counts())
    assert c[0] == 0 and abs(c[-1] - 1.0) < 1e-6
    assert (np.diff(c) >= -1e-9).all()


def test_skew_raises_hit_rate_of_static_tiering():
    """The core premise: skew is what makes partial residency worth anything."""
    n_layers, n_e = 2, 64
    _, cu = coverage_curve(synth_trace(4096, n_layers, n_e, 8, 0.0, seed=8).counts())
    _, cz = coverage_curve(synth_trace(4096, n_layers, n_e, 8, 1.2, seed=8).counts())
    half = len(cu) // 4          # keep 25% of experts
    assert cz[half] > cu[half] + 0.10


# ----------------------------------------------------------- simulator
def _sim(model="olmoe", mem="hbm_cxl_local", n_gpus=1):
    return Simulator(load_model(model), load_memory(mem), load_gpu("h100"),
                     n_gpus=n_gpus)


def test_all_resident_has_zero_stall():
    tr = synth_trace(2048, 16, 64, 8, seed=9)
    r = _sim().run(tr, batch_size=64, policy="all_resident")
    assert r.stall_s == 0.0 and r.hit_rate == 1.0 and r.overhead == 0.0


def test_hit_rate_bounded():
    tr = synth_trace(2048, 16, 64, 8, seed=10)
    for pol in ("static_popularity", "balanced_static", "lru", "lfu"):
        r = _sim().run(tr, batch_size=64, policy=pol)
        assert 0.0 <= r.hit_rate <= 1.0
        assert 0.0 <= r.expert_hit_rate <= 1.0
        assert r.slowdown >= 1.0 - 1e-9


def test_more_hbm_never_hurts():
    tr = synth_trace(4096, 94, 128, 8, seed=11, zipf_s=1.0)
    m, mem, g = load_model("qwen3_235b"), load_memory("hbm_cxl_local"), load_gpu("h100")
    prev = None
    for n in (1, 2, 4, 6):
        r = Simulator(m, mem, g, n_gpus=n).run(tr, 128, "balanced_static")
        if prev is not None:
            assert r.hit_rate >= prev - 1e-9
        prev = r.hit_rate


def test_bigger_batch_cuts_per_token_traffic():
    # a model whose expert pool does not fit in one GPU
    tr = synth_trace(8192, 94, 128, 8, seed=12, zipf_s=1.0)
    s = _sim(model="qwen3_235b", n_gpus=1)
    small = s.run(tr, 8, "balanced_static").bytes_per_token
    big = s.run(tr, 256, "balanced_static").bytes_per_token
    assert big < small / 4


def test_slower_cxl_is_never_faster():
    tr = synth_trace(4096, 16, 64, 8, seed=13)
    fast = _sim(mem="hbm_cxl_local").run(tr, 64, "balanced_static")
    slow = _sim(mem="hbm_cxl_local_pessimistic").run(tr, 64, "balanced_static")
    assert slow.overhead >= fast.overhead - 1e-12


def test_hbm_only_infeasibility_is_reported():
    m, gpu = load_model("deepseek_v3"), load_gpu("h100")
    s = Simulator(m, load_memory("hbm_only"), gpu, n_gpus=1)
    assert not s.fits()
    r = s.run(synth_trace(512, 58, 256, 8, seed=14), 64, "balanced_static")
    assert any("INFEASIBLE" in n for n in r.notes)


def test_pooled_result_carries_modeled_warning():
    r = _sim(mem="hbm_cxl_pooled").run(synth_trace(1024, 16, 64, 8, seed=15),
                                       64, "balanced_static")
    assert any("MODELED" in n for n in r.notes)


def test_prefetch_reduces_or_matches_overhead():
    tr = synth_trace(4096, 16, 64, 8, seed=16, zipf_s=1.0)
    s = _sim()
    off = s.run(tr, 64, "balanced_static", prefetch_depth=0, prefetch_width=0.0)
    on = s.run(tr, 64, "balanced_static", prefetch_depth=4, prefetch_width=0.25)
    assert on.overhead <= off.overhead + 1e-12


# ------------------------------------------------------------ analyses
def test_batch_regime_is_monotone_decreasing():
    tr = synth_trace(4096, 16, 64, 8, seed=17, zipf_s=1.0)
    df = batch_regime(load_model("olmoe"), tr, (1, 8, 64, 512))
    v = df.bytes_per_token_MB.tolist()
    assert v == sorted(v, reverse=True)
    assert df.amortisation_vs_b1.iloc[-1] > 4


def test_hitrate_requirement_rises_with_tighter_target():
    tr = synth_trace(2048, 16, 64, 8, seed=18)
    df = hitrate_requirement(load_model("olmoe"), load_memory("hbm_cxl_local"),
                             load_gpu("h100"), tr, (64,), (0.05, 0.5))
    a = df[df.target_overhead == 0.05].required_hit_rate.iloc[0]
    b = df[df.target_overhead == 0.50].required_hit_rate.iloc[0]
    assert a >= b


def test_prefetch_depth_grows_as_hit_rate_falls():
    tr = synth_trace(2048, 16, 64, 8, seed=19)
    df = prefetch_depth(load_model("olmoe"), load_memory("hbm_cxl_local"),
                        load_gpu("h100"), tr, (64,), (0.5, 0.95))
    lo = df[df.hit_rate == 0.5].layers_of_lookahead.iloc[0]
    hi = df[df.hit_rate == 0.95].layers_of_lookahead.iloc[0]
    assert lo >= hi


def test_capacity_table_shows_gpu_reduction():
    ms = [load_model(n) for n in ("qwen3_235b", "deepseek_v3")]
    df = capacity_table(ms, load_gpu("h100"), cxl_gb=512.0)
    assert (df.gpu_reduction >= 1.0).all()
    assert (df.capacity_gain > 3.0).all()


# ---------------------------------------------------------------- hooks
def test_router_detection_without_torch_is_graceful():
    from memoe.hooks import find_routers, ROUTER_HINTS
    assert "gate" in ROUTER_HINTS and "router" in ROUTER_HINTS

    class FakeW:
        ndim = 2
        shape = (64, 2048)

    class FakeMod:
        weight = FakeW()

    class FakeModel:
        def named_modules(self):
            return [("layers.0.mlp.gate", FakeMod()),
                    ("layers.0.mlp.experts.0", object()),
                    ("layers.1.mlp.router", FakeMod())]

    found = find_routers(FakeModel(), n_experts=64)
    assert [n for n, _ in found] == ["layers.0.mlp.gate", "layers.1.mlp.router"]
