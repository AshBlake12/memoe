"""MEMoE: MoE-Oriented Memory Expansion.

Analysis + simulation framework for CXL-attached memory expansion under
Mixture-of-Experts inference.
"""
__version__ = "1.0.0"

from .config import load_yaml, load_model, load_memory, load_gpu
from .moe import MoEModel
from .memory import Tier, MemorySystem, GPU
from .trace import RoutingTrace, synth_trace
from .analysis import capacity_table, batch_regime, hitrate_requirement, prefetch_depth
from .skew import skew_stats, coverage_curve
from .sim import Simulator, SimResult
from .policy import POLICIES

__all__ = [
    "load_yaml", "load_model", "load_memory", "load_gpu", "MoEModel",
    "Tier", "MemorySystem", "GPU", "RoutingTrace", "synth_trace",
    "capacity_table", "batch_regime", "hitrate_requirement", "prefetch_depth",
    "skew_stats", "coverage_curve", "Simulator", "SimResult", "POLICIES",
]
