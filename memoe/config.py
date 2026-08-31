"""YAML loading.

NOTE: utf-8-sig is mandatory. PowerShell 5.1's `Set-Content -Encoding UTF8`
writes a BOM, which turns the first YAML key into "\ufeffname" and silently
breaks parsing. utf-8-sig strips it and is a no-op on clean UTF-8 files.
"""
from __future__ import annotations
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"


def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(
            f"{p} did not parse to a mapping (got {type(data).__name__}); "
            "check the file is non-empty and has no BOM")
    return data


def _resolve(kind, name):
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    return CONFIG_DIR / kind / f"{Path(name).stem}.yaml"


def load_model(name):
    from .moe import MoEModel
    return MoEModel.from_dict(load_yaml(_resolve("models", name)))


def load_memory(name):
    from .memory import MemorySystem
    return MemorySystem.from_dict(load_yaml(_resolve("memory", name)))


def load_gpu(name):
    from .memory import GPU
    return GPU.from_dict(load_yaml(_resolve("gpu", name)))
