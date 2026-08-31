"""Memory tiers and GPU compute spec."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

GB = 1024 ** 3


@dataclass
class Tier:
    name: str
    capacity_gb: float
    bandwidth_gbs: float    # sustained MEASURED bandwidth, not link peak
    latency_ns: float
    measured: bool = True   # False => modeled/projected (e.g. CXL 3.0 pooling)
    note: str = ""

    @property
    def capacity_bytes(self):
        return self.capacity_gb * GB

    @property
    def bytes_per_s(self):
        return self.bandwidth_gbs * 1e9


@dataclass
class MemorySystem:
    name: str
    hbm: Tier
    cxl: Optional[Tier] = None
    note: str = ""

    @property
    def has_cxl(self):
        return self.cxl is not None

    @property
    def total_capacity_gb(self):
        return self.hbm.capacity_gb + (self.cxl.capacity_gb if self.cxl else 0.0)

    @property
    def capacity_multiplier(self):
        return self.total_capacity_gb / self.hbm.capacity_gb

    @property
    def is_modeled(self):
        """True if any tier is projected rather than measured on silicon."""
        return bool(self.cxl and not self.cxl.measured)

    @classmethod
    def from_dict(cls, d):
        hbm = Tier(**d["hbm"])
        cxl = Tier(**d["cxl"]) if d.get("cxl") else None
        return cls(name=d["name"], hbm=hbm, cxl=cxl, note=d.get("note", ""))


@dataclass
class GPU:
    name: str
    peak_flops: float     # dense bf16 FLOP/s
    mfu: float = 0.40     # achieved fraction of peak
    hbm_gb: float = 80.0
    note: str = ""

    @property
    def eff_flops(self):
        return self.peak_flops * self.mfu

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
