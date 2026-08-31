"""Analytical model of an MoE checkpoint's memory footprint."""
from __future__ import annotations
from dataclasses import dataclass

GB = 1024 ** 3
MB = 1024 ** 2


@dataclass
class MoEModel:
    name: str
    d_model: int
    d_ff_expert: int           # per-expert intermediate width
    n_experts: int             # routed experts per MoE layer
    top_k: int                 # routed experts activated per token
    n_layers: int              # total transformer blocks
    n_moe_layers: int          # blocks that are MoE (rest dense FFN)
    n_shared_experts: int = 0  # always-on experts (never evicted)
    d_ff_dense: int = 0        # intermediate width of dense blocks
    n_heads: int = 0
    n_kv_heads: int = 0
    d_head: int = 0
    vocab_size: int = 0
    dtype_bytes: float = 2.0   # bf16
    mats_per_expert: int = 3   # gate + up + down (SwiGLU)
    nonexpert_params_override: float = 0.0  # use when the naive attention
    # formula does not apply (e.g. DeepSeek MLA's low-rank KV compression);
    # set to (published total params) - (routed + shared expert params).
    note: str = ""

    # ---------- per expert ----------
    @property
    def expert_params(self):
        return self.mats_per_expert * self.d_model * self.d_ff_expert

    @property
    def expert_bytes(self):
        return self.expert_params * self.dtype_bytes

    # ---------- expert pool ----------
    @property
    def n_expert_instances(self):
        return self.n_experts * self.n_moe_layers

    @property
    def routed_expert_bytes(self):
        return self.n_expert_instances * self.expert_bytes

    @property
    def shared_expert_bytes(self):
        return self.n_shared_experts * self.n_moe_layers * self.expert_bytes

    # ---------- non-expert, pinned in HBM ----------
    @property
    def attention_params(self):
        if not (self.n_heads and self.d_head):
            return 0
        kv = self.n_kv_heads or self.n_heads
        qo = 2 * self.d_model * self.n_heads * self.d_head
        kvp = 2 * self.d_model * kv * self.d_head
        return self.n_layers * (qo + kvp)

    @property
    def dense_ffn_params(self):
        n_dense = self.n_layers - self.n_moe_layers
        return n_dense * self.mats_per_expert * self.d_model * self.d_ff_dense

    @property
    def embedding_params(self):
        return 2 * self.vocab_size * self.d_model

    @property
    def nonexpert_params(self):
        if self.nonexpert_params_override:
            return self.nonexpert_params_override
        return self.attention_params + self.dense_ffn_params + self.embedding_params

    @property
    def resident_bytes(self):
        return self.nonexpert_params * self.dtype_bytes + self.shared_expert_bytes

    @property
    def total_bytes(self):
        return self.resident_bytes + self.routed_expert_bytes

    @property
    def total_params(self):
        return int(self.total_bytes / self.dtype_bytes)

    @property
    def active_params_per_token(self):
        moe = self.n_moe_layers * (self.top_k + self.n_shared_experts) * self.expert_params
        # embeddings are a lookup, not a GEMM, so they are excluded from
        # "active params" in the conventional sense
        nx = self.nonexpert_params
        if not self.nonexpert_params_override:
            nx = self.attention_params + self.dense_ffn_params
        return moe + nx

    @property
    def expert_fraction(self):
        return self.routed_expert_bytes / self.total_bytes

    # ---------- compute ----------
    def flops_per_token(self):
        return 2 * self.active_params_per_token

    def kv_bytes_per_token(self):
        kv = self.n_kv_heads or self.n_heads
        return 2 * self.n_layers * kv * self.d_head * self.dtype_bytes

    @classmethod
    def from_dict(cls, d):
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown keys in model config: {sorted(unknown)}")
        return cls(**d)

    def summary(self):
        return {
            "model": self.name,
            "total_params_B": self.total_params / 1e9,
            "active_params_B": self.active_params_per_token / 1e9,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "expert_MB": self.expert_bytes / MB,
            "expert_instances": self.n_expert_instances,
            "routed_expert_GB": self.routed_expert_bytes / GB,
            "resident_GB": self.resident_bytes / GB,
            "total_GB": self.total_bytes / GB,
            "expert_frac": self.expert_fraction,
        }
