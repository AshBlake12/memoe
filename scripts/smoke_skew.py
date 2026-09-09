# smoke_skew.py — 5k tokens, ~10 min including load
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from memoe.hooks import capture
from memoe.skew import skew_stats, coverage_curve

mid = "allenai/OLMoE-1B-7B-0924-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid, torch_dtype=torch.bfloat16, device_map="cuda").eval()

from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
texts = [t for t in ds["text"] if len(t) > 500][:40]

with capture(model, n_experts=64, top_k=8) as cap:
    with torch.no_grad():
        for t in texts:
            model(**tok(t, return_tensors="pt", truncation=True,
                        max_length=512).to("cuda"))

tr = cap.trace("OLMoE-1B-7B")
tr.save(str(ROOT / "results" / "olmoe_smoke.npz"))
c = tr.counts()
print(tr.ids.shape)
print({k: round(v, 3) for k, v in skew_stats(c)["aggregate"].items()})
f, cov = coverage_curve(c)
for x in (0.10, 0.25, 0.50):
    print(f"top {x:.0%} of experts -> {cov[int(x*100)]:.1%} of routings")
