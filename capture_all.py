import sys, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, ".")
from memoe.hooks import capture
from memoe.skew import skew_stats, coverage_curve

mid = "allenai/OLMoE-1B-7B-0924-Instruct"
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(
    mid, dtype=torch.bfloat16, device_map="cuda").eval()

def get_texts(domain):
    from datasets import load_dataset
    if domain == "prose":
        d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        return [t for t in d["text"] if len(t) > 800][:150]
    if domain == "math":
        d = load_dataset("openai/gsm8k", "main", split="train")
        return [q + "\n" + a for q, a in zip(d["question"][:400], d["answer"][:400])]
    if domain == "chat":
        d = load_dataset("tatsu-lab/alpaca", split="train")
        return [i + "\n" + o for i, o in
                zip(d["instruction"][:400], d["output"][:400]) if len(o) > 200]
    if domain == "code":
        out = []
        for f in glob.glob("/usr/lib/python3*/**/*.py", recursive=True)[:300]:
            try:
                s = open(f, encoding="utf-8", errors="ignore").read()
                if len(s) > 800:
                    out.append(s[:4000])
            except Exception:
                pass
        return out
    raise ValueError(domain)

for domain in ["prose", "math", "code", "chat"]:
    texts = get_texts(domain)
    with capture(model, n_experts=64, top_k=8) as cap:
        with torch.no_grad():
            for t in texts:
                model(**tok(t, return_tensors="pt", truncation=True,
                            max_length=512).to("cuda"))
    tr = cap.trace(f"OLMoE-1B-7B-{domain}")
    tr.save(f"results/traces/olmoe_{domain}.npz")
    c = tr.counts(); st = skew_stats(c)["aggregate"]
    f, cov = coverage_curve(c)
    print(f"\n=== {domain}: {tr.ids.shape[0]} tokens")
    print(f"  zipf_s={st['zipf_s']:.3f} norm_entropy={st['norm_entropy']:.3f} "
          f"gini={st['gini']:.3f} max_share={st['max_share']:.4f}")
    print(f"  top10={cov[10]:.3f} top25={cov[25]:.3f} top50={cov[50]:.3f}")
