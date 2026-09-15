#!/usr/bin/env python3
"""
MEMoE interactive dashboard.

Writes a single self-contained HTML file. No server, no callbacks, no Dash.
Open it in a browser or drop it on a projector.

    uv run python scripts/build_dashboard.py
    -> results/dashboard.html

Every number below is a measurement or a value derived from one. The source of
each block is named in the comment above it so nothing here can drift from the
report without someone noticing.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

# --------------------------------------------------------------------------
# measured data
# --------------------------------------------------------------------------

# DRAMSim3, DDR4 x8 configs, 12 MB sequential expert read (196,608 x 64 B).
# bandwidth from average_interarrival, not the diluted headline figure.
DRAM_SWEEP = [
    # grade, theoretical GB/s, measured GB/s, efficiency %, loaded latency ns
    ("DDR4-1866", 14.9, 12.76, 85.5, 237.9),
    ("DDR4-2133", 17.1, 13.24, 77.6, 228.9),
    ("DDR4-2400", 19.2, 14.96, 77.9, 202.9),
    ("DDR4-2666", 21.3, 15.29, 71.7, 197.3),
    ("DDR4-2933", 23.5, 15.65, 66.7, 192.9),
    ("DDR4-3200", 25.6, 16.88, 65.9, 179.4),
]

CXL_1CH = 16.88   # measured
CXL_4CH = 67.50   # 4 x measured single channel, scaled not measured
HBM3E_BW = 3350.0  # specification, never enters the stall model

# OLMoE-1B-7B forward hooks, four workloads, 258,629 tokens total.
ROUTING = [
    # domain, tokens, zipf s, normalised entropy, gini, top-25% share
    ("chat", 18239, 0.494, 0.972, 0.261, 40.7),
    ("prose", 37355, 0.598, 0.963, 0.300, 43.5),
    ("math", 63257, 0.899, 0.913, 0.442, 56.0),
    ("code", 139778, 1.132, 0.877, 0.522, 63.6),
]

# fraction of experts touched per layer, per batch. Real traces.
COVERAGE_BATCH = [16, 64, 256, 1024, 4096]
COVERAGE = {
    "chat": [0.648, 0.882, 0.986, 1.000, 1.000],
    "code": [0.535, 0.762, 0.903, 0.973, 0.995],
    "math": [0.624, 0.874, 0.973, 0.994, 1.000],
    "prose": [0.669, 0.895, 0.974, 0.997, 1.000],
}

# top-25% hot-set overlap per layer. Diagonal is the split-half self-overlap
# control. 25% is chance.
DOMAINS = ["chat", "code", "math", "prose"]
OVERLAP = {
    ("chat", "chat"): 79.7, ("code", "code"): 89.8,
    ("math", "math"): 94.9, ("prose", "prose"): 75.0,
    ("math", "code"): 50.0, ("prose", "math"): 35.9,
    ("prose", "chat"): 34.8, ("math", "chat"): 30.1,
    ("code", "chat"): 18.4, ("prose", "code"): 15.2,
}

# fetch hit rate when a placement profiled on one workload is deployed on
# another, 50% residency. Random placement scores 0.500.
PENALTY_ROWS = ["chat", "code", "math", "prose"]      # profiled on
PENALTY_COLS = ["chat", "code", "math", "prose"]      # deployed on
PENALTY = [
    [0.686, 0.412, 0.496, 0.538],
    [0.470, 0.840, 0.618, 0.448],
    [0.558, 0.624, 0.790, 0.552],
    [0.551, 0.383, 0.584, 0.709],
]

# simulator sweep across synthetic skew, 90% of experts resident.
SKEW_SWEEP_BATCH = [64, 512, 8192]
SKEW_SWEEP = {
    "s = 0.0": [0.899, 0.899, 0.899],
    "s = 0.6": [0.900, 0.899, 0.899],
    "s = 1.0": [0.907, 0.899, 0.899],
    "s = 1.5": [0.927, 0.899, 0.899],
}

# footprint model, validated against published parameter counts.
MODELS = {
    "OLMoE-1B-7B":     dict(experts=64,  k=8, weights=12.9,   pool=12.0,   gpus_hbm=1,  gpus_cxl=1, kv_intensity=1.0),
    "Mixtral-8x7B":    dict(experts=8,   k=2, weights=87.0,   pool=84.0,   gpus_hbm=2,  gpus_cxl=1, kv_intensity=4.0),
    "Qwen3-235B-A22B": dict(experts=128, k=8, weights=437.8,  pool=423.0,  gpus_hbm=6,  gpus_cxl=1, kv_intensity=16.0),
    "DeepSeek-V3":     dict(experts=256, k=9, weights=1249.2, pool=1218.0, gpus_hbm=16, gpus_cxl=3, kv_intensity=1.0),
}

# per-GPU effective compute used by the stall model: 40% MFU on H100 BF16.
FLOPS = 3.956e14

# share of the expert pool that fits on CXL inside a 10% stall budget,
# from the real traces against the measured tier.
OFFLOADABLE = [
    # batch, 16.88 GB/s, 67.5 GB/s
    ("4,096", "1.9 - 2.5%", "9.3 - 10.3%"),
    ("16,384", "9.3 - 9.8%", "32.5 - 35.6%"),
    ("65,536 (code)", "35.4%", "82.2%"),
    ("131,072 (code)", "59.5%", "94.0%"),
]

# qwen3-235B, 32k context, 5.875 GB of KV read per decode step.
KV_STEPS = [("HBM3e", 1.883, 531), ("CXL 4 channel", 93.455, 11), ("CXL 1 channel", 373.710, 3)]

# qwen3-235B against a 67.5 GB/s pool.
POOLING = [
    # nodes, per-node BW, dedicated GB, pooled GB, B* at h=0.9
    (1, 67.50, 423.0, 423.0, 93771),
    (2, 33.75, 846.0, 423.0, 187543),
    (4, 16.88, 1692.0, 423.0, 375087),
    (8, 8.44, 3384.0, 423.0, 750174),
    (16, 4.22, 6768.0, 423.0, 1500349),
]

# OLMoE at 67.5 GB/s, 300 ns access latency.
GRANULARITY = [
    ("Whole expert, 12 MB", 1, 0.16),
    ("4 MB", 3, 0.48),
    ("512 KB", 24, 3.72),
    ("64 KB", 192, 23.61),
    ("4 KB", 3072, 83.18),
]

# --------------------------------------------------------------------------
# design tokens
# --------------------------------------------------------------------------

C = dict(
    paper="#FBFBF9",
    surface="#FFFFFF",
    ink="#14171C",
    body="#333B47",
    muted="#5A6472",
    rule="#DCE0E6",
    rule_soft="#EDEFF2",
    hbm="#1B4FA8",
    cxl="#9A4C10",
    neutral="#6B7280",
    bad="#A61B2B",
    good="#16653C",
)

DOMAIN_COLOR = {
    "chat": "#1B4FA8",
    "prose": "#2E7D8F",
    "math": "#7A4EA8",
    "code": "#9A4C10",
}

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

LAYOUT = dict(
    font=dict(family=FONT, size=14, color=C["body"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=64, r=24, t=16, b=56),
    hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="#14171C", bordercolor="#14171C"),
    xaxis=dict(showgrid=False, linecolor=C["rule"], ticks="outside", tickcolor=C["rule"],
               ticklen=5, title_font=dict(size=13, color=C["muted"]), zeroline=False),
    yaxis=dict(gridcolor=C["rule_soft"], linecolor="rgba(0,0,0,0)", zeroline=False,
               title_font=dict(size=13, color=C["muted"])),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(size=13),
                bgcolor="rgba(0,0,0,0)"),
)


def fig_shell(height: int, **over) -> dict:
    lay = json.loads(json.dumps(LAYOUT))
    lay["height"] = height
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(lay.get(k), dict):
            lay[k].update(v)
        else:
            lay[k] = v
    return lay


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_dram() -> go.Figure:
    grades = [r[0] for r in DRAM_SWEEP]
    peak = [r[1] for r in DRAM_SWEEP]
    meas = [r[2] for r in DRAM_SWEEP]
    eff = [r[3] for r in DRAM_SWEEP]
    f = go.Figure()
    f.add_bar(x=grades, y=peak, name="DIMM specification",
              marker_color="#98A2B0", hovertemplate="%{y:.1f} GB/s<extra>specification</extra>")
    f.add_bar(x=grades, y=meas, name="Sustained in simulation",
              marker_color=C["cxl"], hovertemplate="%{y:.2f} GB/s<extra>measured</extra>")
    f.add_scatter(x=grades, y=eff, name="Share of specification reached", yaxis="y2",
                  mode="lines+markers", line=dict(color=C["ink"], width=2, dash="dot"),
                  marker=dict(size=7), hovertemplate="%{y:.1f}%<extra></extra>")
    f.update_layout(**fig_shell(
        400,
        barmode="group",
        bargap=0.32,
        yaxis=dict(title="GB/s", range=[0, 28]),
        yaxis2=dict(title="percent", overlaying="y", side="right", range=[50, 100], dtick=10,
                    showgrid=False, title_font=dict(size=13, color=C["muted"]),
                    tickfont=dict(color=C["muted"])),
        margin=dict(l=64, r=64, t=16, b=56),
    ))
    return f


def fig_latency() -> go.Figure:
    grades = [r[0] for r in DRAM_SWEEP]
    lat = [r[4] for r in DRAM_SWEEP]
    f = go.Figure()
    f.add_bar(x=lat, y=grades, orientation="h", marker_color=C["cxl"],
              text=[f"{v:.1f} ns" for v in lat], textposition="outside",
              textfont=dict(family=MONO, size=13, color=C["body"]),
              hovertemplate="%{x:.1f} ns<extra></extra>")
    f.update_layout(**fig_shell(
        300,
        xaxis=dict(title="loaded read latency, ns", range=[0, 275]),
        yaxis=dict(showgrid=False),
        margin=dict(l=110, r=24, t=8, b=48),
    ))
    return f


def fig_coverage() -> go.Figure:
    f = go.Figure()
    for d in ["prose", "chat", "math", "code"]:
        f.add_scatter(x=COVERAGE_BATCH, y=COVERAGE[d], name=d, mode="lines+markers",
                      line=dict(color=DOMAIN_COLOR[d], width=2.5), marker=dict(size=8),
                      hovertemplate="batch %{x}: %{y:.1%} of experts<extra>" + d + "</extra>")
    f.add_hline(y=1.0, line=dict(color=C["rule"], width=1, dash="dash"))
    f.add_annotation(x=3.55, y=1.02, text="every expert touched", showarrow=False,
                     font=dict(size=12, color=C["muted"]), xanchor="right")
    f.update_layout(**fig_shell(
        380,
        xaxis=dict(title="tokens per batch", type="log", tickvals=COVERAGE_BATCH,
                   ticktext=[str(b) for b in COVERAGE_BATCH]),
        yaxis=dict(title="share of experts touched per layer", range=[0.45, 1.08],
                   tickformat=".0%"),
    ))
    return f


def fig_overlap() -> go.Figure:
    z, text = [], []
    for r in DOMAINS:
        zr, tr = [], []
        for c in DOMAINS:
            v = OVERLAP.get((r, c)) or OVERLAP.get((c, r))
            zr.append(v)
            tr.append(f"{v:.1f}%")
        z.append(zr)
        text.append(tr)
    f = go.Figure(go.Heatmap(
        z=z, x=DOMAINS, y=DOMAINS, text=text, texttemplate="%{text}",
        textfont=dict(family=MONO, size=15),
        zmin=0, zmax=100, colorscale=[[0.0, "#F7E9E5"], [0.25, "#FFFFFF"],
                                      [0.6, "#BFD0E8"], [1.0, "#1B4FA8"]],
        colorbar=dict(title=dict(text="overlap %", font=dict(size=12, color=C["muted"])),
                      thickness=12, outlinewidth=0, tickfont=dict(size=12, color=C["muted"])),
        hovertemplate="%{y} vs %{x}: %{z:.1f}%<extra></extra>",
    ))
    f.update_layout(**fig_shell(
        380,
        xaxis=dict(showgrid=False, side="bottom"),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=80, r=24, t=16, b=48),
    ))
    return f


def fig_penalty() -> go.Figure:
    text = [[f"{v:.3f}" for v in row] for row in PENALTY]
    f = go.Figure(go.Heatmap(
        z=PENALTY, x=PENALTY_COLS, y=PENALTY_ROWS, text=text, texttemplate="%{text}",
        textfont=dict(family=MONO, size=15),
        zmin=0.35, zmax=0.85, zmid=0.5,
        colorscale=[[0.0, "#E8B4B8"], [0.3, "#FFFFFF"], [1.0, "#7FA8D9"]],
        colorbar=dict(title=dict(text="hit rate", font=dict(size=12, color=C["muted"])),
                      thickness=12, outlinewidth=0, tickfont=dict(size=12, color=C["muted"])),
        hovertemplate="profiled on %{y}, running %{x}: %{z:.3f}<extra></extra>",
    ))
    f.update_layout(**fig_shell(
        380,
        xaxis=dict(title="workload actually served", showgrid=False),
        yaxis=dict(title="workload the placement was profiled on", showgrid=False,
                   autorange="reversed"),
        margin=dict(l=110, r=24, t=16, b=56),
    ))
    return f


def fig_kv() -> go.Figure:
    names = [r[0] for r in KV_STEPS]
    steps = [r[2] for r in KV_STEPS]
    cols = [C["hbm"], C["cxl"], C["cxl"]]
    f = go.Figure()
    f.add_bar(x=names, y=steps, marker_color=cols,
              text=[f"{s} steps/s" for s in steps], textposition="outside",
              textfont=dict(family=MONO, size=14, color=C["body"]),
              hovertemplate="%{y} decode steps per second<extra></extra>")
    f.update_layout(**fig_shell(
        340,
        yaxis=dict(title="decode steps per second", type="log", range=[0.2, 3.1]),
        xaxis=dict(showgrid=False),
        showlegend=False,
    ))
    return f


def fig_pooling() -> go.Figure:
    nodes = [r[0] for r in POOLING]
    saving = [r[2] / r[3] for r in POOLING]
    bstar = [r[4] / POOLING[0][4] for r in POOLING]
    f = go.Figure()
    f.add_scatter(x=nodes, y=saving, name="capacity you save", mode="lines+markers",
                  line=dict(color=C["good"], width=6), marker=dict(size=10),
                  opacity=0.55, hovertemplate="%{y:.0f}x capacity<extra></extra>")
    f.add_scatter(x=nodes, y=bstar, name="batch size you now need", mode="lines+markers",
                  line=dict(color=C["bad"], width=2.5, dash="dot"), marker=dict(size=8),
                  hovertemplate="%{y:.0f}x batch<extra></extra>")
    f.add_annotation(x=0.6, y=0.55, xref="paper", yref="paper",
                     text="The two lines sit exactly on top of each other.<br>"
                          "That coincidence is the result.",
                     showarrow=False, align="left", xanchor="left",
                     font=dict(size=13, color=C["muted"]))
    f.update_layout(**fig_shell(
        340,
        xaxis=dict(title="nodes sharing the pool", type="log", tickvals=nodes,
                   ticktext=[str(n) for n in nodes]),
        yaxis=dict(title="multiple of the single-node case", type="log",
                   tickvals=[1, 2, 4, 8, 16], ticktext=["1x", "2x", "4x", "8x", "16x"]),
    ))
    return f


def fig_granularity() -> go.Figure:
    names = [r[0] for r in GRANULARITY]
    share = [r[2] for r in GRANULARITY]
    cols = [C["good"] if s < 5 else (C["neutral"] if s < 30 else C["bad"]) for s in share]
    f = go.Figure()
    f.add_bar(x=names, y=share, marker_color=cols,
              text=[f"{s:.2f}%" for s in share], textposition="outside",
              textfont=dict(family=MONO, size=13, color=C["body"]),
              hovertemplate="%{y:.2f}% of transfer time is latency<extra></extra>")
    f.update_layout(**fig_shell(
        340,
        yaxis=dict(title="percent of transfer time", range=[0, 100]),
        xaxis=dict(showgrid=False),
        showlegend=False,
        margin=dict(l=76, r=24, t=28, b=56),
    ))
    return f


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def plot(fig: go.Figure, div_id: str) -> str:
    inner = fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                        config={"displayModeBar": False, "responsive": True})
    return f'<div class="plotbox">{inner}</div>' 


def table(headers, rows, aligns=None, note=None) -> str:
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    head = "".join(f'<th class="a-{a}">{h}</th>' for h, a in zip(headers, aligns))
    body = ""
    for r in rows:
        cells = "".join(
            f'<td class="a-{a}{" num" if a == "right" else ""}">{c}</td>'
            for c, a in zip(r, aligns))
        body += f"<tr>{cells}</tr>"
    cap = f'<p class="tnote">{note}</p>' if note else ""
    return f'<div class="twrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{cap}'


SECTIONS = [
    ("summary", "What we measured"),
    ("tiers", "The memory tiers"),
    ("routing", "How real routing behaves"),
    ("placement", "Why placement does not help"),
    ("capacity", "The capacity model"),
    ("kv", "Experts versus KV cache"),
    ("transfer", "Pooling and transfer size"),
    ("method", "Method and limits"),
]


def build() -> str:
    figs = {
        "dram": plot(fig_dram(), "f-dram"),
        "lat": plot(fig_latency(), "f-lat"),
        "cov": plot(fig_coverage(), "f-cov"),
        "ovl": plot(fig_overlap(), "f-ovl"),
        "pen": plot(fig_penalty(), "f-pen"),
        "kv": plot(fig_kv(), "f-kv"),
        "pool": plot(fig_pooling(), "f-pool"),
        "gran": plot(fig_granularity(), "f-gran"),
    }

    try:
        from plotly.offline import get_plotlyjs
        plotly_js = f"<script>{get_plotlyjs()}</script>"
    except Exception:
        plotly_js = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

    nav = "".join(
        f'<a href="#{sid}" data-sec="{sid}">{name}</a>' for sid, name in SECTIONS)

    model_opts = "".join(f'<option value="{m}">{m}</option>' for m in MODELS)
    model_js = json.dumps({m: dict(E=v["experts"], k=v["k"]) for m, v in MODELS.items()})

    capacity_rows = [
        (m, f"{v['weights']:,.1f}", f"{v['pool']:,.1f}",
         f"{100 * v['pool'] / v['weights']:.1f}%", v["gpus_hbm"], v["gpus_cxl"])
        for m, v in MODELS.items()
    ]

    css = """
:root{
  --paper:%(paper)s; --surface:%(surface)s; --ink:%(ink)s; --body:%(body)s;
  --muted:%(muted)s; --rule:%(rule)s; --rule-soft:%(rule_soft)s;
  --hbm:%(hbm)s; --cxl:%(cxl)s; --bad:%(bad)s; --good:%(good)s;
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px; --s7:48px; --s8:64px;
  --font:%(font)s; --mono:%(mono)s;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--body);font-family:var(--font);
  font-size:16px;line-height:1.6;-webkit-font-smoothing:antialiased}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}

.shell{display:grid;grid-template-columns:240px minmax(0,1fr);gap:var(--s7);
  max-width:1360px;margin:0 auto;padding:0 var(--s5)}

/* masthead */
.mast{border-bottom:1px solid var(--rule);background:var(--surface)}
.mast-in{max-width:1360px;margin:0 auto;padding:var(--s7) var(--s5) var(--s6)}
.mast h1{font-size:34px;line-height:1.15;font-weight:650;color:var(--ink);
  margin:0 0 var(--s2);letter-spacing:-0.02em}
.mast p.sub{margin:0;color:var(--muted);font-size:16px;max-width:62ch}
.byline{margin:var(--s4) 0 0;font-size:14px;color:var(--muted)}

/* the three constants everything rests on */
.constants{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
  gap:0;margin-top:var(--s6);border-top:2px solid var(--ink)}
.constant{padding:var(--s5) var(--s5) var(--s5) 0;border-right:1px solid var(--rule)}
.constant:last-child{border-right:0}
.constant .v{font-family:var(--mono);font-size:30px;font-weight:600;color:var(--ink);
  letter-spacing:-0.02em;line-height:1.1}
.constant .l{font-size:15px;color:var(--body);margin-top:var(--s2)}
.constant .src{font-size:13px;color:var(--muted);margin-top:var(--s1)}

/* nav rail */
nav{position:sticky;top:var(--s5);align-self:start;padding-top:var(--s7);
  border-right:1px solid var(--rule);min-height:60vh}
nav a{display:block;padding:10px var(--s4) 10px 0;color:var(--muted);
  text-decoration:none;font-size:15px;border-right:2px solid transparent;margin-right:-1px}
nav a:hover{color:var(--ink)}
nav a.on{color:var(--ink);font-weight:600;border-right-color:var(--ink)}
nav a:focus-visible{outline:2px solid var(--hbm);outline-offset:2px}

main{padding:var(--s7) 0 var(--s8);min-width:0}
section{padding-bottom:var(--s8);scroll-margin-top:var(--s5)}
section > h2{font-size:26px;font-weight:650;color:var(--ink);margin:0 0 var(--s3);
  letter-spacing:-0.015em}
section > p.lede{font-size:17px;max-width:68ch;margin:0 0 var(--s6)}
h3{font-size:19px;font-weight:600;color:var(--ink);margin:var(--s7) 0 var(--s2)}
p{max-width:70ch}

.card{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
  padding:var(--s5);margin-bottom:var(--s5)}
.card h3{margin-top:0}
.reads{font-size:15px;color:var(--muted);margin:var(--s2) 0 var(--s4);max-width:72ch}

/* findings */
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:var(--s4);margin:var(--s5) 0}
.finding{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
  padding:var(--s5);border-top:3px solid var(--cxl)}
.finding h3{margin:0 0 var(--s2);font-size:17px}
.finding p{margin:0;font-size:15px}
.finding .fig{font-family:var(--mono);font-size:22px;font-weight:600;color:var(--ink);
  display:block;margin-bottom:var(--s2)}

/* tables */
.twrap{overflow-x:auto;margin:var(--s4) 0 0}
table{border-collapse:collapse;width:100%%;font-size:15px}
th{text-align:left;font-weight:600;color:var(--ink);font-size:14px;
  padding:var(--s2) var(--s4) var(--s2) 0;border-bottom:2px solid var(--ink);
  white-space:nowrap}
td{padding:10px var(--s4) 10px 0;border-bottom:1px solid var(--rule-soft);
  vertical-align:baseline}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:14.5px}
.a-right{text-align:right}
.a-left{text-align:left}
tr:last-child td{border-bottom:0}
.tnote{font-size:13.5px;color:var(--muted);margin:var(--s3) 0 0;max-width:72ch}
mark{background:#FDF3D8;color:var(--ink);padding:1px 3px;border-radius:2px}

/* calculator */
.calc{display:grid;grid-template-columns:minmax(0,320px) minmax(0,1fr);gap:var(--s6);
  align-items:start}
.controls label{display:block;font-size:14px;color:var(--ink);font-weight:600;
  margin-bottom:var(--s2)}
.control{margin-bottom:var(--s5)}
.control .val{font-family:var(--mono);color:var(--cxl);font-weight:600}
select,input[type=range]{width:100%%}
select{font-family:var(--font);font-size:16px;padding:10px var(--s3);
  border:1px solid var(--rule);border-radius:4px;background:var(--surface);
  color:var(--ink);min-height:44px}
input[type=range]{accent-color:var(--cxl);height:44px}
select:focus-visible,input:focus-visible{outline:2px solid var(--hbm);outline-offset:2px}
.readout{border-top:2px solid var(--ink);padding-top:var(--s4)}
.readout .big{font-family:var(--mono);font-size:44px;font-weight:600;color:var(--ink);
  line-height:1.05;letter-spacing:-0.03em}
.readout .unit{font-size:16px;color:var(--muted);margin-top:var(--s2)}
.verdict{margin-top:var(--s5);padding:var(--s4);border-radius:4px;font-size:15px}
.verdict.ok{background:#EAF3EE;border-left:3px solid var(--good);color:#14432B}
.verdict.no{background:#FBEEEF;border-left:3px solid var(--bad);color:#6E1420}
.formula{font-family:var(--mono);font-size:15px;background:#F4F6F8;
  border:1px solid var(--rule);border-radius:4px;padding:var(--s4);
  color:var(--ink);overflow-x:auto;margin:var(--s4) 0}

.note{border-left:3px solid var(--rule);padding:var(--s2) 0 var(--s2) var(--s4);
  color:var(--muted);font-size:15px;margin:var(--s5) 0;max-width:70ch}
.pill{display:inline-block;font-size:13px;padding:2px 8px;border-radius:3px;
  background:#F0F2F5;color:var(--muted);margin-left:var(--s2);vertical-align:middle}
.pill.meas{background:#EAF3EE;color:#14432B}
.pill.scaled{background:#FDF3D8;color:#6B4D06}
code{font-family:var(--mono);font-size:0.92em;background:#F4F6F8;padding:1px 5px;
  border-radius:3px}

footer{border-top:1px solid var(--rule);padding:var(--s6) 0 var(--s8);
  color:var(--muted);font-size:14px}

.plotbox{margin-top:var(--s2)}

@media (max-width:900px){
  .plotbox{overflow-x:auto;-webkit-overflow-scrolling:touch;
    margin-left:calc(-1 * var(--s5));margin-right:calc(-1 * var(--s5));
    padding:0 var(--s5)}
  .plotbox > div{min-width:600px}
  .shell{grid-template-columns:1fr;gap:0;padding:0 var(--s4)}
  nav{position:sticky;top:0;z-index:20;display:flex;overflow-x:auto;gap:var(--s4);
    padding:var(--s2) 0;border-right:0;border-bottom:1px solid var(--rule);
    background:var(--paper);min-height:0}
  nav a{white-space:nowrap;padding:10px 0;border-right:0;border-bottom:2px solid transparent}
  nav a.on{border-bottom-color:var(--ink)}
  .constants{grid-template-columns:1fr}
  .constant{border-right:0;border-bottom:1px solid var(--rule);padding-right:0}
  .constant:last-child{border-bottom:0}
  .calc{grid-template-columns:1fr}
  .mast h1{font-size:27px}
  .mast-in{padding:var(--s6) var(--s4) var(--s5)}
  main{padding-top:var(--s6)}
  section > h2{font-size:22px}
}
""" % dict(C, font=FONT, mono=MONO)

    js = """
const MODELS = %(models)s;
const els = id => document.getElementById(id);
const fmt = n => n >= 1e6 ? (n/1e6).toFixed(2)+" M" : Math.round(n).toLocaleString();

function recompute(){
  const m = els('c-model').value;
  const h = +els('c-h').value / 1000;
  const eps = +els('c-eps').value / 100;
  const bw = +els('c-bw').value;
  const {E, k} = MODELS[m];
  const B = (1 - h) * E * 3.956e14 / (bw * 1e9 * k * eps);

  els('c-h-v').textContent = (h*100).toFixed(1) + "%%";
  els('c-eps-v').textContent = (eps*100).toFixed(0) + "%%";
  els('c-bw-v').textContent = bw.toFixed(2) + " GB/s";
  els('c-out').textContent = fmt(B);

  const v = els('c-verdict');
  if (B <= 256){
    v.className = "verdict ok";
    v.textContent = "Reachable during decode. Continuous batching runs 64 to 256 concurrent sequences, so this configuration works in normal serving.";
  } else if (B <= 65536){
    v.className = "verdict ok";
    v.textContent = "Reachable during prefill. A batch this size is a long prompt or a batched prefill, not a decode step.";
  } else {
    v.className = "verdict no";
    v.textContent = "Out of reach. No real deployment batches this many tokens at once. Buy more HBM, more bandwidth, or accept a larger stall.";
  }
}

['c-model','c-h','c-eps','c-bw'].forEach(id =>
  els(id).addEventListener('input', recompute));
recompute();

// nav highlighting
const links = [...document.querySelectorAll('nav a')];
const obs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting){
      links.forEach(a => a.classList.toggle('on', a.dataset.sec === e.target.id));
    }
  });
}, {rootMargin: "-20%% 0px -70%% 0px"});
if (links.length) links[0].classList.add('on');
document.querySelectorAll('section').forEach(s => obs.observe(s));
""" % dict(models=model_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MEMoE: CXL memory expansion for Mixture-of-Experts inference</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;650&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>{css}</style>
{plotly_js}
</head>
<body>

<header class="mast">
  <div class="mast-in">
    <h1>Can CXL memory hold the experts?</h1>
    <p class="sub">A measured answer for Mixture-of-Experts inference. Yes during prefill,
      no during decode, and not for the reason we expected when we started.</p>
    <p class="byline">Team King Bob &middot; Shashank, Shreyansh Jain, Kanav Sharma &middot;
      Nebula 2026, software category</p>

    <div class="constants">
      <div class="constant">
        <div class="v">16.88 GB/s</div>
        <div class="l">What one CXL channel actually sustains</div>
        <div class="src">DRAMSim3, DDR4-3200, 12 MB sequential expert read</div>
      </div>
      <div class="constant">
        <div class="v">179.4 ns</div>
        <div class="l">Loaded read latency on that channel</div>
        <div class="src">Same run. Irrelevant at expert granularity, dominant below it.</div>
      </div>
      <div class="constant">
        <div class="v">258,629</div>
        <div class="l">Tokens of real routing we captured</div>
        <div class="src">OLMoE-1B-7B forward hooks across prose, math, code, chat</div>
      </div>
    </div>
  </div>
</header>

<div class="shell">
<nav aria-label="Sections">{nav}</nav>
<main>

<section id="summary">
  <h2>What we measured</h2>
  <p class="lede">Routed experts are 93 to 97 percent of every checkpoint we studied, and
    they sit idle almost all the time. That is the part CXL could hold. The question is
    what it costs, and the answer turned on three things we did not expect.</p>

  <div class="findings">
    <div class="finding">
      <h3>Batching stops helping</h3>
      <span class="fig">97 to 100%</span>
      <p>of experts are touched per layer once a batch reaches 1,024 tokens. Past that
        point the fast tier is a fixed partition, not a cache.</p>
    </div>
    <div class="finding">
      <h3>Popularity buys nothing</h3>
      <span class="fig">0.899</span>
      <p>fetch hit rate at 90 percent residency, identical at every skew we tested
        including perfectly uniform routing. Ranking experts by popularity is wasted work.</p>
    </div>
    <div class="finding">
      <h3>Profiling can backfire</h3>
      <span class="fig">0.383</span>
      <p>hit rate when a placement tuned on prose serves code. Random placement scores
        0.500. The wrong profile is worse than no profile.</p>
    </div>
  </div>

  <p>Those three together move the design away from a popularity-based tiering engine and
    toward two decisions that do matter: how much fast memory you buy, and how early you
    start the transfer.</p>

  <div class="card">
    <h3>What fits where</h3>
    <p class="reads">Weights and expert pool in GB. GPU counts assume 80 GB of HBM per
      device, and 512 GB of CXL per device in the right-hand column.</p>
    {table(["Model", "Weights", "Expert pool", "Experts as share", "GPUs, HBM only", "GPUs with CXL"], capacity_rows)}
  </div>
</section>

<section id="tiers">
  <h2>The memory tiers</h2>
  <p class="lede">We did not want to quote CXL bandwidth from a paper. A CXL Type-3
    expander is DRAM behind a controller, so we simulated the DRAM directly and read the
    rate it actually sustained while busy.</p>

  <div class="card">
    <h3>Faster DRAM returns a smaller share of its rating<span class="pill meas">measured</span></h3>
    <p class="reads">Grey is what the DIMM is sold as. Orange is what a bulk expert fetch
      actually got. The dotted line is the ratio between them, on the right-hand axis.</p>
    {figs["dram"]}
    <p class="tnote">Efficiency falls from 85.5 percent at DDR4-1866 to 65.9 percent at
      DDR4-3200, because timing constraints like tRC, tFAW and refresh do not scale with
      the data rate. Anyone specifying a CXL expander should budget against the measured
      figure, not the DIMM peak.</p>
  </div>

  <div class="card">
    <h3>Loaded read latency</h3>
    <p class="reads">Latency improves as the grade rises, but far more slowly than
      bandwidth. It stops mattering entirely once transfers are expert-sized.</p>
    {figs["lat"]}
  </div>

  <div class="note">Two numbers on this page are not measurements and we label them
    everywhere they appear. HBM bandwidth is taken from specification, because DRAMSim3's
    trace driver issues one request per cycle and caps observable bandwidth below what real
    HBM does. And the four-channel figure of 67.5 GB/s is the measured single channel scaled
    by four, because multi-channel runs did not scale in simulation even with the address
    mapping interleaving correctly.</div>
</section>

<section id="routing">
  <h2>How real routing behaves</h2>
  <p class="lede">Everything about expert tiering depends on how skewed real routers are,
    so we captured routing from a live checkpoint instead of assuming a distribution.</p>

  <div class="card">
    <h3>Skew is a property of the input, not the router</h3>
    <p class="reads">Same model, same weights, four workloads. Zipf s varies by a factor of
      2.3. Normalised entropy of 1.0 would be perfectly flat.</p>
    {table(
        ["Workload", "Tokens", "Zipf s", "Entropy", "Gini", "Top quarter of experts"],
        [(d, f"{t:,}", f"{s:.3f}", f"{e:.3f}", f"{g:.3f}", f"{p:.1f}%") for d, t, s, e, g, p in ROUTING],
        note="Chat routing is nearly uniform, with no expert taking more than 4 percent of "
             "routings. That is what a load-balancing auxiliary loss during training is "
             "designed to produce.")}
  </div>

  <div class="card">
    <h3>Coverage saturates before serving batch sizes begin</h3>
    <p class="reads">Share of experts touched per layer, per batch. Once the lines reach
      the top, every non-resident expert is fetched on every batch forever. There is no
      reuse distance left to shorten.</p>
    {figs["cov"]}
    <p class="tnote">Both real serving regimes sit inside the saturated zone. Continuous
      batching decode runs 64 to 256 concurrent sequences. Prefill runs thousands of
      tokens. Neither is small batch.</p>
  </div>
</section>

<section id="placement">
  <h2>Why placement does not help</h2>
  <p class="lede">The standard approach to expert offload is to profile which experts are
    popular and keep those in fast memory. We tested it and it does nothing, then tested
    whether it could actively hurt, and it can.</p>

  <div class="card">
    <h3>Hit rate equals residency, at every skew</h3>
    <p class="reads">Fraction of expert fetches avoided at 90 percent residency, swept from
      perfectly uniform routing to more skewed than anything we measured.</p>
    {table(
        ["Batch"] + list(SKEW_SWEEP.keys()),
        [[str(b)] + [f"{SKEW_SWEEP[s][i]:.3f}" for s in SKEW_SWEEP]
         for i, b in enumerate(SKEW_SWEEP_BATCH)],
        note="At batch 512 and above the hit rate equals the resident fraction exactly. "
             "Static placement, LRU and sampled LFU all perform identically, which is "
             "convenient: static placement is decided once at load time and costs nothing "
             "on the critical path.")}
  </div>

  <div class="card">
    <h3>Hot expert sets split into two camps</h3>
    <p class="reads">Overlap between the top quarter of experts in each pair of workloads.
      Independent sets would overlap 25 percent by chance. The diagonal is a split-half
      control on the same workload, which has to be high for the rest to mean anything.</p>
    {figs["ovl"]}
    <p class="tnote">Prose and chat share experts. Math and code share experts. Across the
      two groups, overlap drops <em>below</em> chance: code routes to experts that prose
      actively avoids. That is domain specialisation visible in routing statistics alone,
      from 260,000 tokens on a 7B model, with no interpretability tooling.</p>
  </div>

  <div class="card">
    <h3>The wrong profile is worse than no profile</h3>
    <p class="reads">Each row is a placement profiled on one workload. Each column is the
      workload it then had to serve. Random placement would score 0.500 everywhere.</p>
    {figs["pen"]}
    <p class="tnote">Prose profiled onto code scores 0.383, and code profiled onto prose
      scores 0.448. Both are worse than picking experts at random. For mixed serving traffic
      there is no good static expert placement, so choose the resident set by capacity and
      stop there.</p>
  </div>
</section>

<section id="capacity">
  <h2>The capacity model</h2>
  <p class="lede">Since skew is irrelevant and policy choice is irrelevant, two things are
    left: how much fast memory you buy, and how well you schedule the transfer. This is the
    first one, as a formula you can put numbers into.</p>

  <div class="formula">B* = (1 - h) &times; E &times; F / (BW &times; k &times; &epsilon;)</div>
  <p>B* is the batch size at which the CXL fetch stops costing more than your stall budget.
    <em>h</em> is the share of fetches the resident tier already covers, <em>E</em> is
    experts per layer, <em>k</em> is experts activated per token, <em>F</em> is per-GPU
    compute, and <em>&epsilon;</em> is the stall you will tolerate. B* does not depend on
    expert size, and it does not depend on GPU count. Scaling out does not get you past it.</p>

  <div class="card">
    <h3>Try it</h3>
    <p class="reads">Move the sliders and read the batch size you would need for the
      configuration to pay off.</p>
    <div class="calc">
      <div class="controls">
        <div class="control">
          <label for="c-model">Model</label>
          <select id="c-model">{model_opts}</select>
        </div>
        <div class="control">
          <label for="c-h">Experts kept in HBM <span class="val" id="c-h-v"></span></label>
          <input type="range" id="c-h" min="0" max="999" step="1" value="900">
        </div>
        <div class="control">
          <label for="c-eps">Stall you will tolerate <span class="val" id="c-eps-v"></span></label>
          <input type="range" id="c-eps" min="2" max="25" step="1" value="10">
        </div>
        <div class="control">
          <label for="c-bw">CXL bandwidth <span class="val" id="c-bw-v"></span></label>
          <input type="range" id="c-bw" min="16.88" max="200" step="0.01" value="67.5">
        </div>
      </div>
      <div>
        <div class="readout">
          <div class="big" id="c-out">&mdash;</div>
          <div class="unit">tokens per batch needed before CXL pays for itself</div>
        </div>
        <div class="verdict" id="c-verdict"></div>
      </div>
    </div>
    <p class="tnote">Compute is fixed at 3.956 &times; 10<sup>14</sup> FLOP/s, which is 40
      percent of an H100's BF16 peak. The model counts routed expert GEMMs only, so
      attention and dense layers would hide more transfer than this admits. The numbers are
      conservative in that direction.</p>
  </div>

  <div class="card">
    <h3>What that means for a real deployment<span class="pill meas">from real traces</span></h3>
    <p class="reads">Share of the expert pool that can live on CXL inside a 10 percent stall
      budget, running the captured traces against the measured tier.</p>
    {table(["Tokens per batch", "One channel, 16.88 GB/s", "Four channels, 67.5 GB/s"],
           [(b, a, c) for b, a, c in OFFLOADABLE],
           note="Only the code trace was long enough to reach the largest batches. At a "
                "four-channel expander and a 64k prefill batch, 82 percent of the expert "
                "pool can sit on CXL. This is a prefill technique, not a decode technique.")}
  </div>
</section>

<section id="kv">
  <h2>Experts versus KV cache</h2>
  <p class="lede">The KV cache is the other thing that destroys HBM budgets, so we asked
    whether it is also a CXL candidate. It is not, and the reason is clean enough to state
    as a rule.</p>

  <p>A tier with bandwidth BW feeding a device at F FLOP/s is balanced at F/BW FLOPs per
    byte. Below that ratio you are bandwidth bound. Each fetched expert serves
    <code>B &times; k / E</code> tokens, so expert intensity <strong>grows with the
    batch</strong>. Each cached KV element is read once per step and used by
    <code>n_heads / n_kv_heads</code> query heads, so KV intensity is <strong>fixed</strong>.
    Batching cannot rescue it, because every sequence carries its own KV.</p>

  <div class="card">
    <h3>One 32k-context Qwen3 sequence, 5.875 GB of KV per step</h3>
    {figs["kv"]}
    <p class="tnote">A 48 times slowdown on the four-channel tier. Active KV must stay in
      HBM. Cold KV is a different question, since preempted requests and prefix caches are
      not read every step, but we did not model that.</p>
  </div>

  <div class="card">
    <h3>Arithmetic intensity, by model</h3>
    {table(
        ["Model", "KV intensity", "Balance point on 4-channel CXL", "Batch for experts to match KV"],
        [("OLMoE-1B-7B", "1.0", "5,861", "46,886"),
         ("Mixtral-8x7B", "4.0", "5,861", "23,443"),
         ("Qwen3-235B-A22B", "16.0", "5,861", "93,772"),
         ("DeepSeek-V3", "1.0", "5,861", "166,706")],
        note="Qwen3's 16-to-1 grouped query attention gives the best KV intensity in the "
             "set, and it is still 366 times below the balance point.")}
  </div>
</section>

<section id="transfer">
  <h2>Pooling and transfer size</h2>
  <p class="lede">Two smaller results that change what you would build.</p>

  <div class="card">
    <h3>Pooling trades capacity for batch size, one for one</h3>
    <p class="reads">N nodes sharing one copy of the cold experts save N times the capacity,
      but contend for one link, so per-node bandwidth falls as 1/N. B* is inversely
      proportional to bandwidth, so both move together. Qwen3-235B against a 67.5 GB/s pool.</p>
    {figs["pool"]}
    <p class="tnote">At 16 nodes you save 6.3 TB and need a 1.5 million token batch, which
      is not a real operating point. Pooling is worth it at small N, or when the pool link
      scales with membership. The latter is what CXL 3.0 switching promises, and it is
      exactly the part that is pre-production, so we model it and say so.</p>
  </div>

  <div class="card">
    <h3>Fetch whole experts, never fragments<span class="pill scaled">modelled at 67.5 GB/s</span></h3>
    <p class="reads">Share of transfer time spent waiting on latency rather than moving
      bytes, at 300 ns access latency. OLMoE, 12 MB per expert.</p>
    {figs["gran"]}
    <p class="tnote">Sweeping latency from 200 to 800 ns barely moves anything at expert
      granularity. Anything that chunks below roughly 512 KB converts a bandwidth problem
      into a latency problem.</p>
  </div>
</section>

<section id="method">
  <h2>Method and limits</h2>
  <p class="lede">What produced these numbers, and what they do not cover.</p>

  <div class="card">
    <h3>Where each number comes from</h3>
    {table(["Result", "Produced by", "Status"],
           [("Expert pool and GPU counts", "Analytical footprint model from YAML configs",
             "Validated against published parameter counts"),
            ("CXL bandwidth and latency", "DRAMSim3, DDR4 x8, 12 MB sequential read", "Measured"),
            ("Four-channel bandwidth", "Single channel scaled by four", "Scaled"),
            ("HBM bandwidth", "Specification", "Not measured, and never enters the stall model"),
            ("Routing skew and coverage", "OLMoE-1B-7B forward hooks, 258,629 tokens", "Measured"),
            ("Placement and prefetch results", "Trace-driven simulator, 36 tests", "Simulated on real traces"),
            ("Pooling", "Analytical, CXL 3.0 switching", "Modelled, no silicon exists to measure"),
            ("CXL software path", "QEMU 9.1.0, Ubuntu 24.04 guest, CXL Type-3 device",
             "Enumeration works, region commit blocked")],
           aligns=["left", "left", "left"])}
  </div>

  <div class="card">
    <h3>The QEMU result</h3>
    <p>Everything up to region commit worked. The kernel negotiated CXL control through
      ACPI <code>_OSC</code>, all four drivers bound, and the full decoder hierarchy
      enumerated with correct sizes and targets. Region creation then failed, because
      before committing a region the kernel must invalidate CPU caches over the range and
      gates that on a check which returns false whenever <code>X86_FEATURE_HYPERVISOR</code>
      is set, which is to say inside any virtual machine. The bypass exists
      (<code>CONFIG_CXL_REGION_INVALIDATION_TEST</code>) and Ubuntu's stock kernel does not
      enable it.</p>
    <p>We confirmed it by elimination: identical failure across volatile and persistent
      device modes, with and without an explicit region size, and with the memory window
      both oversized and matched exactly. This establishes that the CXL software stack is
      real and functional through discovery, driver binding and decoder programming, and
      that region management cannot be exercised in a VM on a stock kernel. QEMU's CXL
      support is functional emulation, so even a successful commit could not have validated
      a performance claim.</p>
  </div>

  <div class="card">
    <h3>What we are not claiming</h3>
    <p class="reads">Ordered by how much each one could change a conclusion.</p>
    {table(["Limit", "Effect"],
           [("Contention and queueing are not modelled",
             "The clearest gap. Concurrent KV traffic over the same link is not simulated."),
            ("Routing comes from one model, OLMoE-1B-7B",
             "Skew and specialisation may differ on larger MoE checkpoints."),
            ("Cross-domain analysis equalises to the shortest trace",
             "That analysis rests on 18,239 tokens per workload."),
            ("Compute counts routed expert GEMMs only, at 40% MFU",
             "Attention and dense layers would hide more transfer. Overheads are conservative."),
            ("Four-channel bandwidth is scaled, not measured",
             "The headline offload figures inherit that assumption."),
            ("Pooling is modelled",
             "No shipping CXL 3.0 switch silicon exists to measure against."),
            ("We simulate placement and transfer, not kernels",
             "A real implementation must also solve overlap scheduling and fragmentation.")],
           aligns=["left", "left"])}
  </div>

  <div class="card">
    <h3>Recommendations</h3>
    <p>Put experts on CXL and keep active KV in HBM; the distinction is arithmetic
      intensity, not size. Treat it as a prefill technique. Do not build a popularity-based
      tiering engine, and choose the resident set by capacity instead. Spend the engineering
      effort on prefetch scheduling, depth 4 to 8, aimed at the head of the cold tail rather
      than the hot experts the fast tier already holds. Fetch whole experts. Specify the
      expander by measured bandwidth rather than DIMM peak. Be cautious about pooling beyond
      a few nodes.</p>
  </div>
</section>

<footer>
  <p>MEMoE &middot; Team King Bob &middot; Nebula 2026. Reproduce with
    <code>uv run python scripts/run_all.py</code>,
    <code>run_real.py</code>, <code>run_extras.py</code>,
    <code>sweep_dramsim.py</code>, then <code>build_dashboard.py</code>.
    Routing capture needs a GPU and lives in <code>capture_traces.py</code>.</p>
</footer>

</main>
</div>

<script>{js}</script>
</body>
</html>
"""
    return html


def main() -> None:
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dashboard.html"
    path.write_text(build(), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
