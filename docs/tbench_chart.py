"""Generate docs/tbench-size-vs-score.png — the Terminal-Bench 2.0 size-vs-accuracy chart.

Run:  uv run --with matplotlib python docs/tbench_chart.py

Data provenance (update the tables below, then re-run):
- Open-weight points: tbench.ai Terminal-Bench 2.0 leaderboard (verified, best entry
  per model), positioned by total parameters.
- Claude reference lines: the leaderboard's own "Claude Code" rows ONLY — Opus 4.6,
  Sonnet 4.5, Haiku 4.5 (proprietary — parameter count undisclosed, hence lines not
  points). Newer Claude models (Opus 4.8, Fable 5) have no public leaderboard entry,
  so they are deliberately not shown.
- Ornith point: provisional in-house run (k=1) — full verified run in flight.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

GRAY = "#8b95a1"
INK = "#1f2d3d"
ORANGE = "#e8590c"
BAND = "#fdf1e3"

# (params_B, score_%, label, label_dx_pts, label_dy_pts, ha)
# label=None → unlabeled secondary entry for the same/nearby model family.
POINTS = [
    (9,    9.3,  "little-coder + Qwen3.5-9B",      10,  0,  "left"),
    (20,   3.5,  None,                              0,  0,  "left"),
    (32,   19.4, "Bash Agent + TermiGen-32B",     -10,  0,  "right"),
    (35,   24.6, "little-coder + Qwen3.6-35B",     10,  0,  "left"),
    (120,  19.0, "Terminus 2 + GPT-OSS-120B",      10, -4,  "left"),
    (230,  45.0, "IndusAGI + MiniMax M2.7",       -10,  8,  "right"),
    (230,  42.7, None,                              0,  0,  "left"),
    (360,  52.4, "Terminus 2 + GLM 5",             12,  6,  "left"),
    (360,  33.3, None,                              0,  0,  "left"),
    (360,  24.4, None,                              0,  0,  "left"),
    (480,  27.1, "Dakou + Qwen3-Coder-480B",       12, -14, "left"),
    (685,  39.5, None,                              0,  0,  "left"),
    (1000, 35.6, "Terminus 2 + DeepSeek-V3.2",    -12,  8,  "right"),
    (1000, 43.2, "Terminus 2 + Kimi K2.5",        -12, 10,  "right"),
]

CHAD = (35, 40.0)   # provisional — full verified run in flight

# (score_%, label, emphasized) — leaderboard "Claude Code" rows only
REFERENCE_LINES = [
    (58.0, "Claude Opus 4.6", False),
    (40.1, "Claude Sonnet 4.5", True),
    (27.5, "Claude Haiku 4.5", False),
]

fig, ax = plt.subplots(figsize=(10.6, 6.1), dpi=200)
fig.subplots_adjust(left=0.075, right=0.97, top=0.82, bottom=0.14)

ax.set_xscale("log")
ax.set_xlim(6, 3000)
ax.set_ylim(0, 62)

# laptop-class band
ax.axvspan(6, 42, color=BAND, zorder=0)
ax.text(11, 59, "laptop-class  ≤ 40B", color="#c2410c", fontsize=11)

# reference lines. Label side flips per line so a label never collides with a
# nearby point (Sonnet's sits below its line, clear of the Kimi K2.5 point).
LABEL_BELOW = {"Claude Sonnet 4.5"}
for score, label, emph in REFERENCE_LINES:
    below = label in LABEL_BELOW
    ty, va = (score - 1.0, "top") if below else (score + 0.8, "bottom")
    if emph:
        ax.axhline(score, color=INK, lw=2.2, ls=(0, (9, 5)), zorder=2)
        ax.text(2900, ty, label, ha="right", va=va, color=INK, fontsize=13)
    else:
        ax.axhline(score, color="#6b7684", lw=1.2, ls=(0, (6, 4)), zorder=2)
        ax.text(2900, ty, label, ha="right", va=va, color="#4b5563", fontsize=11.5)

# open-weight points
for x, y, label, dx, dy, ha in POINTS:
    ax.plot(x, y, "o", ms=7, color=GRAY, mec="none", zorder=3)
    if label:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", color="#6b7684", fontsize=10.5)

# chad + Ornith (label above-LEFT, inside the empty part of the laptop band — below is
# Qwen3.6-35B's label, right is MiniMax's)
ax.plot(*CHAD, "o", ms=16, color=ORANGE, mec="white", mew=1.5, zorder=4)
ax.annotate("chad + Ornith 35B\n~40%  (provisional)", CHAD,
            textcoords="offset points", xytext=(-14, 16), ha="right", va="bottom",
            color=ORANGE, fontsize=13.5, zorder=4)

# axes cosmetics
ax.set_xticks([10, 30, 100, 300, 1000])
ax.set_xticklabels(["10B", "30B", "100B", "300B", "1T"], fontsize=12)
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(range(0, 61, 10))
ax.set_yticklabels(["0", "10", "20", "30", "40", "50", "60%"], fontsize=12)
ax.grid(axis="y", color="#e5e7eb", lw=0.8, zorder=1)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(length=0)
ax.set_xlabel("Total parameters  (log scale)", fontsize=13, color=INK, labelpad=8)
ax.set_ylabel("Terminal-Bench 2.0 accuracy", fontsize=13, color=INK)

fig.text(0.055, 0.945, "Sonnet on your laptop", fontsize=21, color=INK)
fig.text(0.075, 0.845,
         "chad + Ornith (a 35B MoE) lands at the Claude Sonnet 4.5 line on "
         "Terminal-Bench 2.0\n— matching open models many times its size.",
         fontsize=12.5, color="#4b5563", va="bottom", linespacing=1.4)
fig.text(0.055, 0.015,
         "Open-weight entries positioned by size (verified, best per model). Claude shown as "
         "reference lines, via Claude Code (proprietary — size undisclosed).\n"
         "All entries: tbench.ai Terminal-Bench 2.0 leaderboard. "
         "Ornith: provisional in-house (k=1).",
         fontsize=8.5, color="#9aa3ad", va="bottom", linespacing=1.5)

fig.savefig("docs/tbench-size-vs-score.png")
print("wrote docs/tbench-size-vs-score.png")
