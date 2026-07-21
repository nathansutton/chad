"""Generate docs/tbench-size-vs-score.png — the Terminal-Bench 2.1 cost-vs-accuracy chart.

Run:  uv run --with matplotlib python docs/tbench_chart.py

The story TB2.1 tells is cost, not size: every verified entry is a proprietary
frontier model whose parameter count is undisclosed, so there is nothing to plot on a
size axis — but every entry's dollar cost for a full run IS published. chad is the only
point on the board with no API cost at all (a local MLX/llama.cpp model on an Apple
Silicon laptop) and it clears 57% while the paid field runs $130–$2,000 per run.

Data provenance (update the tables below, then re-run):
- All 17 frontier entries: tbench.ai Terminal-Bench 2.1 leaderboard (verified, k=5),
  each as (cost_usd, accuracy_%). Accuracy and cost are read straight off the board.
- "Claude Code" rows are emphasized in ink (the harness-vs-harness reference thread);
  every other agent is a light background point.
- chad point: provisional in-house run (k=1, self-run, unverified) — 51/89 = 57.3%.
  Cost is electricity only (no API/token cost); it is placed in the far-left "local"
  band at a nominal position, NOT a measured dollar figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

GRAY = "#c2c9d1"       # non-Claude frontier entries (background field)
GRAY_LBL = "#8b95a1"
INK = "#1f2d3d"        # Claude Code entries (emphasized)
ORANGE = "#e8590c"     # chad
BAND = "#fdf1e3"

# TB2.1 leaderboard, verified k=5. (cost_usd, accuracy_%, label, is_claude_code)
# label=None → plotted but unlabeled (keeps the dense field readable).
ENTRIES = [
    (552.67, 83.8, "Fable 5",   True),
    (2059.19, 83.1, "GPT-5.5",  False),
    (438.64, 80.4, None,        False),  # Terminus 2 · Fable 5
    (134.09, 79.3, "Grok 4.5",  False),  # cheapest paid entry
    (286.94, 78.9, "Opus 4.8",  True),
    (421.15, 78.4, None,        False),  # Codex · GPT-5.6 Terra
    (493.85, 78.0, None,        False),  # Terminus 2 · GPT-5.5
    (198.05, 76.2, None,        False),  # mini-SWE-agent · Muse Spark 1.1
    (241.45, 75.7, None,        False),  # Codex · GPT-5.6 Luna
    (288.18, 74.6, "Sonnet 5",  True),
    (224.44, 73.9, None,        False),  # Terminus 2 · Gemini 3 Pro
    (599.52, 68.9, "Opus 4.7",  True),
    (582.26, 66.1, None,        False),  # Terminus 2 · Opus 4.7
    (247.76, 65.8, None,        False),  # Gemini CLI · Gemini 3 Pro
    (236.49, 65.8, None,        False),  # Gemini CLI · Gemini 3.1 Pro
    (229.99, 65.6, None,        False),  # Terminus 2 · Gemini 3.1 Pro
    (277.14, 58.7, "GLM-5.1",   True),   # the paid floor
]

# per-label placement: (dx_pts, dy_pts, ha, va) — hand-tuned to clear the cluster.
LABEL_POS = {
    "Fable 5":   (0, 13, "center", "bottom"),
    "GPT-5.5":   (0, -14, "center", "top"),
    "Grok 4.5":  (-12, 0, "right", "center"),   # into the open mid-left space
    "Opus 4.8":  (12, 2, "left", "center"),
    "Sonnet 5":  (12, 0, "left", "center"),
    "Opus 4.7":  (12, 0, "left", "center"),
    "GLM-5.1":   (12, 0, "left", "center"),
}

# chad — provisional (k=1, self-run). No API cost; nominal x inside the "local" band.
CHAD_X, CHAD_Y = 2.4, 57.3

fig, ax = plt.subplots(figsize=(11.0, 6.7), dpi=200)
fig.subplots_adjust(left=0.072, right=0.975, top=0.79, bottom=0.195)

ax.set_xscale("log")
ax.set_xlim(1.2, 4600)
ax.set_ylim(52, 88)

# "local · no API cost" band at the far left, where chad lives alone.
ax.axvspan(1.2, 9, color=BAND, zorder=0)
ax.text(1.5, 86.4, "local · no API cost", color="#c2410c", fontsize=10.5)

# cost-gap annotation: chad's electricity vs the cheapest PAID entry ($134).
ax.annotate("", xy=(115, 55.0), xytext=(CHAD_X + 0.7, 55.0),
            arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.3))
ax.text(19, 54.0, "every paid entry: \\$130–\\$2,000 / run", color="#c2410c",
        fontsize=10.5, ha="center", style="italic")

# inline legend (top-left open space, above the cluster)
ax.plot(11, 87.0, "o", ms=8, color=INK, mec="white", mew=1.0)
ax.text(13.5, 87.0, "Claude Code", color=INK, fontsize=10, va="center")
ax.plot(11, 84.9, "o", ms=7, color=GRAY, mec="none")
ax.text(13.5, 84.9, "other agents", color=GRAY_LBL, fontsize=10, va="center")

# frontier field
for cost, acc, label, claude in ENTRIES:
    if claude:
        ax.plot(cost, acc, "o", ms=8.5, color=INK, mec="white", mew=1.0, zorder=4)
    else:
        ax.plot(cost, acc, "o", ms=7, color=GRAY, mec="none", zorder=3)
    if label:
        dx, dy, ha, va = LABEL_POS[label]
        col = INK if claude else GRAY_LBL
        ax.annotate(label, (cost, acc), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, color=col, fontsize=11 if claude else 10.5)

# chad
ax.plot(CHAD_X, CHAD_Y, "o", ms=17, color=ORANGE, mec="white", mew=1.6, zorder=6)
ax.annotate("chad + Ornith 35B\n57%  ·  a laptop + electricity", (CHAD_X, CHAD_Y),
            textcoords="offset points", xytext=(17, 5), ha="left", va="center",
            color=ORANGE, fontsize=13, fontweight="bold", zorder=6)

# axes cosmetics
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["\\$10", "\\$100", "\\$1,000"], fontsize=12)
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(range(55, 86, 5))
ax.set_yticklabels(["55", "60", "65", "70", "75", "80", "85%"], fontsize=12)
ax.grid(axis="y", color="#e5e7eb", lw=0.8, zorder=1)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d1d5db")
ax.tick_params(length=0)
ax.set_xlabel("Cost per full benchmark run  (log scale, USD)", fontsize=13,
              color=INK, labelpad=7)
ax.set_ylabel("Terminal-Bench 2.1 accuracy", fontsize=13, color=INK)

fig.text(0.055, 0.925, "Frontier scores, laptop cost", fontsize=21, color=INK)
fig.text(0.072, 0.825,
         "On Terminal-Bench 2.1 every verified entry is a proprietary model in a "
         "datacenter, costing\n\\$130–\\$2,000 per run. chad + Ornith (a 35B MoE) clears "
         "57% on an Apple Silicon laptop — for the electricity.",
         fontsize=12.5, color="#4b5563", va="bottom", linespacing=1.4)
fig.text(0.055, 0.02,
         "Frontier entries: tbench.ai Terminal-Bench 2.1 leaderboard (verified, k=5), "
         "plotted at their published run cost. Claude Code rows in ink.\n"
         "chad: provisional in-house run (k=1, self-run, unverified) — 51/89. No API "
         "cost; placed in the local band at a nominal position (electricity only).",
         fontsize=8.5, color="#9aa3ad", va="bottom", linespacing=1.5)

fig.savefig("docs/tbench-size-vs-score.png")
print("wrote docs/tbench-size-vs-score.png")
