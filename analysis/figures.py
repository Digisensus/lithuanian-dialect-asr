import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
R = json.load(open(ROOT / "analysis" / "results.json"))
try:
    from .publication_data import table as publication_table, scores as publication_scores, customer_results
except ImportError:
    from publication_data import table as publication_table, scores as publication_scores, customer_results
T = publication_table()
HOURS = {r["recipe_id"]: {p["code"]: p["hours"] for p in r["parts"]}
         for r in (json.load(open(p)) for p in sorted((ROOT / "recipes").glob("*.json")))}
REGION = {"A": "Aukštaitija", "D": "Dzūkija", "Z": "Žemaitija", "S": "Suvalkija"}

GREY, BLUE, RED, GREEN, ORANGE, PINK = "#7a7a7a", "#0072B2", "#D55E00", "#009E73", "#E69F00", "#F4A582"
COL1, COL2 = 3.4, 7.0
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})


def seeds(recipe, test, view=None):
    out = []
    for s in (1, 2):
        row = T.get(f"{recipe}/s{s}")
        if row:
            v = row[test]
            out.append(v[view] if isinstance(v, dict) else v)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def added_hours(recipe):
    return sum(h for code, h in HOURS[recipe].items() if code != "S")


def _walk(o):
    if isinstance(o, dict):
        if {"diff", "ci95", "a", "b"} <= o.keys():
            yield o
        else:
            for v in o.values():
                yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


PAIRED = {(p["a"], p["b"], p["test"], p["view"], p.get("region")): p for p in _walk(R)}


def paired(a, b, test, view="dialect", region=None):
    return PAIRED.get((a, b, test, view, region))


def seed_sd(test, view="dialect"):
    return R["seed_noise"][f"{test}:{view}"]["pooled_seed_sd"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"paper/figures/{name}.pdf")


def fig_dose():
    recs = ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial"]
    xs = [added_hours(r) for r in recs]
    customer = customer_results()
    panels = [("dial.test", "dialect", "Dialect test\nDialect spelling"),
              ("dial.test", "either", "Dialect test\nEither spelling accepted"),
              ("phone.test", None, "LIEPA-3 telephone test\nStandard speech"),
              ("customer", None, "Customer service speech\nIndependent test")]
    fig, grid = plt.subplots(2, 2, figsize=(COL2, 4.9), layout="constrained")
    axes = grid.ravel()
    for ax, (test, view, title) in zip(axes, panels):
        if test == "customer":
            ys = [customer["seed_wer"][next(k for k in customer["seed_wer"] if k.startswith(f"ltd26/{r}/s1/"))]["all"]
                  if r in ("S", "S+D80.dial") else customer["wer"][r]["all"]["wer"] for r in recs]
        else:
            ys = [T[f"{r}/s1"][test][view] if view else T[f"{r}/s1"][test] for r in recs]
        ax.plot(xs, ys, "-o", color=BLUE, ms=4)
        for j in (0, len(xs)-1):
            ax.annotate(f"{ys[j]:.2f}%", (xs[j], ys[j]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.set_title(title, loc="left")
        ax.set_xticks(xs, [f"{x:.0f}" for x in xs])
        ax.set_xlim(-9, max(xs)+9)
        ax.set_xlabel("Dialect training hours added")
        ax.set_ylabel("WER (%)")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    for ax in [axes[0], axes[1], axes[3]]:
        ax.set_ylim(25, 45)
    axes[2].set_ylim(9.5, 12)
    axes[2].text(0.5, 0.94, "Expanded vertical scale", transform=axes[2].transAxes,
                 ha="center", va="top", fontsize=6.5, color=GREY)
    save(fig, "fig1_dose")


def fig_spelling_dose():
    groups = [("Dialect-spelling transcripts", BLUE, "o", ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial"]),
              ("Standard-spelling transcripts", RED, "s", ["S", "S+D25.std", "S+D50.std", "S+D80.std"])]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), layout="constrained")
    for ax, (test, view, title) in zip(axes, [("dial.test", "either", "Dialect test: either spelling accepted"),
                                             ("phone.test", None, "Telephone test: standard speech")]):
        for label, color, marker, recs in groups:
            xs = [added_hours(r) for r in recs]
            ax.plot(xs, [mean(seeds(r, test, view)) for r in recs], marker=marker, color=color, label=label)
            for x, r in zip(xs, recs):
                sv = seeds(r, test, view)
                if len(sv) > 1:
                    ax.plot([x, x], [min(sv), max(sv)], color=color, lw=3, alpha=.3)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Dialect training hours added")
        ax.set_ylabel("WER (%)")
        ax.set_xticks([added_hours(r) for r in groups[0][3]], ["0", "13", "25", "50", "82"])
        ax.grid(axis="y", lw=.3, alpha=.5)
    axes[0].set_ylim(24, 36)
    axes[1].set_ylim(9.5, 12)
    axes[1].text(.5,.95,"Expanded vertical scale",transform=axes[1].transAxes,ha="center",va="top",fontsize=6.5,color=GREY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    save(fig, "fig_spelling_dose")


ROWS = [
    ("RQ1", "S+D80.dial", "S", "dialect", True),
    ("RQ2", "S+D80.dial", "S+X80", "dialect", True),
    ("RQ2", "S+D80.dial", "S+X80", "either", False),
    ("RQ2", "S+X80", "S", "dialect", True),
    ("RQ3", "S+D80.dial", "S", "standard", False),
    ("RQ3", "S+D80.dial", "S", "either", False),
    ("RQ3", "S+D80.std", "S", "either", False),
    ("RQ3", "S+D80.std", "S+D80.dial", "either", True),
]
VIEW = {"dialect": "dialect refs", "standard": "standard refs", "either": "either form"}


def fig_contrasts():
    fig, (axg, ax1, ax2) = plt.subplots(1, 3, figsize=(COL2, 3.0), layout="constrained",
                                        gridspec_kw={"width_ratios": [0.07, 1.7, 1]})
    n = len(ROWS)
    for i, (grp, a, b, view, phone) in enumerate(ROWS):
        y = n - 1 - i
        for ax, test, v, show in ((ax1, "dial.test.v1", view, True), (ax2, "phone.test.v1", "dialect", phone)):
            if not show:
                continue
            for k, (dy, mk, fc) in ((1, (0.14, "o", "k")), (2, (-0.14, "s", "white"))):
                p = paired(f"{a}/s{k}", f"{b}/s{k}", test, v)
                if p is None:
                    continue
                lo, hi = p["ci95"]
                ax.errorbar(p["diff"], y + dy, xerr=[[p["diff"] - lo], [hi - p["diff"]]], fmt=mk, ms=3.6,
                            color="k", mfc=fc, mew=0.8, elinewidth=0.9, capsize=1.6)
    labels = [f"{a} − {b}   [{VIEW[view]}]" for _g, a, b, view, _p in ROWS]
    ax1.set_yticks(range(n - 1, -1, -1), labels)
    ax2.tick_params(labelleft=False)
    axg.axis("off")
    for ax in (axg, ax1, ax2):
        ax.set_ylim(-0.6, n - 0.4)
    for ax in (ax1, ax2):
        ax.axvline(0, color=GREY, lw=0.7)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    prev = None
    for i, (grp, *_rest) in enumerate(ROWS):
        y = n - 1 - i
        if grp != prev:
            axg.text(0, y, grp, va="center", ha="left", fontsize=7.5, fontweight="bold")
            if prev is not None:
                for ax in (ax1, ax2):
                    ax.axhline(y + 0.5, color=GREY, lw=0.4, ls=":")
        prev = grp
    ax1.set_title("(a) Dialect test (12 unseen speakers)", loc="left")
    ax2.set_title("(b) Phone test", loc="left")
    ax1.set_xlabel("WER difference A − B (percentage points)")
    ax2.set_xlabel("WER difference A − B\n(percentage points)")
    handles = [Line2D([], [], color="k", marker="o", ls="", ms=3.6, label="seed 1"),
               Line2D([], [], color="k", marker="s", mfc="white", ls="", ms=3.6, label="seed 2"),
               Line2D([], [], color="k", lw=0.9, label="95 % paired bootstrap CI (10,000 resamples of speakers; of clips for the phone test)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "fig2_contrasts")


def fig_spelling():
    recs = [("S", GREY, "No dialect training data"), ("S+D80.dial", BLUE, "Dialect-spelling training transcripts"),
            ("S+D80.std", RED, "Standard-spelling training transcripts")]
    views = [("dialect", "Dialect spelling\naccepted"), ("standard", "Standard spelling\naccepted"),
             ("either", "Either spelling\naccepted")]
    fig, ax = plt.subplots(figsize=(COL2, 3.15), layout="constrained")
    w = 0.26
    for j, (rec, col, _lab) in enumerate(recs):
        for i, (view, _vl) in enumerate(views):
            sv = seeds(rec, "dial.test", view)
            x = i + (j - 1) * w
            ax.bar(x, mean(sv), w * 0.92, color=col, alpha=0.85, zorder=2)
            ax.plot([x] * len(sv), sv, "o", color="k", ms=2, zorder=3)
            ax.text(x, max(sv) + 0.6, f"{mean(sv):.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(views)), [vl for _v, vl in views])
    ax.set_ylabel("WER on dialect test (%)")
    ax.set_ylim(0, max(max(seeds(r, "dial.test", v)) for r, *_ in recs for v, _ in views) * 1.12)
    ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
    handles = [Patch(color=c, alpha=0.85, label=lab) for _r, c, lab in recs]
    handles.append(Line2D([], [], color="k", marker="o", ls="", ms=2, label="Individual runs (bars: mean of two)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7.5, columnspacing=1.0)
    save(fig, "fig3_spelling")


def fig_regions():
    regs = sorted(R["RQ4_region"])
    bars = [("no_dialect_S", GREY, "No dialect training"),
            ("left_out", BLUE, "Other regions only"),
            ("all_regions_D80", RED, "All four regions")]
    fig, ax = plt.subplots(figsize=(COL2, 2.55), layout="constrained")
    w = 0.26
    for j, (key, color, _label) in enumerate(bars):
        for i, region in enumerate(regs):
            value = R["RQ4_region"][region][key]
            x = i + (j - 1) * w
            ax.bar(x, value, w * .92, color=color, alpha=.85, zorder=2)
            ax.text(x, value + .6, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(regs)), [REGION[r] for r in regs])
    ax.set_xlabel("Region being tested")
    ax.set_ylabel("WER: either spelling accepted (%)")
    ax.set_ylim(0, max(R["RQ4_region"][r]["no_dialect_S"] for r in regs) * 1.15)
    ax.grid(axis="y", lw=.3, alpha=.5, zorder=0)
    fig.legend(handles=[Patch(color=color, alpha=.85, label=label) for _, color, label in bars],
               loc="outside lower center", ncol=3, fontsize=8)
    save(fig, "fig4_regions")


def fig_seen():
    points = []
    for model, row in T.items():
        if model.startswith("B/"):
            continue
        recipe = model.rsplit("/s", 1)[0]
        color = BLUE if recipe.endswith(".dial") else RED if recipe.endswith(".std") else GREY
        points.append((row["dial.test"]["either"], row["dial.seen"]["either"], color, recipe.startswith("D80")))
    fig, ax = plt.subplots(figsize=(5.7, 4.4), layout="constrained")
    for x, y, color, dialect_only in points:
        ax.plot(x, y, "o", color=color, mfc="white" if dialect_only else color, mew=1, ms=5, zorder=3)
    lo = min(min(point[0], point[1]) for point in points) - 1.5
    hi = max(max(point[0], point[1]) for point in points) + 1.5
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=.9, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    above = sum(y > x for x, y, *_ in points)
    ax.text(.97, .05, f"{above} of {len(points)} training runs above the diagonal:\nhigher WER on the training-pool speaker set",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.text(.04, .96, "Dashed diagonal: equal WER", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, color=GREY)
    ax.set_xlabel("WER on speakers absent from training (%)")
    ax.set_ylabel("WER on withheld recordings from\nthe dialect training pool's speakers (%)")
    ax.grid(lw=.3, alpha=.5)
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", label="Dialect-spelling training transcripts"),
               Line2D([], [], color=RED, marker="o", ls="", label="Standard-spelling training transcripts"),
               Line2D([], [], color=GREY, marker="o", ls="", label="No dialect training data"),
               Line2D([], [], color="k", marker="o", mfc="white", ls="", label="Hollow points: dialect-only training")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=7, columnspacing=1)
    save(fig, "fig5_seen")


def fig_benchmarks():
    recs = ["S", "S+D12.dial", "S+D25.dial", "S+D50.dial", "S+D80.dial", "S+X80", "S+D25.std", "S+D50.std",
            "S+D80.std", "S+D80-A.std", "S+D80-D.std", "S+D80-S.std", "S+D80-Z.std", "D80.dial", "D80.std"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True, layout="constrained")
    for ax, (test, title) in zip(axes, [("fleurs.lt.test", "(a) FLEURS lt test"),
                                        ("cv.lt.test", "(b) Common Voice 19.0 lt test")]):
        for i, rec in enumerate(recs):
            col = BLUE if rec.endswith(".dial") else RED if rec.endswith(".std") else GREY
            sv = seeds(rec, test)
            if len(sv) > 1:
                ax.plot([i, i], sv, "-", color=col, lw=0.8, zorder=2)
            ax.plot(i, sv[0], "o", color=col, ms=3.6, zorder=3)
            if len(sv) > 1:
                ax.plot(i, sv[1], "s", color=col, mfc="none", mew=0.9, ms=4.6, zorder=4)
        for b, ls, lab in (("B/parakeet-tdt-0.6b-v3", "--", "stock parakeet-tdt-0.6b-v3"),
                           ("B/whisper-large-v3", ":", "Whisper large-v3")):
            ax.axhline(T[b][test], color="k", ls=ls, lw=0.8, label=lab)
        ax.set_xticks(range(len(recs)), recs, rotation=60, ha="right", rotation_mode="anchor")
        ax.set_title(title, loc="left")
        ax.text(0.99, 0.97, f"seed sd {seed_sd(test):.2f} pp", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=GREY)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("WER (%)")
    axes[0].set_ylim(bottom=0)
    handles = [Line2D([], [], color=GREY, marker="o", ls="", ms=3.6, label="seed 1"),
               Line2D([], [], color=GREY, marker="s", mfc="none", ls="", ms=4.6, label="seed 2"),
               Line2D([], [], color="k", ls="--", lw=0.8, label="stock parakeet-tdt-0.6b-v3"),
               Line2D([], [], color="k", ls=":", lw=0.8, label="Whisper large-v3")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    save(fig, "figS1_benchmarks")


if __name__ == "__main__":
    fig_dose()
    fig_spelling_dose()
    fig_contrasts()
    fig_spelling()
    fig_regions()
    fig_seen()
    fig_benchmarks()
