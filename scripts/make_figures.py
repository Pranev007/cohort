#!/usr/bin/env python
"""Generate the README figures.

Every number is read from the committed result JSONs — `benchmarks/baseline.json`
and `artifacts/reports/real_corpora.json` — so a chart can never drift from the
run that produced it. Regenerate after any evaluation change:

    python scripts/make_figures.py

Light and dark variants are emitted for each figure. GitHub honours
`prefers-color-scheme` inside a <picture> element, so the README stays legible in
both themes without shipping a washed-out compromise palette.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"

# Okabe-Ito, colour-blind safe. The peer/global contrast carries the whole story,
# so those two must stay distinguishable for deuteranopes.
PEER = "#0072B2"
GLOBAL = "#D55E00"
RANDOM = "#999999"
ACCENT = "#009E73"

THEMES = {
    "light": {"fg": "#24292f", "muted": "#57606a", "grid": "#d0d7de"},
    "dark": {"fg": "#e6edf3", "muted": "#8b949e", "grid": "#30363d"},
}


def _style(ax, theme: dict, ylabel: str = "") -> None:
    ax.set_facecolor("none")
    ax.figure.patch.set_alpha(0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9)
    ax.yaxis.label.set_color(theme["muted"])
    ax.xaxis.label.set_color(theme["muted"])
    ax.title.set_color(theme["fg"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _legend(ax, theme: dict, loc: str) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc=loc)
    for text in leg.get_texts():
        text.set_color(theme["muted"])


def _save(fig, name: str, variant: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}-{variant}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


def _arm(ablations: dict, needle: str) -> float:
    """Pull one ablation arm's PR-AUC, tolerating both report shapes."""
    for key, value in ablations.items():
        if needle in key.lower():
            return value["metrics"]["pr_auc"] if "metrics" in value else value["pr_auc"]
    raise KeyError(f"no ablation arm matching {needle!r}")


# --------------------------------------------------------------------------
def fig_peer_vs_global(synth: dict, enron: dict, variant: str) -> Path:
    """The headline: the method works on one corpus and not on the other."""
    theme = THEMES[variant]
    data = [
        (
            "Synthetic\n15k documents",
            synth["headline"]["pr_auc"],
            _arm(synth["ablations"], "global"),
            _arm(synth["ablations"], "random"),
        ),
        (
            "Enron\n23k real messages",
            enron["detection"]["pr_auc"],
            _arm(enron["ablations"], "global"),
            _arm(enron["ablations"], "random"),
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.26
    for i, (_label, peer, glob, rand) in enumerate(data):
        ax.bar(i - width, peer, width, color=PEER, label="Peer cohorts" if i == 0 else None)
        ax.bar(i, glob, width, color=GLOBAL, label="Global baseline" if i == 0 else None)
        ax.bar(i + width, rand, width, color=RANDOM, label="Random cohorts" if i == 0 else None)

        for x, value, bold in ((i - width, peer, True), (i, glob, False), (i + width, rand, False)):
            ax.text(
                x,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=8.5,
                color=theme["fg"],
                fontweight="bold" if bold else "normal",
            )

        ratio = peer / glob if glob else float("inf")
        beneficial = ratio > 1.15
        ax.text(
            i,
            max(peer, glob) + 0.075,
            f"{ratio:.1f}x better" if beneficial else "no benefit",
            ha="center",
            fontsize=10.5,
            color=ACCENT if beneficial else GLOBAL,
            fontweight="bold",
        )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=10, color=theme["fg"])
    ax.set_ylim(0, max(d[1] for d in data) * 1.34)
    _style(ax, theme, "PR-AUC (average precision)")
    ax.set_title(
        "Peer baselining beats a global baseline — on one corpus, not the other",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    _legend(ax, theme, "upper right")
    return _save(fig, "peer-vs-global", variant)


def fig_coupling(synth: dict, enron: dict, variant: str) -> Path:
    """Why: cohort membership predicts posture on one corpus and not the other."""
    theme = THEMES[variant]
    synth_map = synth["stats"]["coupling_by_feature"]
    enron_map = enron["notes"]["coupling_by_feature"]

    shared = sorted((f for f in enron_map if f in synth_map), key=lambda f: -synth_map[f])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rows = range(len(shared))
    height = 0.36
    ax.barh(
        [i - height / 2 for i in rows],
        [synth_map[f] for f in shared],
        height,
        color=PEER,
        label="Synthetic",
    )
    ax.barh(
        [i + height / 2 for i in rows],
        [enron_map[f] for f in shared],
        height,
        color=GLOBAL,
        label="Enron",
    )

    ax.set_yticks(list(rows))
    ax.set_yticklabels(shared, fontsize=8.5, color=theme["fg"], fontfamily="monospace")
    ax.invert_yaxis()
    _style(ax, theme)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.6, alpha=0.6)
    ax.set_xlabel(
        "Posture variation explained by cohort membership   (eta-squared / NMI)", fontsize=9
    )
    ax.set_title(
        "The precondition: does semantic category predict security posture?",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    _legend(ax, theme, "lower right")

    ax.axvline(0.10, color=theme["muted"], linestyle=":", linewidth=1)
    ax.text(
        0.108,
        len(shared) - 0.6,
        "below ~0.10 the peer\ndistribution equals the global one",
        fontsize=8,
        color=theme["muted"],
        va="bottom",
    )
    return _save(fig, "coupling", variant)


def fig_sensitivity(synth: dict, variant: str) -> Path:
    """Detection degrades gracefully as cohort quality is deliberately wrecked."""
    theme = THEMES[variant]
    rows = sorted(synth["cohort_quality_sensitivity"], key=lambda r: r["ari"])
    glob = _arm(synth["ablations"], "global")

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(
        [r["ari"] for r in rows],
        [r["pr_auc"] for r in rows],
        marker="o",
        color=PEER,
        linewidth=2,
        markersize=6,
        label="Peer cohorts",
    )
    for r in rows:
        ax.annotate(
            f"{r['n_cohorts']} cohorts",
            (r["ari"], r["pr_auc"]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=7.5,
            color=theme["muted"],
        )

    ax.axhline(glob, color=GLOBAL, linestyle="--", linewidth=1.5, label="Global baseline")
    ax.set_xlabel("Cohort quality (Adjusted Rand Index vs true categories)", fontsize=9)
    _style(ax, theme, "PR-AUC")
    ax.set_ylim(0, max(r["pr_auc"] for r in rows) * 1.28)
    ax.set_title(
        "Detection degrades gracefully as clustering is wrecked",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )
    _legend(ax, theme, "lower right")
    return _save(fig, "sensitivity", variant)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "benchmarks" / "baseline.json")
    ap.add_argument(
        "--real", type=Path, default=ROOT / "artifacts" / "reports" / "real_corpora.json"
    )
    args = ap.parse_args()

    synth = json.loads(args.baseline.read_text(encoding="utf-8"))
    real = json.loads(args.real.read_text(encoding="utf-8"))
    enron = next(r for r in real if "Enron" in r["corpus"])

    written: list[Path] = []
    for variant in THEMES:
        written.append(fig_peer_vs_global(synth, enron, variant))
        written.append(fig_coupling(synth, enron, variant))
        written.append(fig_sensitivity(synth, variant))

    for path in written:
        print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    print(f"\n{len(written)} figures written to {ASSETS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
