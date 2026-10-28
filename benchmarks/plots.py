"""Benchmark figures, drawn in a light and a dark variant for the documentation.

Library identity is colour (categorical slots 1-4, validated with the dataviz palette script)
plus marker shape, and every point is labelled, so colour is never the only cue. All plots use a
single axis scale per panel; the tables on the page carry the exact values.
"""

from __future__ import annotations

from pathlib import Path

import adapters
import data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, ScalarFormatter

plt.switch_backend("Agg")  # figures are written to files, never shown

LIBS = {  # name: (light colour, dark colour, marker)
    "fiducio": ("#2a78d6", "#3987e5", "o"),
    "netcal": ("#eb6834", "#d95926", "s"),
    "torch-uncertainty": ("#1baf7a", "#199e70", "^"),
    "dirichletcal": ("#eda100", "#c98500", "D"),
}
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#333331"),
}
METHODS = [m for m in adapters.ADAPTERS if m != "identity"]
LIB_OF = {m: cls.lib for m, (cls, _) in adapters.ADAPTERS.items()}
SCENARIO_TITLES = {
    "over": "over-confident",
    "under": "under-confident",
    "bias": "class bias",
    "rare": "rare class (~1%)",
}


def _colour(lib, mode):
    return LIBS[lib][0 if mode == "light" else 1]


def _setup(mode):
    t = THEMES[mode]
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["ink2"],
            "ytick.color": t["ink2"],
            "axes.edgecolor": t["grid"],
            "font.size": 10,
        }
    )
    return t


def _clean(ax, t, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _plain_log(axis, ticks):
    """Log axis with a few plain-number ticks (matplotlib's default 3x10^2 labels collide)."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def _legend_handles(mode):
    return [
        Line2D([], [], marker=m, color=_colour(lib, mode), linestyle="", markersize=9, label=lib)
        for lib, (_c, _d, m) in LIBS.items()
    ]


def calibration(ok, mode, path):
    """NLL excess over the oracle and ECE for 4 classes, one column per scenario."""
    t = _setup(mode)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), sharey=True)
    rows = [("NLL − oracle NLL (lower is better)", "nll"), ("ECE (lower is better)", "ece")]
    ypos = {m: i for i, m in enumerate(reversed(METHODS))}
    for j, scenario in enumerate(data.SCENARIOS):
        d = ok[(ok.scenario == scenario) & (ok.C == 4)]
        oracle = d[d.method == "oracle"].set_index("seed")
        for i, (label, col) in enumerate(rows):
            ax = axes[i, j]
            _clean(ax, t)
            for m in METHODS:
                g = d[d.method == m].set_index("seed")
                v = (g[col] - oracle[col]) if col == "nll" else g[col]
                if v.empty:
                    continue
                ci = 1.96 * v.std() / len(v) ** 0.5
                lib = LIB_OF[m]
                ax.errorbar(
                    v.mean(),
                    ypos[m],
                    xerr=ci,
                    fmt=LIBS[lib][2],
                    color=_colour(lib, mode),
                    markersize=7,
                    capsize=2,
                    linewidth=1.2,
                )
            if col == "ece":
                ax.axvline(oracle[col].mean(), color=t["ink2"], linestyle=":", linewidth=1)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if i == 0:
                ax.set_title(SCENARIO_TITLES[scenario], color=t["ink"], fontsize=11)
            if j == 0:
                ax.set_ylabel(label, color=t["ink2"], fontsize=9)
    axes[0, 0].set_yticks(list(ypos.values()), list(ypos))
    fig.suptitle(
        "Calibration quality, 4 classes (mean ± 95% CI over 10 seeds; dotted line: oracle ECE)",
        color=t["ink"],
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.legend(
        handles=_legend_handles(mode),
        loc="lower center",
        ncol=4,
        frameon=False,
        labelcolor=t["ink2"],
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scaling(sc, mode, path):
    """Fit time and peak memory from 100k to 1M voxels (log axes): hollow = 100k, filled = 1M."""
    t = _setup(mode)
    sc = sc[(sc.status == "ok") & (sc.method != "identity")]
    sizes = sorted(sc.n.unique())
    lo, hi = sizes[0], sizes[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    ypos = {m: i for i, m in enumerate(reversed(METHODS))}
    for ax, (col, label) in zip(
        axes,
        [("fit_s", "fit time (s, log scale)"), ("peak_rss_mb", "peak memory (MB, log scale)")],
        strict=True,
    ):
        _clean(ax, t)
        ax.set_xscale("log")
        for m in METHODS:
            a = sc[(sc.method == m) & (sc.n == lo)]
            b = sc[(sc.method == m) & (sc.n == hi)]
            if a.empty or b.empty:
                continue
            lib = LIB_OF[m]
            c, y = _colour(lib, mode), ypos[m]
            x0, x1 = a[col].iloc[0], b[col].iloc[0]
            ax.plot([x0, x1], [y, y], color=c, linewidth=1.5, zorder=1)
            ax.plot(
                x0,
                y,
                LIBS[lib][2],
                color=t["surface"],
                markeredgecolor=c,
                markeredgewidth=1.8,
                markersize=9,
                zorder=2,
            )
            ax.plot(
                x1,
                y,
                LIBS[lib][2],
                color=c,
                markeredgecolor=t["surface"],
                markeredgewidth=1.2,
                markersize=10,
                zorder=3,
            )
        ax.set_xlabel(label)
    _plain_log(axes[0].xaxis, [0.1, 1, 10])
    _plain_log(axes[1].xaxis, [300, 500, 1000, 2000, 5000])
    axes[0].set_yticks(list(ypos.values()), list(ypos))
    fig.suptitle(
        f"Scaling: hollow = {lo:,} voxels, filled = {hi:,} voxels (4 classes, 4 threads)",
        color=t["ink"],
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.legend(
        handles=_legend_handles(mode),
        loc="lower center",
        ncol=4,
        frameon=False,
        labelcolor=t["ink2"],
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def tradeoff(sc, mode, path):
    """Fit time against peak memory at the largest size; lower left is better."""
    t = _setup(mode)
    sc = sc[(sc.status == "ok") & (sc.method != "identity")]
    sc = sc[sc.n == sc.n.max()]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    _clean(ax, t, grid="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for lib, (_c, _d, marker) in LIBS.items():
        g = sc[sc.lib == lib]
        if g.empty:
            continue
        c = _colour(lib, mode)
        ax.plot(
            g.fit_s,
            g.peak_rss_mb,
            marker,
            color=c,
            markeredgecolor=t["surface"],
            markeredgewidth=1.2,
            markersize=11,
            linestyle="",
        )
        names = ", ".join(m.split(" ")[-1] for m in g.method)
        ax.annotate(
            f"{lib}\n{names}",
            (g.fit_s.mean(), g.peak_rss_mb.mean()),
            textcoords="offset points",
            xytext=(12, 8),
            color=t["ink"],
            fontsize=9,
        )
    ax.set_xlabel("fit time (s, log scale)")
    ax.set_ylabel("peak memory (MB, log scale)")
    ax.set_xlim(ax.get_xlim()[0] * 0.7, ax.get_xlim()[1] * 2.2)
    _plain_log(ax.xaxis, [0.5, 1, 2, 5, 10, 20])
    _plain_log(ax.yaxis, [400, 600, 1000, 2000, 5000])
    ax.set_title(
        f"Cost of fitting {int(sc.n.max()):,} voxels, 4 classes: lower left is better",
        color=t["ink"],
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_all(ok, sc, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        calibration(ok, mode, out_dir / f"calibration-{mode}.png")
        scaling(sc, mode, out_dir / f"scaling-{mode}.png")
        tradeoff(sc, mode, out_dir / f"tradeoff-{mode}.png")
