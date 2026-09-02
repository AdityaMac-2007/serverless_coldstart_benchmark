"""
Fixes Figure 1 (violin/box/strip) and Figure 3 (timeseries), and makes the
whole pipeline scale cleanly as more cycles/rows get appended to the CSVs.
Figure 2 (bar chart) is unchanged — it's already good.

ROUND 1 — fixed by rendering the previous version, not just reading it:

Figure 1
  • Sharing one y-range per row (`_share_ylim`) is right in principle, but
    it silently absorbed the default [0, 1] limits of any axis that hit the
    `vals.empty` early-return, because that axis was still appended to
    `cold_axes`/`warm_axes` before the `continue`. One provider with a
    missing phase in one cycle corrupts the axis range for every other
    provider in that row. Fixed by appending to the shared-axis list only
    after data is confirmed non-empty.
  • The median annotation's `xytext` offset was computed correctly for the
    *sample it was drawn on*, but `_share_ylim` runs after every axis is
    built and stretches all row axes to a common range — so the offset,
    computed against each axis's own pre-share y-span, ends up wrong on
    every axis except whichever one happened to set the row's max range.
    Fixed by computing the annotation offset from the shared row range,
    not the per-axis range, which means annotating after sharing.
  • Jitter used a fixed-seed RNG re-instantiated inside the innermost loop.
    Same seed + a length that changes as rows are appended in the future
    means points don't move because the *data* moved, they move because
    the RNG stream realigns differently — jitter looks unstable across
    reruns for no real reason. Fixed with one Generator instance for the
    whole figure.

Figure 3
  • Cold start (~400–1200 ms) and warm ping (~5–25 ms) were drawn on one
    linear y-axis per panel. At that ratio the warm line and its ±1σ band
    are visually a flat line at y≈0 — completely unreadable, confirmed by
    rendering it. This is the exact bug the Figure-1 fix explicitly solved
    by splitting rows; it just wasn't carried over to Figure 3. Fixed with
    a twin y-axis per panel (cold on the left in red, warm on the right in
    blue), so both series use their natural range.
  • `ax.set_xticks(xs)` puts one tick per cycle. Fine at 10 cycles, but the
    ask is "more data in the future" — at 40+ cycles this becomes label
    spam. Fixed with a MaxNLocator that caps ticks and always includes the
    first/last cycle.
  • Legend was only ever attached to `providers[0]`'s axis. If that
    provider's CSV is ever absent, no legend renders anywhere in the
    figure. Fixed with a single shared legend built from one panel's
    handles (cold + warm) but placed at the figure level so it doesn't
    depend on which provider happens to be first.

Scaling for future data (the actual ask)
  • `load_data` re-read and re-cleaned every CSV from scratch on every run
    with no way to skip unchanged files. Added an on-disk cache keyed by
    each file's mtime+size, so re-running after only one CSV changed
    doesn't re-parse the other two.
  • IQR outlier filtering, IQR was computed once per (provider, phase) over
    the whole history. That's correct statistically (it's supposed to be
    global), but it means every append re-filters everything from cycle 1
    — which is fine for correctness but was being done with a Python-level
    groupby + boolean-index loop. Replaced with a vectorized transform,
    same result, no behavior change, just doesn't get slower per-group as
    rows grow.
  • Both figures previously assumed exactly the 3 known providers via a
    hardcoded `DATA_FILES = {display_name: filename}` dict — a genuinely
    new provider's CSV would never even be read, let alone hit a palette
    KeyError, because nothing pointed at it. Replaced with
    `discover_data_files()`, which still gives the 3 known filenames their
    curated display names but also globs for any other `*_latency_data.csv`
    in the directory, deriving a display name from the filename. A 4th
    provider now Just Works by dropping a correctly-named CSV next to the
    others — no code change. Its color comes from `_FALLBACK_CYCLE` since
    it isn't in `PROVIDER_PALETTE`.

ROUND 2 — found by actually running this against the real
aws_lambda_latency_data.csv / gcp_latency_data.csv / azure_latency_data.csv,
by inspecting the raw rows chronologically rather than trusting the column
schema:

  • `Cycle` was parsed from a number embedded in the Phase text
    ("Cycle_2_Warm_1" -> 2). That number turns out to be a *local* loop
    counter that resets to 1 every time the benchmark script restarts a
    run — the raw data shows "Cycle_1" recurring across a dozen-plus
    separate runs spread over a full week. Grouping by that label in
    fig_timeseries averaged together measurements from completely
    different points in time under one x-axis tick, which defeats the
    figure's actual purpose (stability *across* cycles). Fixed by dropping
    the embedded number and assigning a real chronological cycle index per
    provider instead: each Cold Start row starts a new cycle
    (`(Phase == "Cold Start").cumsum()`), and every Warm Ping is
    attributed to the most recent preceding Cold Start. Added
    `_CLEAN_LOGIC_VERSION` to the cache key so this change (and any future
    change to the cleaning logic) invalidates old cached .pkl files
    instead of silently continuing to serve rows grouped the old way just
    because the source CSV itself hasn't changed.
  • A provider whose CSV is found but has zero rows survive cleaning —
    here, AWS Lambda, which returns HTTP 403 on every single request in
    this dataset, so the 2xx-only filter drops 100% of it — never became a
    category at all, so it silently vanished from both figures with no
    visual trace, indistinguishable from "only 2 providers were ever
    benchmarked." Fixed by keeping every *discovered* provider (i.e. every
    provider whose CSV file exists) as a category regardless of whether
    any rows survived cleaning. fig_violin already had a graceful "no
    data" placeholder per phase — it just never got the chance to use it,
    because the provider was being filtered out one level up. fig_timeseries
    gets the same treatment, plus a guard so an empty provider renders a
    "no data" panel instead of crashing on xs.min()/max() over an empty
    array.

  Not a code fix, but worth flagging: AWS Lambda returning 403 on every
  request for the entire monitoring window looks like a real endpoint or
  auth problem (Function URL auth type, a missing resource policy, or a
  WAF/API Gateway rule) rather than noise. This pipeline has no way to
  tell "AWS is fine but the benchmark's auth is misconfigured" apart from
  "AWS is genuinely failing" — worth checking independently of anything
  here.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["TeX Gyre Pagella", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "legend.title_fontsize": 7.5,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "figure.constrained_layout.use": False,  # we lay out by hand (gridspec rects)
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.7,
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.linewidth":     0.4,
    "grid.alpha":         0.45,
    "grid.color":         "#aaaaaa",
    "axes.axisbelow":     True,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "lines.linewidth":    1.4,
    "patch.linewidth":    0.5,
    "legend.frameon":     True,
    "legend.framealpha":  0.93,
    "legend.edgecolor":   "#cccccc",
})

PROVIDER_PALETTE = {
    "AWS Lambda":      "#E07B39",
    "GCP Cloud Run":   "#2E6DB4",
    "Azure Functions": "#2A9D60",
}
# Fallback so an unrecognized provider (new CSV added later) gets a stable
# color instead of a KeyError.
_FALLBACK_CYCLE = ["#7B5EA7", "#B4652E", "#4C8577", "#A34F6B"]

PHASE_COLORS = {
    "Cold Start": "#C0392B",
    "Warm Ping":  "#2471A3",
}
PROVIDERS_ORDER = ["AWS Lambda", "GCP Cloud Run", "Azure Functions"]

# Known filename -> display name for the providers we've always tracked.
# Kept explicit (rather than derived from the filename) because "AWS Lambda"
# isn't recoverable from "aws_lambda_latency_data.csv" without guessing.
KNOWN_DATA_FILES = {
    "aws_lambda_latency_data.csv": "AWS Lambda",
    "gcp_latency_data.csv":        "GCP Cloud Run",
    "azure_latency_data.csv":      "Azure Functions",
}
# Any other *_latency_data.csv dropped in the same directory is picked up
# automatically (see discover_data_files) so a new provider doesn't require
# a code change here — its display name is derived from the filename.
CUTOFF_DATE = pd.Timestamp("2026-09-01")

# Bumped whenever _clean_one's cleaning/derivation logic changes materially,
# so the on-disk cache (keyed on each raw file's mtime+size) can't silently
# keep serving rows produced by an older version of the logic just because
# the source CSV itself happens not to have changed.
_CLEAN_LOGIC_VERSION = "2"

_CACHE_DIR = Path(".cache_latency")


def _provider_color(provider: str, idx: int) -> str:
    if provider in PROVIDER_PALETTE:
        return PROVIDER_PALETTE[provider]
    return _FALLBACK_CYCLE[idx % len(_FALLBACK_CYCLE)]


def _display_name_from_filename(filename: str) -> str:
    """'digitalocean_latency_data.csv' -> 'Digitalocean'. Best-effort only —
    used solely for providers not in KNOWN_DATA_FILES."""
    stem = filename.removesuffix("_latency_data.csv").removesuffix(".csv")
    return stem.replace("_", " ").strip().title()


def discover_data_files(data_dir: Path) -> dict[str, str]:
    """
    Returns {display_name: filename} for every provider CSV present.
    Known filenames (KNOWN_DATA_FILES) always get their curated display
    name. Any other file matching *_latency_data.csv is picked up too, so
    adding a 4th provider later means dropping a CSV next to the others —
    no code change required here.
    """
    files: dict[str, str] = {}
    for filename, display in KNOWN_DATA_FILES.items():
        if (data_dir / filename).exists():
            files[display] = filename

    for fpath in sorted(data_dir.glob("*_latency_data.csv")):
        if fpath.name in KNOWN_DATA_FILES:
            continue  # already handled above with its curated name
        files[_display_name_from_filename(fpath.name)] = fpath.name

    return files


# ── Data loader ──────────────────────────────────────────────────────────────
def _file_fingerprint(fpath: Path) -> str:
    """Cheap change-detector: mtime + size, no need to hash file contents."""
    st = fpath.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def _clean_one(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    df = df.copy()
    df["Provider"] = provider
    df = df[df["Timestamp"] >= CUTOFF_DATE].copy()
    df["Latency_ms"] = pd.to_numeric(df["Latency_ms"], errors="coerce")
    df.dropna(subset=["Latency_ms"], inplace=True)
    df = df[(df["Latency_ms"] > 0) &
            df["Status_Code"].astype(str).str.startswith("2")].copy()

    def cat(raw):
        r = raw.lower()
        if "cold" in r:
            return "Cold Start"
        if "warm" in r:
            return "Warm Ping"
        return None

    df["Phase"] = df["Phase"].astype(str).apply(cat)
    df.dropna(subset=["Phase"], inplace=True)

    # `Cycle` used to come from a number embedded in the Phase text (e.g.
    # "Cycle_2_Warm_1" -> 2). Verified against the raw data: that number is
    # a *local* loop counter that resets to 1 every time the benchmark
    # script restarts a run, so "Cycle_1" recurs across many separate runs
    # spanning the whole monitoring window — grouping by it would average
    # together measurements from very different points in time under one
    # x-axis tick. Instead, each Cold Start row starts a new cycle,
    # chronologically, and every Warm Ping is attributed to the most
    # recent preceding Cold Start.
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Cycle"] = (df["Phase"] == "Cold Start").cumsum()

    df["WarmSlot"] = (df[df["Phase"] == "Warm Ping"]
                       .groupby("Cycle").cumcount() + 1)

    # IQR outlier removal, vectorized via groupby-transform instead of a
    # per-group Python loop with boolean-index writes. Same semantics
    # (global IQR per provider+phase across all cycles seen so far), just
    # doesn't slow down as more (provider, phase) groups appear.
    grp = df.groupby(["Provider", "Phase"])["Latency_ms"]
    q1 = grp.transform(lambda v: v.quantile(0.25))
    q3 = grp.transform(lambda v: v.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - 2.5 * iqr, q3 + 2.5 * iqr
    df = df[(df["Latency_ms"] >= lo) & (df["Latency_ms"] <= hi)].copy()
    return df


def load_data(data_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    """
    Loads + cleans all provider CSVs found by discover_data_files (known
    providers plus any new *_latency_data.csv dropped in the same
    directory). Each file is cached individually (keyed on mtime+size plus
    _CLEAN_LOGIC_VERSION) so appending rows to one CSV — or changing the
    cleaning logic — doesn't force a re-parse of files that didn't need it.
    Delete `.cache_latency/` to force a full rebuild regardless.
    """
    if use_cache:
        _CACHE_DIR.mkdir(exist_ok=True)

    data_files = discover_data_files(data_dir)
    frames = []
    for provider, filename in data_files.items():
        fpath = data_dir / filename
        if not fpath.exists():
            continue

        if use_cache:
            fp = _file_fingerprint(fpath)
            cache_key = hashlib.sha1(
                f"{_CLEAN_LOGIC_VERSION}:{filename}:{fp}".encode()
            ).hexdigest()[:16]
            cache_file = _CACHE_DIR / f"{cache_key}.pkl"
            if cache_file.exists():
                frames.append(pd.read_pickle(cache_file))
                continue

        raw = pd.read_csv(fpath, parse_dates=["Timestamp"])
        cleaned = _clean_one(raw, provider)

        if use_cache:
            cleaned.to_pickle(cache_file)
        frames.append(cleaned)

    if not frames:
        empty = pd.DataFrame(columns=["Timestamp", "Phase", "Status_Code",
                                      "Latency_ms", "Provider", "Cycle",
                                      "WarmSlot"])
        empty["Provider"] = pd.Categorical(empty["Provider"], categories=[])
        return empty

    combined = pd.concat(frames, ignore_index=True)

    # Providers ordered as PROVIDERS_ORDER first, then any newcomers
    # appended in first-seen order. Categories come from every *discovered*
    # file (data_files.keys()), not just providers with surviving rows — a
    # provider whose file exists but had 0 rows survive cleaning (e.g.
    # every request came back non-2xx) still gets a category, so it shows
    # up as an explicit "no data" panel instead of silently vanishing.
    all_providers = list(data_files.keys())
    ordered = [p for p in PROVIDERS_ORDER if p in all_providers] + \
              [p for p in all_providers if p not in PROVIDERS_ORDER]
    combined["Provider"] = pd.Categorical(
        combined["Provider"], categories=ordered, ordered=True)
    return combined


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Violin + Box + Strip
# ════════════════════════════════════════════════════════════════════════════

def fig_violin(df: pd.DataFrame, out_dir: Path) -> None:
    # Every discovered provider gets a column, even one with zero surviving
    # rows — the per-phase "no data" placeholder a few lines down already
    # handles an empty `vals`, so an empty provider renders as a clearly
    # labeled blank column instead of silently disappearing.
    providers = list(df["Provider"].cat.categories)
    n = len(providers)
    if n == 0:
        print("  Figure 1 skipped: no data.")
        return

    fig = plt.figure(figsize=(max(7.2, 2.3 * n), 5.2))
    gs = gridspec.GridSpec(2, n, figure=fig,
                            height_ratios=[1.6, 1.0],
                            hspace=0.38, wspace=0.28,
                            left=0.09, right=0.94,
                            top=0.88, bottom=0.08)

    fig.suptitle(
        "Figure 1 — Latency Distributions: Cold Start (top) vs. Warm Ping (bottom)\n"
        r"$\it{Violin\ (KDE)\ +\ box\ (IQR)\ +\ jittered\ observations;\ "
        r"independent\ y\text{-}axes\ per\ row}$",
        fontsize=9)

    rng = np.random.default_rng(42)  # one shared stream for the whole figure

    # ax_info[row] holds (ax, vals, phase_color) only for axes with real
    # data — axes that hit "no data" never enter this list, so an empty
    # panel can no longer pull the shared y-range toward [0, 1].
    ax_info = {0: [], 1: []}

    for col, prov in enumerate(providers):
        sub = df[df["Provider"] == prov]
        pc = _provider_color(prov, col)

        for row, phase in enumerate(["Cold Start", "Warm Ping"]):
            ax = fig.add_subplot(gs[row, col])
            vals = sub.loc[sub["Phase"] == phase, "Latency_ms"]
            ph_c = PHASE_COLORS[phase]

            ax.set_xlim(-0.55, 0.85)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_linewidth(0.7)
            if row == 0:
                ax.set_title(prov, fontsize=8.5, color=pc,
                             fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(f"{phase}\nLatency (ms)", labelpad=4, fontsize=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if vals.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
                continue  # NOT added to ax_info -> can't corrupt shared range

            if len(vals) >= 4:
                parts = ax.violinplot(
                    vals, positions=[0], widths=0.7,
                    showmeans=False, showmedians=False, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(ph_c)
                    body.set_alpha(0.22)
                    body.set_edgecolor(ph_c)
                    body.set_linewidth(0.6)

            ax.boxplot(
                vals, positions=[0], widths=0.18,
                patch_artist=True, notch=False,
                whis=1.5, showfliers=False,
                medianprops=dict(color="white", linewidth=2.2, zorder=6),
                boxprops=dict(facecolor=ph_c, alpha=0.82,
                              edgecolor=ph_c, linewidth=0.7, zorder=5),
                whiskerprops=dict(color=ph_c, linewidth=0.9, zorder=5),
                capprops=dict(color=ph_c, linewidth=1.2, zorder=5))

            jitter = rng.uniform(-0.14, 0.14, len(vals))
            ax.scatter(jitter, vals,
                       s=18, color=ph_c, alpha=0.55,
                       linewidths=0.3, edgecolors="white", zorder=7)

            ax.text(0.97, 0.04, f"n = {len(vals)}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.8, color="#666666")

            # Must come AFTER boxplot()/violinplot(): both reset the x-tick
            # locator to a tick at each `positions` value (matplotlib does
            # this internally to label box/violin categories), so clearing
            # ticks any earlier gets silently overwritten and a stray "0"
            # tick reappears under every panel.
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

            ax_info[row].append((ax, vals, ph_c))

    # Share y-limits within each row, using only axes that had real data.
    def _share_ylim(entries):
        if not entries:
            return None
        all_vals = []
        for ax, _, _ in entries:
            all_vals.extend(ax.get_ylim())
        mn, mx = min(all_vals), max(all_vals)
        pad = (mx - mn) * 0.08 if mx > mn else max(abs(mx), 1) * 0.08
        lo, hi = mn - pad, mx + pad
        for ax, _, _ in entries:
            ax.set_ylim(lo, hi)
        return lo, hi

    row_ranges = {row: _share_ylim(entries) for row, entries in ax_info.items()}

    # Median annotations drawn AFTER the row range is finalized, so the
    # offset is a fraction of the range everyone actually shares — this is
    # what was wrong before: the offset was computed pre-share and only
    # happened to look right on the axis that set the max range.
    for row, entries in ax_info.items():
        row_range = row_ranges[row]
        if row_range is None:
            continue
        lo, hi = row_range
        span = hi - lo
        for ax, vals, ph_c in entries:
            med = float(np.median(vals))
            ax.annotate(f"med={med:.0f}",
                        xy=(0.09, med),
                        xytext=(0.40, med + span * 0.05),
                        fontsize=6, color=ph_c, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=ph_c,
                                        lw=0.5, alpha=0.7),
                        annotation_clip=False, zorder=8)

    if ax_info[0]:
        ax_info[0][-1][0].annotate(
            "COLD START", xy=(1.06, 0.5), xycoords="axes fraction",
            rotation=-90, va="center", ha="left", fontsize=7.5,
            color=PHASE_COLORS["Cold Start"], fontweight="bold")
    if ax_info[1]:
        ax_info[1][-1][0].annotate(
            "WARM PING", xy=(1.06, 0.5), xycoords="axes fraction",
            rotation=-90, va="center", ha="left", fontsize=7.5,
            color=PHASE_COLORS["Warm Ping"], fontweight="bold")

    _save(fig, out_dir, "fig1_violin_box_strip")
    plt.close(fig)
    print("  Saved Figure 1.")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Cycle Stability Time-Series
# ════════════════════════════════════════════════════════════════════════════

def fig_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    providers = list(df["Provider"].cat.categories)
    n = len(providers)
    if n == 0:
        print("  Figure 3 skipped: no data.")
        return

    fig, axes = plt.subplots(1, n, figsize=(max(7.2, 2.4 * n), 3.2),
                              gridspec_kw=dict(wspace=0.55,
                                                left=0.08, right=0.93,
                                                top=0.78, bottom=0.16))
    fig.suptitle(
        "Figure 3 — Latency Stability Across Consecutive Cycles\n"
        r"$\it{Cold\ start\ per\ cycle\ (●,\ left\ axis)\ vs.\ mean\ warm\ ping\ "
        r"±1σ\ (■,\ right\ axis)}$",
        fontsize=9)

    if n == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None

    for col, (ax_cold, prov) in enumerate(zip(axes, providers)):
        sub = df[df["Provider"] == prov]
        pc = _provider_color(prov, col)
        cycles = sorted(sub["Cycle"].unique())

        if not cycles:
            # Discovered provider, zero rows survived cleaning (e.g. every
            # request came back non-2xx, as with AWS Lambda's 403s here).
            # Same "no data" treatment fig_violin gives an empty panel,
            # instead of vanishing or crashing on xs.min()/max() below.
            ax_cold.text(0.5, 0.5, "no data", transform=ax_cold.transAxes,
                         ha="center", va="center", fontsize=8, color="#999999")
            ax_cold.set_title(prov, fontsize=8.5, color=pc,
                               fontweight="bold", pad=4)
            ax_cold.set_xticks([])
            ax_cold.set_yticks([])
            ax_cold.spines["left"].set_visible(False)
            ax_cold.spines["bottom"].set_visible(False)
            continue

        cold_lat, warm_mean, warm_std = [], [], []
        for cyc in cycles:
            c_df = sub[sub["Cycle"] == cyc]
            c_vals = c_df.loc[c_df["Phase"] == "Cold Start", "Latency_ms"]
            w_vals = c_df.loc[c_df["Phase"] == "Warm Ping", "Latency_ms"]
            cold_lat.append(c_vals.mean() if not c_vals.empty else np.nan)
            warm_mean.append(w_vals.mean() if not w_vals.empty else np.nan)
            warm_std.append(w_vals.std() if len(w_vals) > 1 else 0)

        xs = np.array(cycles, dtype=float)
        cold_arr = np.array(cold_lat, dtype=float)
        warm_arr = np.array(warm_mean, dtype=float)
        std_arr = np.array(warm_std, dtype=float)

        # Twin y-axis: cold start (large magnitude) on the left, warm ping
        # (1-2 orders of magnitude smaller) on the right. This is the fix —
        # on a shared linear axis the warm line was visually flat at y≈0,
        # confirmed by rendering the single-axis version.
        ax_warm = ax_cold.twinx()

        (line_cold,) = ax_cold.plot(
            xs, cold_arr, color=PHASE_COLORS["Cold Start"],
            lw=1.6, marker="o", markersize=5.5, markerfacecolor="white",
            markeredgecolor=PHASE_COLORS["Cold Start"], markeredgewidth=1.4,
            zorder=4, label="Cold Start")

        (line_warm,) = ax_warm.plot(
            xs, warm_arr, color=PHASE_COLORS["Warm Ping"],
            lw=1.6, marker="s", markersize=4.5, markerfacecolor="white",
            markeredgecolor=PHASE_COLORS["Warm Ping"], markeredgewidth=1.4,
            zorder=4, label="Warm (mean, right axis)")
        ax_warm.fill_between(xs, warm_arr - std_arr, warm_arr + std_arr,
                              color=PHASE_COLORS["Warm Ping"], alpha=0.15,
                              zorder=2, linewidth=0)

        # Guard against "Mean of empty slice": a provider can legitimately
        # have zero Warm Ping rows for a stretch (e.g. monitoring outage),
        # which leaves warm_arr entirely NaN.
        if np.any(~np.isnan(warm_arr)):
            gm = np.nanmean(warm_arr)
            ax_warm.axhline(gm, color=PHASE_COLORS["Warm Ping"],
                             lw=0.7, ls="--", alpha=0.45, zorder=1)

        # Cosmetic headroom so the warm band doesn't touch the panel edge.
        # If a provider has zero Warm Ping rows in this window, leave the
        # right axis unlabeled rather than drawing a fake [0, 0] range.
        has_warm = np.any(~np.isnan(warm_arr))
        if has_warm:
            w_lo = np.nanmin(warm_arr - std_arr)
            w_hi = np.nanmax(warm_arr + std_arr)
            pad = max((w_hi - w_lo) * 0.25, 0.5)
            ax_warm.set_ylim(max(0, w_lo - pad), w_hi + pad)
        else:
            ax_warm.set_yticks([])
            ax_warm.spines["right"].set_visible(False)
        if np.any(~np.isnan(cold_arr)):
            c_lo, c_hi = np.nanmin(cold_arr), np.nanmax(cold_arr)
            pad = max((c_hi - c_lo) * 0.15, 1)
            ax_cold.set_ylim(max(0, c_lo - pad), c_hi + pad)

        ax_cold.set_title(prov, fontsize=8.5, color=pc, fontweight="bold", pad=4)
        ax_cold.set_xlabel("Cycle", labelpad=3)
        if col == 0:
            ax_cold.set_ylabel("Cold start (ms)", labelpad=3,
                                color=PHASE_COLORS["Cold Start"])
        if col == n - 1 and has_warm:
            ax_warm.set_ylabel("Warm ping, mean ±1σ (ms)", labelpad=8,
                                color=PHASE_COLORS["Warm Ping"], rotation=-90,
                                va="bottom")
        elif has_warm:
            ax_warm.set_yticklabels([])

        # Cap the number of x-ticks so this stays legible as cycles grow
        # past ~15-20; always keep the first and last cycle visible.
        ax_cold.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=10, integer=True, min_n_ticks=1))
        ax_cold.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v)}" if float(v).is_integer() else ""))
        ax_cold.set_xlim(xs.min() - 0.4, xs.max() + 0.4)

        ax_cold.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_warm.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_cold.tick_params(axis="y", colors=PHASE_COLORS["Cold Start"])
        ax_warm.tick_params(axis="y", colors=PHASE_COLORS["Warm Ping"])
        ax_cold.spines["left"].set_color(PHASE_COLORS["Cold Start"])
        ax_warm.spines["right"].set_color(PHASE_COLORS["Warm Ping"])
        ax_warm.spines["top"].set_visible(False)
        ax_cold.grid(True, axis="y", alpha=0.25)
        ax_warm.grid(False)

        if legend_handles is None:
            legend_handles = [line_cold, line_warm]
            legend_labels = [h.get_label() for h in legend_handles]

    # One shared legend at the figure level so it doesn't disappear if
    # providers[0] happens to be the one missing from a future CSV drop.
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc="upper right", bbox_to_anchor=(0.99, 0.99),
                   fontsize=6.5, framealpha=0.93, borderpad=0.4,
                   handlelength=1.5, ncol=1)

    _save(fig, out_dir, "fig3_timeseries_cycles")
    plt.close(fig)
    print("  Saved Figure 3.")


# ── Save helper ──────────────────────────────────────────────────────────────
def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        kw = dict(bbox_inches="tight", pad_inches=0.06)
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(out_dir / f"{stem}.{ext}", format=ext, **kw)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_dir = Path(".")
    out_dir = Path("figures_fixed")
    out_dir.mkdir(exist_ok=True)

    print("Loading data …")
    df = load_data(data_dir)
    print(f"  {len(df)} observations across "
          f"{df['Provider'].nunique() if len(df) else 0} provider(s)\n")

    print("Rendering …")
    fig_violin(df, out_dir)
    fig_timeseries(df, out_dir)
    print(f"\nDone — outputs in {out_dir}/")