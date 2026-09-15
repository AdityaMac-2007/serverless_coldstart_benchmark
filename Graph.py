from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("latency_viz")


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

class Phase(Enum):
    COLD = "Cold Start"
    WARM = "Warm Ping"


@dataclass(frozen=True, slots=True)
class VizConfig:
    """Centralized configuration for figure generation."""
    font_family: str = "sans-serif"
    base_font_size: float = 9.0
    figure_dpi: int = 150
    save_dpi: int = 300
    
    # Layout (IEEE double-column geometry)
    figure_width: float = 7.2  # inches
    panel_height: float = 3.1
    panel_height_ts: float = 3.2
    
    # Filtering parameters
    cutoff_date: pd.Timestamp = pd.Timestamp("2026-09-01")
    min_latency_ms: float = 0.0
    
    # Cache parameters
    cache_dir: Path = Path(".cache_latency")
    cache_logic_version: str = "8"
    
    def apply_rc(self) -> None:
        """Apply standardized matplotlib parameters."""
        plt.rcParams.update({
            "font.family": self.font_family,
            "font.size": self.base_font_size,
            "axes.titlesize": 10,
            "axes.labelsize": self.base_font_size,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "figure.dpi": self.figure_dpi,
            "savefig.dpi": self.save_dpi,
            "figure.constrained_layout.use": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.45,
            "grid.color": "#aaaaaa",
            "axes.axisbelow": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.93,
            "legend.edgecolor": "#cccccc",
        })


PALETTE = {
    "AWS Lambda":          "#E07B39",
    "AWS Lambda (Mumbai)": "#E07B39",
    "Google Cloud Run":    "#2E6DB4",
    "Azure Functions":     "#2A9D60",
    Phase.COLD:            "#C0392B",
    Phase.WARM:            "#2471A3",
}

FALLBACK_CYCLE = ["#7B5EA7", "#B4652E", "#4C8577", "#A34F6B"]

PROVIDERS_ORDER = [
    "AWS Lambda",
    "AWS Lambda (Mumbai)",
    "Google Cloud Run",
    "Azure Functions",
]

KNOWN_DATA_FILES: dict[str, str] = {
    # Primary Experiment (83 cycles)
    "aws_lambda_latency_data.csv":        "AWS Lambda",
    "gcp_latency_data.csv":               "Google Cloud Run",
    "azure_latency_data.csv":             "Azure Functions",
    # Domestic Replication (99 cycles)
    "aws_mumbai_backup_latency_data.csv": "AWS Lambda (Mumbai)",
    "gcp_backup_latency_data.csv":        "Google Cloud Run",
    "azure_backup_latency_data.csv":      "Azure Functions",
}


# ════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """Handles data discovery, cleaning, caching, and loading."""
    
    def __init__(self, data_dir: Path, config: VizConfig | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or VizConfig()
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def _display_name_from_filename(filename: str) -> str:
        stem = filename.removesuffix("_latency_data.csv").removesuffix(".csv")
        return stem.replace("_", " ").strip().title()
    
    def discover(self, prefer_backup: bool = False) -> dict[str, str]:
        files: dict[str, str] = {}
        for filename, display in KNOWN_DATA_FILES.items():
            fpath = self.data_dir / filename
            if not fpath.exists():
                continue
            if prefer_backup and "backup" in filename:
                files[display] = filename
            elif not prefer_backup and "backup" not in filename:
                files[display] = filename
        
        for fpath in sorted(self.data_dir.glob("*_latency_data.csv")):
            if fpath.name in KNOWN_DATA_FILES:
                continue
            name = self._display_name_from_filename(fpath.name)
            if name not in files:
                files[name] = fpath.name
        
        return files
    
    @staticmethod
    def _content_fingerprint(fpath: Path) -> str:
        h = hashlib.sha256()
        h.update(fpath.read_bytes())
        return h.hexdigest()[:24]
    
    def _cache_path(self, filename: str, fingerprint: str) -> Path:
        key = hashlib.sha1(
            f"{self.config.cache_logic_version}:{filename}:{fingerprint}".encode()
        ).hexdigest()[:16]
        return self.cache_dir / f"{key}.pkl"
    
    def clean(self, df: pd.DataFrame, provider: str) -> pd.DataFrame:
        df = df.copy()
        df["Provider"] = provider
        df = df[df["Timestamp"] >= self.config.cutoff_date].copy()
        df["Latency_ms"] = pd.to_numeric(df["Latency_ms"], errors="coerce")
        df.dropna(subset=["Latency_ms"], inplace=True)
        
        df = df[
            (df["Latency_ms"] > self.config.min_latency_ms) &
            df["Status_Code"].astype(str).str.startswith("2")
        ].copy()
        
        if df.empty:
            logger.warning("No valid observations for %s after filtering.", provider)
            return df
        
        def _categorize_phase(raw: object) -> str | None:
            r = str(raw).lower()
            if "cold" in r:
                return Phase.COLD.value
            if "warm" in r:
                return Phase.WARM.value
            return None
        
        df["Phase"] = df["Phase"].apply(_categorize_phase)
        df.dropna(subset=["Phase"], inplace=True)
        
        if df.empty:
            logger.warning("No valid phases for %s after categorization.", provider)
            return df
        
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df["Cycle"] = (df["Phase"] == Phase.COLD.value).cumsum()
        
        warm_mask = df["Phase"] == Phase.WARM.value
        df.loc[warm_mask, "WarmSlot"] = (
            df[warm_mask].groupby("Cycle").cumcount() + 1
        )
        
        return df
    
    def load(self, prefer_backup: bool = False, use_cache: bool = True) -> pd.DataFrame:
        data_files = self.discover(prefer_backup=prefer_backup)
        if not data_files:
            logger.error("No data files found in %s", self.data_dir.resolve())
            return self._empty_frame()
        
        frames: list[pd.DataFrame] = []
        for provider, filename in data_files.items():
            fpath = self.data_dir / filename
            if not fpath.exists():
                logger.warning("File missing: %s", fpath)
                continue
            
            if use_cache:
                fp = self._content_fingerprint(fpath)
                cache_file = self._cache_path(filename, fp)
                if cache_file.exists():
                    logger.debug("Cache hit for %s", filename)
                    frames.append(pd.read_pickle(cache_file))
                    continue
            
            try:
                raw = pd.read_csv(fpath, parse_dates=["Timestamp"])
                cleaned = self.clean(raw, provider)
            except Exception as exc:
                logger.error("Failed to process %s: %s", filename, exc)
                continue
            
            if cleaned.empty:
                continue
            
            if use_cache:
                fp = self._content_fingerprint(fpath)
                cache_file = self._cache_path(filename, fp)
                cleaned.to_pickle(cache_file)
            
            frames.append(cleaned)
        
        if not frames:
            logger.error("All files failed to load or were empty.")
            return self._empty_frame()
        
        combined = pd.concat(frames, ignore_index=True)
        all_providers = list(data_files.keys())
        ordered = [p for p in PROVIDERS_ORDER if p in all_providers] + \
                  [p for p in all_providers if p not in PROVIDERS_ORDER]
        
        combined["Provider"] = pd.Categorical(
            combined["Provider"], categories=ordered, ordered=True
        )
        return combined
    
    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        cols = ["Timestamp", "Phase", "Status_Code", "Latency_ms",
                "Provider", "Cycle", "WarmSlot"]
        df = pd.DataFrame(columns=cols)
        df["Provider"] = pd.Categorical(df["Provider"], categories=[])
        return df


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZATION LAYER — Base
# ════════════════════════════════════════════════════════════════════════════

class FigureBase:
    """Abstract base for publication figures."""
    
    def __init__(self, config: VizConfig | None = None) -> None:
        self.config = config or VizConfig()
    
    @staticmethod
    def _save(fig: Figure, out_dir: Path, stem: str) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kw = dict(bbox_inches="tight", pad_inches=0.06)
        fig.savefig(out_dir / f"{stem}.png", dpi=300, format="png", **kw)
        fig.savefig(out_dir / f"{stem}.pdf", format="pdf", **kw)
        logger.info("Saved %s.{png,pdf}", stem)
    
    @staticmethod
    def _provider_color(provider: str, idx: int) -> str:
        return PALETTE.get(provider, FALLBACK_CYCLE[idx % len(FALLBACK_CYCLE)])
    
    def _active_providers(self, df: pd.DataFrame) -> list[str]:
        cats = df["Provider"].cat.categories
        return [p for p in cats if p in df["Provider"].values]


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Empirical Cumulative Distribution Functions (ECDF)
# ════════════════════════════════════════════════════════════════════════════

class FigureECDF(FigureBase):
    """Empirical Cumulative Distribution Functions."""
    
    def render(
        self,
        df: pd.DataFrame,
        out_dir: Path,
        suffix: str = "",
    ) -> None:
        providers = self._active_providers(df)
        if not providers:
            logger.warning("Figure 1 (ECDF) skipped: no data.")
            return
        
        # Increased height from 3.1 to 3.4 to give dedicated breathing room for the legend
        fig, (ax_cold, ax_warm) = plt.subplots(
            1, 2,
            figsize=(self.config.figure_width, 3.4),
            gridspec_kw=dict(
                wspace=0.28, left=0.08, right=0.96, top=0.74, bottom=0.15
            ),
        )
        
        title_desc = "Domestic Mumbai Replication" if suffix else "Primary Experiment"
        fig.suptitle(
            f"Figure 1 — Empirical Cumulative Distribution Functions (ECDF) [{title_desc}]\n"
            r"$\it{Left:\ Cold\ Start\ Latency;\ Right:\ Warm\ Baseline\ "
            r"(Dashed\ Lines:\ p50\ and\ p95)}$",
            fontsize=8.8,
            y=0.98,
        )
        
        for idx, prov in enumerate(providers):
            sub = df[df["Provider"] == prov]
            color = self._provider_color(prov, idx)
            
            for ax, phase in [(ax_cold, Phase.COLD), (ax_warm, Phase.WARM)]:
                vals = sub.loc[sub["Phase"] == phase.value, "Latency_ms"].dropna().values
                if len(vals) == 0:
                    continue
                sorted_vals = np.sort(vals)
                y = np.linspace(0, 1, len(sorted_vals))
                ax.step(
                    sorted_vals, y,
                    label=prov, color=color, lw=1.6, where="post",
                )
        
        for ax, title in [(ax_cold, "Cold Start Latency"), (ax_warm, "Warm Ping Baseline")]:
            ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
            ax.set_xlabel("Latency (ms)", fontsize=8.5)
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 0.95, 1.0])
            ax.set_yticklabels(["0%", "25%", "p50", "75%", "p95", "100%"])
            ax.axhline(0.50, color="#777777", linestyle=":", lw=0.8, zorder=1)
            ax.axhline(0.95, color="#777777", linestyle=":", lw=0.8, zorder=1)
        
        ax_cold.set_ylabel(r"Cumulative Probability ($P(X \leq x)$)", fontsize=8)
        ax_warm.set_ylabel("")
        
        # Legend placed between title and plot subplots
        handles, labels = ax_cold.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.86),
                ncol=len(providers),
                fontsize=8,
                frameon=False,
            )
        
        stem = f"fig1_latency_ecdf{suffix}"
        self._save(fig, out_dir, stem)
        plt.close(fig)
# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Longitudinal Cycle Stability Time-Series
# ════════════════════════════════════════════════════════════════════════════

class FigureTimeSeries(FigureBase):
    """Longitudinal cycle stability."""
    
    def render(
        self,
        df: pd.DataFrame,
        out_dir: Path,
        suffix: str = "",
    ) -> None:
        providers = self._active_providers(df)
        n = len(providers)
        if n == 0:
            logger.warning("Figure 3 (Time-Series) skipped: no data.")
            return
        
        cycle_counts = [
            df.loc[df["Provider"] == p, "Cycle"].nunique()
            for p in providers
        ]
        max_cycles = max(cycle_counts) if cycle_counts else 0
        panel_w = 2.4 + 0.03 * max(0, max_cycles - 15)
        
        fig, axes = plt.subplots(
            1, n,
            figsize=(max(self.config.figure_width, panel_w * n), self.config.panel_height_ts),
            gridspec_kw=dict(
                wspace=0.55, left=0.08, right=0.93, top=0.78, bottom=0.16,
            ),
        )
        fig.suptitle(
            "Figure 3 — Longitudinal Latency Stability Across Consecutive Cycles\n"
            r"$\it{Cold\ start\ per\ cycle\ (●,\ left\ axis)\ vs.\ mean\ warm\ ping\ "
            r"\pm 1\sigma\ (■,\ right\ axis)}$",
            fontsize=9,
        )
        
        if n == 1:
            axes = [axes]
        
        legend_handles: list[plt.Line2D] | None = None
        legend_labels: list[str] | None = None
        
        for col, (ax_cold, prov) in enumerate(zip(axes, providers)):
            sub = df[df["Provider"] == prov]
            pc = self._provider_color(prov, col)
            cycles = sorted(sub["Cycle"].dropna().unique().astype(int))
            
            if not cycles:
                self._render_empty_panel(ax_cold, prov, pc)
                continue
            
            cold_lat, warm_mean, warm_std = [], [], []
            for cyc in cycles:
                c_df = sub[sub["Cycle"] == cyc]
                c_vals = c_df.loc[c_df["Phase"] == Phase.COLD.value, "Latency_ms"]
                w_vals = c_df.loc[c_df["Phase"] == Phase.WARM.value, "Latency_ms"]
                cold_lat.append(float(c_vals.mean()) if not c_vals.empty else np.nan)
                warm_mean.append(float(w_vals.mean()) if not w_vals.empty else np.nan)
                warm_std.append(float(w_vals.std()) if len(w_vals) > 1 else 0.0)
            
            xs = np.array(cycles, dtype=float)
            cold_arr = np.array(cold_lat, dtype=float)
            warm_arr = np.array(warm_mean, dtype=float)
            std_arr = np.array(warm_std, dtype=float)
            
            target_markers = 18
            stride = max(1, math.ceil(len(xs) / target_markers))
            marker_idxs = sorted(set(range(0, len(xs), stride)) | {len(xs) - 1})
            line_w = 1.1 if len(xs) > target_markers else 1.6
            
            ax_warm = ax_cold.twinx()
            
            line_cold, = ax_cold.plot(
                xs, cold_arr,
                color=PALETTE[Phase.COLD],
                lw=line_w, marker="o", markersize=5.5,
                markerfacecolor="white",
                markeredgecolor=PALETTE[Phase.COLD],
                markeredgewidth=1.4,
                markevery=marker_idxs, zorder=4,
                label="Cold Start",
            )
            
            line_warm, = ax_warm.plot(
                xs, warm_arr,
                color=PALETTE[Phase.WARM],
                lw=line_w, marker="s", markersize=4.5,
                markerfacecolor="white",
                markeredgecolor=PALETTE[Phase.WARM],
                markeredgewidth=1.4,
                markevery=marker_idxs, zorder=4,
                label="Warm (mean, right axis)",
            )
            ax_warm.fill_between(
                xs, warm_arr - std_arr, warm_arr + std_arr,
                color=PALETTE[Phase.WARM], alpha=0.15,
                zorder=2, linewidth=0,
            )
            
            if np.any(~np.isnan(warm_arr)):
                ax_warm.axhline(
                    np.nanmean(warm_arr),
                    color=PALETTE[Phase.WARM],
                    lw=0.7, ls="--", alpha=0.45, zorder=1,
                )
            
            self._set_axis_limits(ax_cold, cold_arr, ax_warm, warm_arr, std_arr, col, n)
            self._style_panel(ax_cold, ax_warm, prov, pc, col, n, xs)
            
            if legend_handles is None:
                legend_handles = [line_cold, line_warm]
                legend_labels = [h.get_label() for h in legend_handles]
        
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                loc="upper right", bbox_to_anchor=(0.99, 0.99),
                fontsize=6.5, framealpha=0.93, borderpad=0.4,
                handlelength=1.5, ncol=1,
            )
        
        stem = f"fig3_timeseries_cycles{suffix}"
        self._save(fig, out_dir, stem)
        plt.close(fig)
    
    def _render_empty_panel(self, ax: Axes, prov: str, color: str) -> None:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="#999999")
        ax.set_title(prov, fontsize=8.5, color=color, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _set_axis_limits(
        self,
        ax_cold: Axes,
        cold_arr: np.ndarray,
        ax_warm: Axes,
        warm_arr: np.ndarray,
        std_arr: np.ndarray,
        col: int,
        n: int,
    ) -> None:
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
    
    def _style_panel(
        self,
        ax_cold: Axes,
        ax_warm: Axes,
        prov: str,
        pc: str,
        col: int,
        n: int,
        xs: np.ndarray,
    ) -> None:
        has_warm = len(ax_warm.get_yticks()) > 0 or ax_warm.has_data()
        
        ax_cold.set_title(prov, fontsize=8.5, color=pc, fontweight="bold", pad=4)
        ax_cold.set_xlabel("Cycle", labelpad=3)
        
        if col == 0:
            ax_cold.set_ylabel(
                "Cold start (ms)", labelpad=3,
                color=PALETTE[Phase.COLD],
            )
        if col == n - 1 and has_warm:
            ax_warm.set_ylabel(
                "Warm ping, mean ±1σ (ms)", labelpad=8,
                color=PALETTE[Phase.WARM],
                rotation=-90, va="bottom",
            )
        elif has_warm:
            ax_warm.set_yticklabels([])
        
        nbins = min(16, max(8, len(xs) // 6))
        ax_cold.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=nbins, integer=True, min_n_ticks=1)
        )
        ax_cold.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda v, _: f"{int(v)}" if float(v).is_integer() else ""
            )
        )
        ax_cold.set_xlim(xs.min() - 0.4, xs.max() + 0.4)
        
        fmt = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        ax_cold.yaxis.set_major_formatter(fmt)
        ax_warm.yaxis.set_major_formatter(fmt)
        
        ax_cold.tick_params(axis="y", colors=PALETTE[Phase.COLD])
        ax_warm.tick_params(axis="y", colors=PALETTE[Phase.WARM])
        ax_cold.spines["left"].set_color(PALETTE[Phase.COLD])
        ax_warm.spines["right"].set_color(PALETTE[Phase.WARM])
        ax_warm.spines["top"].set_visible(False)
        ax_cold.grid(True, axis="y", alpha=0.25)
        ax_warm.grid(False)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Geographic Invariance
# ════════════════════════════════════════════════════════════════════════════

class FigureGeographic(FigureBase):
    """Regional comparison and provisioning overhead stability."""
    
    AWS_STOCKHOLM_WARM_MED = 443.59
    AWS_STOCKHOLM_DELTA = 284.80
    AWS_MUMBAI_WARM_MED = 723.36
    AWS_MUMBAI_DELTA = 262.22
    
    REPLICATION_MEDIANS = [985.58, 980.52, 1617.87]
    
    def render(self, data_dir: Path, out_dir: Path) -> None:
        data_dir = Path(data_dir)
        required = [
            data_dir / "aws_mumbai_backup_latency_data.csv",
            data_dir / "gcp_backup_latency_data.csv",
            data_dir / "azure_backup_latency_data.csv",
        ]
        if not all(f.exists() for f in required):
            logger.warning("Figure 4 skipped: backup data not found.")
            return
        
        pipeline = DataPipeline(data_dir)
        
        try:
            df_aws = pipeline.clean(
                pd.read_csv(required[0], parse_dates=["Timestamp"]),
                "AWS Mumbai",
            )
            df_gcp = pipeline.clean(
                pd.read_csv(required[1], parse_dates=["Timestamp"]),
                "Google Cloud Run",
            )
            df_az = pipeline.clean(
                pd.read_csv(required[2], parse_dates=["Timestamp"]),
                "Azure Functions",
            )
        except Exception as exc:
            logger.error("Failed to load backup data for Figure 4: %s", exc)
            return
        
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(self.config.figure_width, self.config.panel_height_ts),
            gridspec_kw=dict(
                wspace=0.35, left=0.09, right=0.95, top=0.82, bottom=0.15,
            ),
        )
        fig.suptitle(
            r"Figure 4 — Geographic Invariance & Domestic Subcontinent Validation ($N=99$)" + "\n"
            r"$\it{Left:\ AWS\ Cross\text{-}Region\ Overhead\ Invariance\ "
            r"(\Delta_{\mathrm{cold}});\ Right:\ All\text{-}India\ Replication\ Boxplots}$",
            fontsize=8.5,
        )
        
        self._panel_overhead(ax1)
        self._panel_boxplot(ax2, [df_aws, df_gcp, df_az])
        
        self._save(fig, out_dir, "fig4_geographic_invariance")
        plt.close(fig)
    
    def _panel_overhead(self, ax: Axes) -> None:
        ax.cla()  # Clears any duplicate draws or ghost layers
        
        categories = ["Stockholm\n(eu-north-1)", "Mumbai\n(ap-south-1)"]
        warm_meds = [self.AWS_STOCKHOLM_WARM_MED, self.AWS_MUMBAI_WARM_MED]
        deltas = [self.AWS_STOCKHOLM_DELTA, self.AWS_MUMBAI_DELTA]
        x_pos = np.arange(len(categories))
        width = 0.42
        
        # Single draw pass
        ax.bar(
            x_pos, warm_meds, width,
            label=r"Warm Baseline ($L_{\mathrm{net}}$)",
            color=PALETTE[Phase.WARM], alpha=0.85,
        )
        ax.bar(
            x_pos, deltas, width, bottom=warm_meds,
            label=r"Provisioning ($\Delta_{\mathrm{cold}}$)",
            color=PALETTE[Phase.COLD], alpha=0.85,
        )
        
        ax.set_ylabel("Latency (ms)", fontsize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_title("AWS Provisioning Overhead Stability", fontsize=8.5, fontweight="bold", pad=6)
        
        # Legend anchored cleanly at upper left
        ax.legend(loc="upper left", fontsize=6.8, framealpha=0.92, edgecolor="#cccccc")
        
        # Values inside bars
        for i, (w, d) in enumerate(zip(warm_meds, deltas)):
            ax.text(
                i, w + d / 2, rf"$\Delta={d:.1f}$",
                ha="center", va="center",
                color="white", fontweight="bold", fontsize=7.5,
            )
        
        # Invariance badge placed on the upper right, completely clear of the legend
        diff = abs(self.AWS_STOCKHOLM_DELTA - self.AWS_MUMBAI_DELTA)
        ax.text(
            1.0, 1180, rf"$\Delta_{{\mathrm{{diff}}}} = {diff:.2f}\,$ms",
            ha="center", va="center", fontsize=7.2, color="#222222",
            bbox=dict(boxstyle="round,pad=0.25", fc="#f8f9fa", ec="#bbbbbb", lw=0.7),
        )
        ax.set_ylim(0, 1320)
    
    def _panel_boxplot(self, ax: Axes, cold_dataframes: list[pd.DataFrame]) -> None:
        cold_data = [
            df.loc[df["Phase"] == Phase.COLD.value, "Latency_ms"].dropna()
            for df in cold_dataframes
        ]
        labels = ["AWS\n(Mumbai)", "GCP\n(Mumbai)", "Azure\n(Pune)"]
        colors = [
            PALETTE["AWS Lambda (Mumbai)"],
            PALETTE["Google Cloud Run"],
            PALETTE["Azure Functions"],
        ]
        
        bp = ax.boxplot(
            cold_data,
            positions=[1, 2, 3],
            widths=0.45,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=2.0),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor(c)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Cold Start Latency (ms)", fontsize=8)
        ax.set_title("Domestic Ingress Distributions ($N=99$)", fontsize=8.5, fontweight="bold")
        
        for i, m in enumerate(self.REPLICATION_MEDIANS, start=1):
            ax.text(
                i, m + 60, f"{m:.0f} ms",
                ha="center", fontsize=7,
                fontweight="bold", color="#333333",
            )


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ════════════════════════════════════════════════════════════════════════════

class PublicationPipeline:
    """End-to-end orchestration."""
    
    def __init__(
        self,
        data_dir: Path | str = ".",
        out_dir: Path | str = "figures_fixed",
        config: VizConfig | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.config = config or VizConfig()
        self.data_pipeline = DataPipeline(self.data_dir, self.config)
        
        self.fig_ecdf = FigureECDF(self.config)
        self.fig_ts = FigureTimeSeries(self.config)
        self.fig_geo = FigureGeographic(self.config)
    
    def run(self) -> None:
        self.config.apply_rc()
        
        # Primary experiment
        logger.info("Generating Primary Experiment Figures (83 Cycles) …")
        df_primary = self.data_pipeline.load(prefer_backup=False)
        if not df_primary.empty:
            self.fig_ecdf.render(df_primary, self.out_dir)
            self.fig_ts.render(df_primary, self.out_dir)
        else:
            logger.warning("Primary dataset is empty or files not present locally.")
        
        # Domestic backup
        logger.info("Generating Domestic Backup Figures (99 Cycles) …")
        df_backup = self.data_pipeline.load(prefer_backup=True)
        if not df_backup.empty:
            self.fig_ecdf.render(df_backup, self.out_dir, suffix="_backup_mumbai")
            self.fig_ts.render(df_backup, self.out_dir, suffix="_backup_mumbai")
        else:
            logger.warning("Backup dataset is empty or files not present locally.")
        
        # Geographic invariance
        logger.info("Generating Figure 4 (Geographic Invariance Comparison) …")
        self.fig_geo.render(self.data_dir, self.out_dir)
        
        logger.info("All figures rendered into: %s/", self.out_dir.resolve())


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PublicationPipeline().run()