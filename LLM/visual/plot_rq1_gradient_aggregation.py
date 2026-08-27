import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
RQ1_DIR = REPO_ROOT / "LLM/results/step400/rq1"
TRAIN16K_INDEP_INPUT = RQ1_DIR / "rq1_train16k_indep_metrics.json"
TRAIN16K_INPUT = RQ1_DIR / "rq1_train16k_metrics.json"
REAL_INPUT = RQ1_DIR / "rq1_128_metrics.json"
PROVISIONAL_INPUT = RQ1_DIR / "rq1_64_metrics.json"
DEFAULT_OUTPUT = REPO_ROOT / "paper/figures/rq1_gradient_aggregation.pdf"
DEFAULT_PNG_OUTPUT = REPO_ROOT / "paper/figures/rq1_gradient_aggregation.png"
DEFAULT_TRAJECTORY_OUTPUT = (
    REPO_ROOT / "paper/figures/rq1_trajectory_level_aggregation.pdf"
)
DEFAULT_TRAJECTORY_PNG_OUTPUT = (
    REPO_ROOT / "paper/figures/rq1_trajectory_level_aggregation.png"
)

K_VALUES = (0, 1, 3, 7, 15)
TRAJECTORY_M_VALUES = (2, 4, 8)
METHODS = ("ttpg", "naive")

METHOD_LABELS = {"ttpg": "TTPG", "naive": "NaivePG"}
METHOD_COLORS = {"ttpg": "#B5475D", "naive": "#6F7F8C"}
METHOD_LINESTYLES = {"ttpg": "-", "naive": (0, (4, 2.2))}
TRAJECTORY_M_MARKERS = {2: "o", 4: "s", 8: "^"}

TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#59636B"
SPINE_COLOR = "#A5ADB3"
GRID_COLOR = "#CBD1D6"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot RQ1 gradient bias, variance, MSE, and cosine similarity. "
            "The default uses the completed independent 8+8-group metrics on "
            "16k training prompts, then falls back to earlier metrics."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics JSON. Default: rq1_train16k_indep_metrics.json if present.",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Top-level metric key. Default: mean128 for 128 groups, otherwise mean.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG_OUTPUT)
    parser.add_argument(
        "--trajectory-metrics",
        type=Path,
        default=PROVISIONAL_INPUT,
        help="64-group JSON containing the measured trajectory-level 'sum' metrics.",
    )
    parser.add_argument(
        "--trajectory-output", type=Path, default=DEFAULT_TRAJECTORY_OUTPUT
    )
    parser.add_argument(
        "--trajectory-png-output", type=Path, default=DEFAULT_TRAJECTORY_PNG_OUTPUT
    )
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Generate only the token-level main figure and leave appendix outputs unchanged.",
    )
    return parser.parse_args()


def discover_input(explicit_path):
    if explicit_path is not None:
        return explicit_path
    if TRAIN16K_INDEP_INPUT.exists():
        return TRAIN16K_INDEP_INPUT
    if TRAIN16K_INPUT.exists():
        return TRAIN16K_INPUT
    if REAL_INPUT.exists():
        return REAL_INPUT
    if PROVISIONAL_INPUT.exists():
        return PROVISIONAL_INPUT
    raise FileNotFoundError(
        f"None of {TRAIN16K_INDEP_INPUT}, {TRAIN16K_INPUT}, {REAL_INPUT}, "
        f"or {PROVISIONAL_INPUT} exists"
    )


def read_payload(path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def choose_metric_key(payload, explicit_key, path):
    if explicit_key is not None:
        if explicit_key not in payload:
            raise KeyError(f"Missing top-level key {explicit_key!r} in {path}")
        return explicit_key
    if isinstance(payload.get("per_k"), dict):
        return None
    for key in ("mean128", "mean"):
        if key in payload:
            return key
    if len(payload) == 1:
        return next(iter(payload))
    raise ValueError(
        f"Could not select token-mean metrics from top-level keys {sorted(payload)}"
    )


def require_finite(value, label, *, nonnegative=False, cosine=False):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}: {value}")
    if nonnegative and value < 0.0:
        raise ValueError(f"Expected nonnegative {label}, found {value}")
    if cosine and not -1.0 <= value <= 1.0:
        raise ValueError(f"Expected cosine in [-1, 1] for {label}, found {value}")
    return value


def load_series(path, metric_key=None):
    payload = read_payload(path)
    selected_key = choose_metric_key(payload, metric_key, path)
    metrics = payload if selected_key is None else payload[selected_key]
    if not isinstance(metrics, dict) or not isinstance(metrics.get("per_k"), dict):
        raise ValueError(f"Missing per_k metrics under {selected_key!r} in {path}")

    per_k = metrics["per_k"]
    if set(per_k) != {str(k) for k in K_VALUES}:
        raise ValueError(
            f"Expected K={list(K_VALUES)} in {path}, found {sorted(per_k)}"
        )

    first_per_m = per_k[str(K_VALUES[0])]["ttpg"].get("per_m", {})
    source_m_values = tuple(sorted(int(value) for value in first_per_m))
    if source_m_values in ((1, 2, 4), (4, 8, 16)):
        provisional = False
        target_m_values = source_m_values
    elif source_m_values == (2, 4, 8):
        provisional = True
        target_m_values = (4, 8, 16)
    else:
        raise ValueError(
            f"Expected M={{1,2,4}}, M={{4,8,16}}, or provisional M={{2,4,8}} in {path}; "
            f"found {source_m_values}"
        )

    series = {
        method: {
            "bias": [],
            "var": {m: [] for m in target_m_values},
            "mse": {m: [] for m in target_m_values},
            "cos": {m: [] for m in target_m_values},
        }
        for method in METHODS
    }

    for k in K_VALUES:
        entry = per_k[str(k)]
        if set(entry) != set(METHODS):
            raise ValueError(
                f"Expected methods {sorted(METHODS)} at K={k}, found {sorted(entry)}"
            )
        for method in METHODS:
            method_entry = entry[method]
            bias = require_finite(
                method_entry["bias"],
                f"{method}/bias/K={k}",
                nonnegative=True,
            )
            series[method]["bias"].append(bias)
            per_m = method_entry.get("per_m", {})

            if not provisional:
                for target_m in target_m_values:
                    source_entry = per_m.get(str(target_m))
                    if not isinstance(source_entry, dict):
                        raise KeyError(
                            f"Missing {method}/K={k}/M={target_m} in {path}"
                        )
                    for metric in ("var", "mse", "cos"):
                        series[method][metric][target_m].append(
                            require_finite(
                                source_entry[metric],
                                f"{method}/{metric}/K={k}/M={target_m}",
                                nonnegative=metric in ("var", "mse"),
                                cosine=metric == "cos",
                            )
                        )
                continue

            # Provisional 64 -> 128 extrapolation. M=4 and M=8 reuse the
            # measured statistics at the same aggregation size. For M=16,
            # gradient variance follows the observed 1/M scaling, and cosine
            # is linearly extrapolated in 1/M from M=4 and M=8. MSE is then
            # reconstructed from the exact finite-R decomposition
            # MSE = ((R - 1) / R) Var + Bias^2 with R = 128 / M.
            old_entries = {}
            for source_m in (2, 4, 8):
                source_entry = per_m.get(str(source_m))
                if not isinstance(source_entry, dict):
                    raise KeyError(
                        f"Missing {method}/K={k}/M={source_m} in {path}"
                    )
                old_entries[source_m] = {
                    "var": require_finite(
                        source_entry["var"],
                        f"{method}/var/K={k}/M={source_m}",
                        nonnegative=True,
                    ),
                    "cos": require_finite(
                        source_entry["cos"],
                        f"{method}/cos/K={k}/M={source_m}",
                        cosine=True,
                    ),
                }

            extrapolated_var = {
                4: old_entries[4]["var"],
                8: old_entries[8]["var"],
                16: old_entries[8]["var"] / 2.0,
            }
            extrapolated_cos = {
                4: old_entries[4]["cos"],
                8: old_entries[8]["cos"],
                16: min(
                    1.0,
                    max(
                        -1.0,
                        1.5 * old_entries[8]["cos"]
                        - 0.5 * old_entries[4]["cos"],
                    ),
                ),
            }
            for target_m in target_m_values:
                repeats = 128 // target_m
                variance = extrapolated_var[target_m]
                mse = (repeats - 1) / repeats * variance + bias**2
                series[method]["var"][target_m].append(variance)
                series[method]["mse"][target_m].append(mse)
                series[method]["cos"][target_m].append(
                    extrapolated_cos[target_m]
                )

    return series, selected_key, provisional, target_m_values


def load_measured_series(path, metric_key, m_values):
    payload = read_payload(path)
    if metric_key not in payload:
        raise KeyError(f"Missing top-level key {metric_key!r} in {path}")
    metrics = payload[metric_key]
    if not isinstance(metrics, dict) or not isinstance(metrics.get("per_k"), dict):
        raise ValueError(f"Missing per_k metrics under {metric_key!r} in {path}")

    per_k = metrics["per_k"]
    if set(per_k) != {str(k) for k in K_VALUES}:
        raise ValueError(
            f"Expected K={list(K_VALUES)} in {path}, found {sorted(per_k)}"
        )

    series = {
        method: {
            "bias": [],
            "var": {m: [] for m in m_values},
            "mse": {m: [] for m in m_values},
            "cos": {m: [] for m in m_values},
        }
        for method in METHODS
    }
    for k in K_VALUES:
        entry = per_k[str(k)]
        if set(entry) != set(METHODS):
            raise ValueError(
                f"Expected methods {sorted(METHODS)} at K={k}, found {sorted(entry)}"
            )
        for method in METHODS:
            method_entry = entry[method]
            series[method]["bias"].append(
                require_finite(
                    method_entry["bias"],
                    f"{metric_key}/{method}/bias/K={k}",
                    nonnegative=True,
                )
            )
            per_m = method_entry.get("per_m", {})
            for m in m_values:
                measured = per_m.get(str(m))
                if not isinstance(measured, dict):
                    raise KeyError(
                        f"Missing {metric_key}/{method}/K={k}/M={m} in {path}"
                    )
                for metric in ("var", "mse", "cos"):
                    series[method][metric][m].append(
                        require_finite(
                            measured[metric],
                            f"{metric_key}/{method}/{metric}/K={k}/M={m}",
                            nonnegative=metric in ("var", "mse"),
                            cosine=metric == "cos",
                        )
                    )
                bias = series[method]["bias"][-1]
                mse = series[method]["mse"][m][-1]
                if mse + 1e-10 < bias**2:
                    raise ValueError(
                        f"Inconsistent {metric_key}/{method}/K={k}/M={m}: "
                        f"MSE={mse} is smaller than Bias^2={bias**2}"
                    )
    return series


def style_axis(axis):
    axis.set_xlim(-0.18, len(K_VALUES) - 0.82)
    axis.set_xticks(range(len(K_VALUES)))
    axis.set_xticklabels([str(value) for value in K_VALUES])
    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linestyle=(0, (3, 3)),
        linewidth=0.55,
        alpha=0.75,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(SPINE_COLOR)
        axis.spines[side].set_linewidth(0.65)
    axis.tick_params(
        axis="both",
        colors=MUTED_TEXT_COLOR,
        labelsize=7.7,
        length=2.5,
        width=0.6,
        pad=2.0,
    )


def plot_method_curve(axis, values, method, marker=None, zorder=3):
    marker_face = METHOD_COLORS[method] if method == "ttpg" else "white"
    axis.plot(
        range(len(K_VALUES)),
        values,
        color=METHOD_COLORS[method],
        linestyle=METHOD_LINESTYLES[method],
        linewidth=1.55 if method == "ttpg" else 1.45,
        marker=marker,
        markersize=4.0 if marker else 0.0,
        markerfacecolor=marker_face,
        markeredgecolor=METHOD_COLORS[method],
        markeredgewidth=0.8,
        solid_capstyle="round",
        zorder=zorder,
    )


def semantic_legend_handles(m_values, m_markers):
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=1.65,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            color="#56616A",
            linestyle="None",
            marker=m_markers[m],
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=rf"$M={m}$",
        )
        for m in m_values
    ]
    return method_handles + marker_handles


def configure_metric_axis(axis, metric, use_log_scale):
    if use_log_scale:
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}")
        )
        return

    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    if metric in ("bias", "mse", "cos"):
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def plot_figure(
    series,
    m_values,
    m_markers,
    log_metrics=("var",),
):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 4, figsize=(7.45, 2.45), squeeze=False)
    axes = axes[0]
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.210,
        top=0.765,
        wspace=0.36,
    )

    titles = (
        r"(a) Relative Bias $\downarrow$",
        r"(b) Relative Variance $\downarrow$",
        r"(c) Relative MSE $\downarrow$",
        r"(d) Cosine Similarity $\uparrow$",
    )
    for axis, title in zip(axes, titles):
        style_axis(axis)
        axis.set_title(title, pad=6.0)

    # The mean across all non-overlapping blocks is invariant to M, so the
    # bias panel contains one curve per aggregation rule rather than three
    # perfectly overlapping copies of each curve.
    for method in METHODS:
        plot_method_curve(
            axes[0],
            series[method]["bias"],
            method,
            marker=None,
            zorder=5 if method == "ttpg" else 3,
        )

    for axis, metric in zip(axes[1:], ("var", "mse", "cos")):
        for target_m in m_values:
            for method in METHODS:
                plot_method_curve(
                    axis,
                    series[method][metric][target_m],
                    method,
                    marker=m_markers[target_m],
                    zorder=5 if method == "ttpg" else 3,
                )

    for axis, metric in zip(axes, ("bias", "var", "mse", "cos")):
        configure_metric_axis(axis, metric, metric in log_metrics)

    figure.legend(
        handles=semantic_legend_handles(m_values, m_markers),
        loc="upper center",
        bbox_to_anchor=(0.525, 0.945),
        ncol=5,
        frameon=False,
        fontsize=8.2,
        handlelength=2.4,
        handletextpad=0.55,
        columnspacing=1.45,
    )
    figure.text(
        0.525,
        0.105,
        r"Additional branches $K$",
        ha="center",
        va="center",
        color=MUTED_TEXT_COLOR,
        fontsize=8.4,
    )
    return figure


def print_summary(series, path, selected_key, provisional, m_values):
    if provisional:
        source = "provisional 64-group extrapolation"
    elif selected_key is None:
        source = "direct metrics"
    else:
        source = f"metrics key {selected_key}"
    print(f"Loaded {source}: {path}", flush=True)
    print(
        "Bias K=0->15: "
        f"NaivePG {series['naive']['bias'][0]:.4f}->{series['naive']['bias'][-1]:.4f}; "
        f"TTPG {series['ttpg']['bias'][0]:.4f}->{series['ttpg']['bias'][-1]:.4f}",
        flush=True,
    )
    for target_m in m_values:
        naive = series["naive"]["mse"][target_m][-1]
        ttpg = series["ttpg"]["mse"][target_m][-1]
        print(
            f"K=15, M={target_m}: MSE NaivePG={naive:.4f}, TTPG={ttpg:.4f}",
            flush=True,
        )


def print_trajectory_summary(series, path):
    print(f"Loaded measured trajectory-level metrics: {path} [sum]", flush=True)
    print(
        "Trajectory-level bias K=0->15: "
        f"NaivePG {series['naive']['bias'][0]:.4f}->{series['naive']['bias'][-1]:.4f}; "
        f"TTPG {series['ttpg']['bias'][0]:.4f}->{series['ttpg']['bias'][-1]:.4f}",
        flush=True,
    )
    print(
        "Trajectory-level K=15, M=8 MSE: "
        f"NaivePG={series['naive']['mse'][8][-1]:.4f}, "
        f"TTPG={series['ttpg']['mse'][8][-1]:.4f}",
        flush=True,
    )


def save_figure(figure, pdf_path, png_path, label):
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {label} PDF: {pdf_path}", flush=True)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            png_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved {label} PNG: {png_path}", flush=True)
    plt.close(figure)


def main():
    args = parse_args()
    metrics_path = discover_input(args.metrics)
    series, selected_key, provisional, main_m_values = load_series(
        metrics_path, args.metric_key
    )
    main_m_markers = dict(zip(main_m_values, ("o", "s", "^")))
    print_summary(series, metrics_path, selected_key, provisional, main_m_values)

    figure = plot_figure(
        series,
        m_values=main_m_values,
        m_markers=main_m_markers,
    )
    save_figure(figure, args.output, args.png_output, "RQ1 token-level")

    if args.skip_trajectory:
        return

    trajectory_series = load_measured_series(
        args.trajectory_metrics,
        metric_key="sum",
        m_values=TRAJECTORY_M_VALUES,
    )
    print_trajectory_summary(trajectory_series, args.trajectory_metrics)
    trajectory_figure = plot_figure(
        trajectory_series,
        m_values=TRAJECTORY_M_VALUES,
        m_markers=TRAJECTORY_M_MARKERS,
        log_metrics=("bias", "var", "mse"),
    )
    save_figure(
        trajectory_figure,
        args.trajectory_output,
        args.trajectory_png_output,
        "RQ1 trajectory-level",
    )


if __name__ == "__main__":
    main()
