import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, LogLocator, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[3]
RQ1_DIR = REPO_ROOT / "LLM/results/step400/rq1"
TRAIN16K_GLOBAL32_INPUT = RQ1_DIR / "rq1_train16k_global32_metrics.json"
DEFAULT_OUTPUT = REPO_ROOT / "paper/figures/rq1_gradient_aggregation.pdf"
DEFAULT_PNG_OUTPUT = REPO_ROOT / "paper/figures/rq1_gradient_aggregation.png"

K_VALUES = (0, 1, 3, 7, 15)
METHODS = ("ttpg", "naive")

METHOD_LABELS = {"ttpg": "TTPG", "naive": "NaivePG"}
METHOD_COLORS = {"ttpg": "#B5475D", "naive": "#6F7F8C"}
METHOD_LINESTYLES = {"ttpg": "-", "naive": (0, (4, 2.2))}

TEXT_COLOR = "#263238"
MUTED_TEXT_COLOR = "#59636B"
SPINE_COLOR = "#A5ADB3"
GRID_COLOR = "#CBD1D6"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot RQ1 gradient bias, variance, MSE, and cosine similarity. "
            "The default uses the global token-level metrics on 16k training prompts."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics JSON. Default: rq1_train16k_global32_metrics.json if present.",
    )
    parser.add_argument(
        "--metric-key",
        default=None,
        help="Top-level metric key. Default: mean128 for 128 groups, otherwise mean.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--png-output", type=Path, default=DEFAULT_PNG_OUTPUT)
    return parser.parse_args()


def discover_input(explicit_path):
    if explicit_path is not None:
        return explicit_path
    if TRAIN16K_GLOBAL32_INPUT.exists():
        return TRAIN16K_GLOBAL32_INPUT
    raise FileNotFoundError(
        f"Global token-level metrics not found: {TRAIN16K_GLOBAL32_INPUT}. "
        "Pass --metrics explicitly to inspect another compatible payload."
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
        target_m_values = source_m_values
    else:
        raise ValueError(
            f"Expected M={{1,2,4}} or M={{4,8,16}} in {path}; "
            f"found {source_m_values}"
        )

    series = {
        method: {
            "bias": {m: [] for m in target_m_values},
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
            per_m = method_entry.get("per_m", {})
            for target_m in target_m_values:
                source_entry = per_m.get(str(target_m))
                if not isinstance(source_entry, dict):
                    raise KeyError(
                        f"Missing {method}/K={k}/M={target_m} in {path}"
                    )
                # Under global token normalization, block ratios do not commute
                # with averaging, so bias is M-dependent. Older payloads store
                # one M-invariant bias at method level; replicate it only when
                # reading those legacy results.
                bias_value = source_entry.get("bias", method_entry.get("bias"))
                if bias_value is None:
                    raise KeyError(
                        f"Missing {method}/bias/K={k}/M={target_m} in {path}"
                    )
                series[method]["bias"][target_m].append(
                    require_finite(
                        bias_value,
                        f"{method}/bias/K={k}/M={target_m}",
                        nonnegative=True,
                    )
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

    return series, selected_key, target_m_values


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

    for axis, metric in zip(axes, ("bias", "var", "mse", "cos")):
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


def print_summary(series, path, selected_key, m_values):
    if selected_key is None:
        source = "direct metrics"
    else:
        source = f"metrics key {selected_key}"
    print(f"Loaded {source}: {path}", flush=True)
    for target_m in m_values:
        print(
            f"Bias M={target_m}, K=0->15: "
            f"NaivePG {series['naive']['bias'][target_m][0]:.4f}->"
            f"{series['naive']['bias'][target_m][-1]:.4f}; "
            f"TTPG {series['ttpg']['bias'][target_m][0]:.4f}->"
            f"{series['ttpg']['bias'][target_m][-1]:.4f}",
            flush=True,
        )
        naive = series["naive"]["mse"][target_m][-1]
        ttpg = series["ttpg"]["mse"][target_m][-1]
        print(
            f"K=15, M={target_m}: MSE NaivePG={naive:.4f}, TTPG={ttpg:.4f}",
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
    series, selected_key, main_m_values = load_series(metrics_path, args.metric_key)
    main_m_markers = dict(zip(main_m_values, ("o", "s", "^")))
    print_summary(series, metrics_path, selected_key, main_m_values)

    figure = plot_figure(
        series,
        m_values=main_m_values,
        m_markers=main_m_markers,
    )
    save_figure(figure, args.output, args.png_output, "RQ1 token-level")



if __name__ == "__main__":
    main()
