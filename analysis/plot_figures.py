"""Reproduce all 12 paper figures from the CSVs in ``aggregate_data/``.

Run from the repository root:
    python analysis/plot_figures.py

Each figure is written directly to ``figures/`` as a 100 dpi PNG and a
175 dpi PDF.
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Configuration

_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(_ROOT, "aggregate_data")
FIGURES_DIR = os.path.join(_ROOT, "figures")

# The 9 evaluation datasets, in the paper's order.
DATASETS = [
    "sat",
    "matmul",
    "sudoku",
    "aime",
    "csqa",
    "gpqa",
    "gsm",
    "mmlu_social_sciences",
    "mmlu_stem",
]
# Display labels used in figure 7 task annotations and figure 9 titles.
DATASET_LABEL = {
    "aime": "AIME",
    "csqa": "CSQA",
    "gpqa": "GPQA",
    "gsm": "GSM8K",
    "matmul": "Matrix Mult.",
    "mmlu_social_sciences": "MMLU (SS)",
    "mmlu_stem": "MMLU (STEM)",
    "sat": "3SAT",
    "sudoku": "Sudoku",
}
# In figure 7, math/logic tasks are colored red, others blue.
PRIMARY_TASKS = {"AIME", "GSM8K", "3SAT", "Sudoku"}
PRIMARY_COLOR, OTHER_COLOR = "#d62728", "#1f77b4"

# Family color scheme
FAMILY_COLORS = {
    "Qwen3-Base": "#87CEEB",
    "Qwen3": "#1f77b4",
    "Qwen2.5-Base": "#FFB347",
    "Qwen2.5": "#d62728",
    "Llama3-Base": "#A5D6A7",
    "Llama3": "#2ca02c",
    "DeepSeek": "#9467bd",
}

# Model sizes in billions of parameters, by nickname (used in figure 9).
MODEL_SIZES = {
    "Qwen3-0.6B-Base": 0.6,
    "Qwen3-1.7B-Base": 1.7,
    "Qwen3-4B-Base": 4.0,
    "Qwen3-8B-Base": 8.2,
    "Qwen3-14B-Base": 14.8,
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.2,
    "Qwen3-14B": 14.8,
    "Qwen3-32B": 32.8,
    "Qwen2.5-0.5B-Base": 0.5,
    "Qwen2.5-1.5B-Base": 1.5,
    "Qwen2.5-3B-Base": 3.1,
    "Qwen2.5-7B-Base": 7.6,
    "Qwen2.5-14B-Base": 14.8,
    "Qwen2.5-32B-Base": 32.8,
    "Qwen2.5-72B-Base": 72.7,
    "Qwen2.5-0.5B": 0.5,
    "Qwen2.5-1.5B": 1.5,
    "Qwen2.5-3B": 3.1,
    "Qwen2.5-7B": 7.6,
    "Qwen2.5-14B": 14.8,
    "Qwen2.5-32B": 32.8,
    "Qwen2.5-72B": 72.7,
    "Llama3-1B-Base": 1.23,
    "Llama3-3B-Base": 3.21,
    "Llama3-8B-Base": 8.03,
    "Llama3-70B-Base": 70.6,
    "Llama3-1B": 1.23,
    "Llama3-3B": 3.21,
    "Llama3-8B": 8.03,
    "Llama3-70B": 70.6,
    "DeepSeek-1.5B": 1.54,
    "DeepSeek-7B": 7.62,
    "DeepSeek-14B": 14.8,
    "DeepSeek-32B": 32.8,
}


def _read_csv(path):
    """Read a CSV file into a list of rows."""
    with open(path) as f:
        return list(csv.reader(f))


def load_matrix(metric, dataset):
    """Read aggregate_data/<metric>/<dataset>.csv as labels and a metric matrix."""
    rows = _read_csv(f"{DATA_DIR}/{metric}/{dataset}.csv")
    col_labels = rows[0][1:]
    row_labels = [r[0] for r in rows[1:]]
    matrix = np.array([[float(x) for x in r[1:]] for r in rows[1:]])
    return row_labels, col_labels, matrix


def load_scalar(metric, dataset):
    """Read aggregate_data/<metric>/<dataset>.csv as labels and values."""
    rows = _read_csv(f"{DATA_DIR}/{metric}/{dataset}.csv")
    labels = [r[0] for r in rows[1:]]
    values = np.array([float(r[1]) for r in rows[1:]])
    return labels, values


def matrix_avg_across_datasets(metric):
    """Return (row_labels, col_labels, mean_matrix) averaged over the 9 datasets."""
    matrices, rows, cols = [], None, None
    for ds in DATASETS:
        r, c, m = load_matrix(metric, ds)
        if rows is None:
            rows, cols = r, c
        elif r != rows or c != cols or m.shape != matrices[0].shape:
            raise ValueError(f"Model labels or matrix shape differ for {metric}/{ds}")
        matrices.append(m)
    return rows, cols, np.mean(matrices, axis=0)


def scalar_avg_across_datasets(metric):
    """Return (labels, values) of a per-model metric averaged over the 9 datasets."""
    labels, vals = None, []
    for ds in DATASETS:
        lab, v = load_scalar(metric, ds)
        if labels is None:
            labels = lab
        elif lab != labels or v.shape != vals[0].shape:
            raise ValueError(f"Model labels or vector shape differ for {metric}/{ds}")
        vals.append(v)
    return labels, np.mean(vals, axis=0)


def categorize_model(nick):
    """Return (family, "Base" or "PostTrained") for a model nickname."""
    lo = nick.lower()
    if "deepseek" in lo:
        return "DeepSeek", "PostTrained"
    fam = (
        "Llama3"
        if "llama" in lo
        else "Qwen3"
        if "qwen3" in lo
        else "Qwen2.5"
        if "qwen2.5" in lo
        else None
    )
    if fam is None:
        raise ValueError(f"unknown family: {nick}")
    tt = "Base" if "base" in lo else "PostTrained"
    return fam, tt


def color_for_model(nick):
    """Return the family color for a model, in its lighter shade for base models."""
    fam, tt = categorize_model(nick)
    return FAMILY_COLORS[f"{fam}-Base"] if tt == "Base" else FAMILY_COLORS[fam]


def family_legend(model_names, alpha=0.65, label_fn=None):
    """Deduplicated legend Patches by (family, training_type), in first-seen order."""
    if label_fn is None:

        def label_fn(fam, tt):
            """Legend label: the family name, with "-Base" appended for base models."""
            return f"{fam}-Base" if tt == "Base" else fam

    out, seen = [], set()
    for nick in model_names:
        fam, tt = categorize_model(nick)
        key = (fam, tt)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Patch(facecolor=color_for_model(nick), alpha=alpha, label=label_fn(fam, tt))
        )
    return out


def bold_legend(legend):
    """Set every legend label in bold."""
    for t in legend.get_texts():
        t.set_fontweight("bold")


def bold_ticks(ax, labelsize, which="both"):
    """Set the tick label size and bold the tick labels on the chosen axes."""
    ax.tick_params(axis=which, which="major", labelsize=labelsize)
    if which in ("both", "x"):
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
    if which in ("both", "y"):
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")


def format_y_2dp(ax):
    """Show y-axis tick labels with two decimal places."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.2f}"))


def save_fig(fig, name):
    """Save a figure to figures/ as a 100 dpi PNG and a 175 dpi PDF, then close it."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    fig.savefig(
        pdf_path,
        dpi=175,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def per_verifier_avg(matrix, row_labels, col_labels, setting):
    """For each verifier (col), average the metric over the chosen solver-row set.

    setting in {'self', 'intra', 'cross'}. 'intra' excludes the diagonal.
    Returns 1D array of length len(col_labels).
    """
    out = np.empty(len(col_labels))
    for v_idx, v in enumerate(col_labels):
        vf, vt = categorize_model(v)
        if setting == "self":
            out[v_idx] = matrix[row_labels.index(v), v_idx]
            continue
        vals = []
        for s_idx, s in enumerate(row_labels):
            if s == v:
                continue
            sf, st = categorize_model(s)
            same_fam_type = sf == vf and st == vt
            if (setting == "intra" and same_fam_type) or (
                setting == "cross" and not same_fam_type
            ):
                vals.append(matrix[s_idx, v_idx])
        out[v_idx] = np.mean(vals) if vals else np.nan
    return out


def pairs_by_setting(row_labels, col_labels, setting):
    """Yield (s_idx, v_idx) pairs for the given setting."""
    for s_idx, s in enumerate(row_labels):
        for v_idx, v in enumerate(col_labels):
            sf, st = categorize_model(s)
            vf, vt = categorize_model(v)
            is_self = s == v
            is_intra = not is_self and sf == vf and st == vt
            is_cross = not is_self and not is_intra
            if (
                (setting == "self" and is_self)
                or (setting == "intra" and is_intra)
                or (setting == "cross" and is_cross)
            ):
                yield s_idx, v_idx


def figure_1():
    """Figure 1: solver accuracy of all 37 models, averaged over the 9 datasets."""
    labels, vals = scalar_avg_across_datasets("solver_accuracy")
    colors = [color_for_model(m) for m in labels]

    fig, ax = plt.subplots(1, 1, figsize=(20, 2.5))
    ax.bar(range(len(labels)), vals, color=colors, alpha=0.65)
    ax.set_ylabel(
        "Solver\nAccuracy ↑", fontsize=24, fontweight="bold", rotation=90, labelpad=20
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")
    bold_ticks(ax, labelsize=15, which="y")
    format_y_2dp(ax)

    legend = fig.legend(
        handles=family_legend(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=7,
        fontsize=20,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    save_fig(fig, "figure_01_solver_accuracy_by_model")


def figure_2():
    """Figure 2: predicted verifier gain against the empirical gain of rejection sampling (12 models)."""
    # Empirical gain CSVs are already 12 x 12 with the right model order.
    rs_models, _, _ = load_matrix("empirical_gain_k3", DATASETS[0])
    _, _, vg_full_rows = matrix_avg_across_datasets("verifier_gain")
    full_rows, _, _ = load_matrix("verifier_gain", DATASETS[0])
    idx = [full_rows.index(m) for m in rs_models]
    vg_12 = vg_full_rows[np.ix_(idx, idx)]
    _, _, eg3 = matrix_avg_across_datasets("empirical_gain_k3")
    _, _, eg10 = matrix_avg_across_datasets("empirical_gain_k10")

    fig, axes = plt.subplots(1, 2, figsize=(6 * 2, 6 * 1.15))
    for col, (title, eg_mat) in enumerate(
        [("Empirical Gain (3 Attempts)", eg3), ("Empirical Gain (10 Attempts)", eg10)]
    ):
        ax = axes[col]
        x_vals, y_vals, colors = [], [], []
        for s_idx in range(len(rs_models)):
            for v_idx in range(len(rs_models)):
                x_vals.append(vg_12[s_idx, v_idx])
                y_vals.append(eg_mat[s_idx, v_idx])
                colors.append(color_for_model(rs_models[v_idx]))
        ax.scatter(x_vals, y_vals, s=80, alpha=0.6, c=colors)

        r = np.corrcoef(x_vals, y_vals)[0, 1]
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            "--",
            color="gray",
            alpha=0.9,
            linewidth=3,
        )
        ax.text(
            0.02,
            0.98,
            f"r = {r:.3f}\nslope = {slope:.3f}",
            transform=ax.transAxes,
            fontsize=17.5,
            fontweight="bold",
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
        )

        ax.set_title(title, fontsize=22.5, fontweight="bold", pad=8)
        ax.set_xlabel("Verifier Gain", fontsize=22.5, fontweight="bold", labelpad=10)
        if col == 0:
            ax.set_ylabel(
                "Rejection Sampling Gain",
                fontsize=22.5,
                fontweight="bold",
                rotation=90,
                labelpad=10,
            )
            format_y_2dp(ax)
        else:
            ax.set_yticklabels([])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.2f}"))
        ax.grid(True, alpha=0.3)
        bold_ticks(ax, labelsize=15)
        ax.set_box_aspect(1)

    legend_models = sorted(set(rs_models), key=rs_models.index)
    legend = fig.legend(
        handles=family_legend(legend_models, alpha=0.6, label_fn=lambda f, t: f),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        fontsize=22.5,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, wspace=0.08)
    save_fig(fig, "figure_02_predicted_vs_empirical_gain")


def _scatter_vs_solver_accuracy(metric_configs, output_name, compact=False):
    """Shared scatter renderer for Figures 3 and 10.

    metric_configs: list of (display_label, function_taking_matrices_returning_y_per_verifier).
    compact: use the smaller-font / single-line-xlabel layout used to render
        Figure 10.
    """
    settings = [
        ("Self-Verification", "self"),
        ("Intra-Family Verification", "intra"),
        ("Cross-Family Verification", "cross"),
    ]
    # Font sizes, label pads, and margins: compact for Figure 10, large for Figure 3.
    if compact:
        stats_fs, ylabel_fs, title_fs, xlabel_fs = 26, 40, 40, 35
        ylabel_pad, tick_fs, legend_fs = 15, 28.5, 40
        legend_y_anchor, bottom_margin = 0.05, 0.08
        xlabel_text = "Verifier's Own Solver Acc."
    else:
        stats_fs, ylabel_fs, title_fs, xlabel_fs = 32, 55, 55, 55
        ylabel_pad, tick_fs, legend_fs = 25, 40, 55
        legend_y_anchor, bottom_margin = 0.04, 0.10
        xlabel_text = "Verifier's Own\nSolver Acc."

    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])
    post_idx = [
        i for i, m in enumerate(rows37) if categorize_model(m)[1] == "PostTrained"
    ]
    post_names = [rows37[i] for i in post_idx]

    avg_mats = {}
    for metric_key in (
        "verifier_gain",
        "verifier_accuracy",
        "verifier_fpr",
        "verifier_fnr",
    ):
        _, _, m = matrix_avg_across_datasets(metric_key)
        avg_mats[metric_key] = m[np.ix_(post_idx, post_idx)]

    _, sa_last = load_scalar("solver_accuracy", DATASETS[-1])
    solver_acc_post = sa_last[post_idx]
    colors = [color_for_model(m) for m in post_names]

    subplot = 9
    n_rows, n_cols = len(metric_configs), 3
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(subplot * n_cols, subplot * n_rows)
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, (metric_label, get_y) in enumerate(metric_configs):
        per_setting_ys = []
        for _, setting in settings:
            per_setting_ys.append(get_y(avg_mats, post_names, setting))
        y_all = np.concatenate(per_setting_ys)
        ymin, ymax = y_all.min(), y_all.max()
        ymargin = (ymax - ymin) * 0.05 if ymax != ymin else 0.05

        for col, ((setting_label, _setting), ys) in enumerate(
            zip(settings, per_setting_ys, strict=False)
        ):
            ax = axes[row, col]
            ax.scatter(solver_acc_post, ys, c=colors, alpha=0.6, s=270)

            r = np.corrcoef(solver_acc_post, ys)[0, 1]
            slope, intercept = np.polyfit(solver_acc_post, ys, 1)
            x_line = np.linspace(solver_acc_post.min(), solver_acc_post.max(), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                "--",
                color="gray",
                alpha=0.9,
                linewidth=5,
            )
            ax.text(
                0.02,
                0.98,
                f"r = {r:.3f}\nslope = {slope:.3f}",
                transform=ax.transAxes,
                fontsize=stats_fs,
                fontweight="bold",
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
            )

            if row == 0:
                title = (
                    setting_label
                    if compact
                    else setting_label.replace(
                        "-Family Verification", "-Family\nVerification"
                    )
                )
                ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=15)
            if col == 0:
                ax.set_ylabel(
                    metric_label,
                    fontsize=ylabel_fs,
                    fontweight="bold",
                    rotation=90,
                    labelpad=ylabel_pad,
                )
                format_y_2dp(ax)
            else:
                ax.set_yticklabels([])
            if row == n_rows - 1:
                ax.set_xlabel(
                    xlabel_text, fontsize=xlabel_fs, fontweight="bold", labelpad=15
                )
            else:
                ax.set_xticklabels([])

            ax.set_ylim(ymin - ymargin, ymax + ymargin)
            ax.grid(True, alpha=0.3)
            bold_ticks(ax, labelsize=tick_fs)

    legend = fig.legend(
        handles=family_legend(post_names, alpha=0.6, label_fn=lambda f, t: f),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        ncol=4,
        fontsize=legend_fs,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom_margin)
    save_fig(fig, output_name)


def _avg_y_for_setting(avg_mats, post_names, setting, metric_key):
    """y[verifier] = avg of metric over solvers in `setting`."""
    return per_verifier_avg(avg_mats[metric_key], post_names, post_names, setting)


def figure_3():
    """Figure 3: verifier accuracy, FPR, FNR, and gain against solver accuracy, per verification setting."""
    cfg = [
        (
            "Verifier Accuracy ↑",
            lambda M, N, s: _avg_y_for_setting(M, N, s, "verifier_accuracy"),
        ),
        ("Verifier FPR ↓", lambda M, N, s: _avg_y_for_setting(M, N, s, "verifier_fpr")),
        ("Verifier FNR ↓", lambda M, N, s: _avg_y_for_setting(M, N, s, "verifier_fnr")),
        (
            "Verifier Gain ↑",
            lambda M, N, s: _avg_y_for_setting(M, N, s, "verifier_gain"),
        ),
    ]
    _scatter_vs_solver_accuracy(cfg, "figure_03_verifier_metrics_vs_solver_accuracy")


def figure_10():
    """Figure 10: verifier accuracy, F1, precision, and gain against solver accuracy."""
    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])
    post_idx = [
        i for i, m in enumerate(rows37) if categorize_model(m)[1] == "PostTrained"
    ]
    _, _, va = matrix_avg_across_datasets("verifier_accuracy")
    va = va[np.ix_(post_idx, post_idx)]
    _, _, vg = matrix_avg_across_datasets("verifier_gain")
    vg = vg[np.ix_(post_idx, post_idx)]
    _, _, vp = matrix_avg_across_datasets("verifier_precision")
    vp = vp[np.ix_(post_idx, post_idx)]
    _, _, vf = matrix_avg_across_datasets("verifier_f1")
    vf = vf[np.ix_(post_idx, post_idx)]

    cfg = [
        ("Verifier Accuracy ↑", lambda M, N, s: per_verifier_avg(va, N, N, s)),
        ("Verifier F1-Score ↑", lambda M, N, s: per_verifier_avg(vf, N, N, s)),
        ("Verifier Precision ↑", lambda M, N, s: per_verifier_avg(vp, N, N, s)),
        ("Verifier Gain ↑", lambda M, N, s: per_verifier_avg(vg, N, N, s)),
    ]
    _scatter_vs_solver_accuracy(
        cfg, "figure_10_verifier_f1_and_precision", compact=True
    )


def figure_4():
    """Figure 4: verifier metrics of the 21 post-trained models, averaged over the 9 datasets."""
    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])
    post_idx = [
        i for i, m in enumerate(rows37) if categorize_model(m)[1] == "PostTrained"
    ]
    post_names = [rows37[i] for i in post_idx]
    colors = [color_for_model(m) for m in post_names]

    metric_configs = [
        ("Verifier\nAccuracy ↑", "verifier_accuracy"),
        ("Verifier\nFPR ↓", "verifier_fpr"),
        ("Verifier\nFNR ↓", "verifier_fnr"),
        ("Verifier\nGain ↑", "verifier_gain"),
    ]
    settings = [
        ("Self-Verification", "self"),
        ("Intra-Family Verification", "intra"),
        ("Cross-Family Verification", "cross"),
    ]

    n_rows, n_cols = len(metric_configs), 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, max(6, 3 * n_rows)))

    for r, (m_label, m_key) in enumerate(metric_configs):
        _, _, full = matrix_avg_across_datasets(m_key)
        mat = full[np.ix_(post_idx, post_idx)]

        ys_per_setting = [
            per_verifier_avg(mat, post_names, post_names, s) for _, s in settings
        ]
        flat = np.concatenate(ys_per_setting)
        ymin, ymax = flat.min(), flat.max()
        ymargin = 0.05
        ylo = max(ymin - ymargin, 0) if ymin >= 0 else ymin - ymargin

        for c, ((s_label, _), ys) in enumerate(
            zip(settings, ys_per_setting, strict=False)
        ):
            ax = axes[r, c]
            ax.bar(range(len(post_names)), ys, color=colors, alpha=0.65)
            if c == 0:
                ax.set_ylabel(
                    m_label, fontsize=29, fontweight="bold", rotation=90, labelpad=15
                )
                format_y_2dp(ax)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(s_label, fontsize=38, fontweight="bold", pad=15)
            ax.set_xticks(range(len(post_names)))
            ax.set_xticklabels([])
            bold_ticks(ax, labelsize=25)
            ax.set_ylim(ylo, ymax + ymargin)
            ax.grid(True, alpha=0.3)

    legend = fig.legend(
        handles=family_legend(post_names, alpha=0.65, label_fn=lambda f, t: f),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        fontsize=38,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, wspace=0.05)
    save_fig(fig, "figure_04_verifier_metrics_by_model")


def _figure_similarity(similarity_metric, xlabel, output_name):
    """Shared renderer for Figures 5 and 11: verifier metrics against solver-verifier similarity."""
    sim_rows, _, _ = load_matrix(similarity_metric, DATASETS[0])  # 12 models
    _, _, sim = matrix_avg_across_datasets(similarity_metric)

    # Load verifier metrics for the 12-model subset (rows of sim define order).
    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])
    idx12 = [rows37.index(m) for m in sim_rows]
    metrics12 = {}
    for key in ("verifier_fpr", "verifier_fnr", "verifier_gain"):
        _, _, full = matrix_avg_across_datasets(key)
        metrics12[key] = full[np.ix_(idx12, idx12)]
    model_colors = [color_for_model(m) for m in sim_rows]

    settings = [
        ("Intra-Family Verification", "intra"),
        ("Cross-Family Verification", "cross"),
    ]
    metrics = [
        ("Verifier FPR ↓", "verifier_fpr"),
        ("Verifier FNR ↓", "verifier_fnr"),
        ("Verifier Gain ↑", "verifier_gain"),
    ]

    n_rows, n_cols = len(metrics), len(settings)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 9 * n_rows))

    per_col_x = [[] for _ in range(n_cols)]
    per_row_y = [[] for _ in range(n_rows)]
    cell_data = [[None] * n_cols for _ in range(n_rows)]
    for r, (_, mkey) in enumerate(metrics):
        for c, (_, setting) in enumerate(settings):
            xs, ys, cs = [], [], []
            for s_idx, v_idx in pairs_by_setting(sim_rows, sim_rows, setting):
                xs.append(sim[s_idx, v_idx])
                ys.append(metrics12[mkey][s_idx, v_idx])
                cs.append(model_colors[v_idx])
            cell_data[r][c] = (xs, ys, cs)
            per_col_x[c].extend(xs)
            per_row_y[r].extend(ys)

    x_limits = [
        (min(v) - (max(v) - min(v)) * 0.05, max(v) + (max(v) - min(v)) * 0.05)
        for v in per_col_x
    ]
    y_limits = [
        (min(v) - (max(v) - min(v)) * 0.05, max(v) + (max(v) - min(v)) * 0.05)
        for v in per_row_y
    ]

    for r, (mlabel, _) in enumerate(metrics):
        for c, (slabel, _) in enumerate(settings):
            ax = axes[r, c]
            xs, ys, cs = cell_data[r][c]
            ax.scatter(xs, ys, s=180, alpha=0.6, c=cs)
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                "--",
                color="gray",
                alpha=0.9,
                linewidth=5,
            )
            r_value = np.corrcoef(xs, ys)[0, 1]
            ax.text(
                0.02,
                0.98,
                f"r = {r_value:.3f}\nslope = {slope:.3f}",
                transform=ax.transAxes,
                fontsize=28.5,
                fontweight="bold",
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
            )

            if c == 0:
                ax.set_ylabel(
                    mlabel, fontsize=40, fontweight="bold", rotation=90, labelpad=20
                )
                format_y_2dp(ax)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(slabel, fontsize=40, fontweight="bold", pad=15)
            ax.set_xlim(*x_limits[c])
            ax.set_ylim(*y_limits[r])
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.2f}"))
            if r == n_rows - 1:
                ax.set_xlabel(xlabel, fontsize=40, fontweight="bold", labelpad=20)
                ax.tick_params(axis="both", which="major", labelsize=24)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
                ax.tick_params(axis="y", which="major", labelsize=24)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight("bold")

    multiline = "\n" in xlabel
    legend_y_anchor = 0.045 if multiline else 0.055
    legend = fig.legend(
        handles=family_legend(sim_rows, alpha=0.6, label_fn=lambda f, t: f),
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y_anchor),
        ncol=4,
        fontsize=36,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12 if multiline else 0.10)
    save_fig(fig, output_name)


def figure_5():
    """Figure 5: verifier metrics against the cosine similarity of solver and verifier outputs."""
    _figure_similarity(
        "similarity", "Similarity Score", "figure_05_verifier_metrics_vs_similarity"
    )


def figure_11():
    """Figure 11: verifier metrics against the verifier's log-likelihood of the solver's output."""
    _figure_similarity(
        "similarity_likelihood",
        "Generation\nLog-Likelihood",
        "figure_11_verifier_metrics_vs_likelihood_similarity",
    )


def _base_post_pairs(model_names):
    """Find matched (base_name, posttrained_name) pairs by stripping '-Base' suffix."""
    pairs = []
    for m in model_names:
        _fam, tt = categorize_model(m)
        if tt == "Base":
            sibling = m.replace("-Base", "")
            if sibling in model_names:
                pairs.append((m, sibling))
    return pairs


def figure_12():
    """Figure 12: solver accuracy of base and post-trained models, per family."""
    labels, vals = scalar_avg_across_datasets("solver_accuracy")
    pairs = _base_post_pairs(labels)
    pairs = [(b, p) for b, p in pairs if not categorize_model(b)[0].startswith("Llama")]
    by_fam = {}
    for b, p in pairs:
        fam = categorize_model(b)[0]
        by_fam.setdefault(fam, []).append(
            (vals[labels.index(b)], vals[labels.index(p)])
        )
    families = sorted(by_fam.keys())
    base_avg = [np.mean([v[0] for v in by_fam[f]]) for f in families]
    post_avg = [np.mean([v[1] for v in by_fam[f]]) for f in families]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    x = np.arange(len(families))
    width = 0.4
    b_b = ax.bar(
        x - width / 2, base_avg, width, alpha=0.6, color="#d62728", label="Base"
    )
    b_p = ax.bar(
        x + width / 2, post_avg, width, alpha=0.6, color="#1f77b4", label="Post-trained"
    )
    ax.set_title("Solver Accuracy ↑", fontsize=14, fontweight="bold", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=13.2)
    format_y_2dp(ax)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontweight="bold", fontsize=13.2)

    legend = ax.legend(
        handles=[
            Patch(facecolor="#d62728", alpha=0.6, label="Base"),
            Patch(facecolor="#1f77b4", alpha=0.6, label="Post-trained"),
        ],
        fontsize=12,
    )
    bold_legend(legend)

    all_vals = base_avg + post_avg
    ymin, ymax = min(all_vals), max(all_vals)
    ax.set_ylim(max(0, ymin - 0.02), ymax + (ymax - ymin) * 0.15)

    for bar_set, vs in [(b_b, base_avg), (b_p, post_avg)]:
        for bar, val in zip(bar_set, vs, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=12.375,
                fontweight="bold",
            )
    plt.tight_layout()
    save_fig(fig, "figure_12_base_vs_posttrained_solvers")


def figure_6():
    """Figure 6: verifier FPR, FNR, and gain of base and post-trained models, per family and setting."""
    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])

    # Average matrices across datasets once, then re-use.
    avg_mats = {
        k: matrix_avg_across_datasets(k)[2]
        for k in ("verifier_fpr", "verifier_fnr", "verifier_gain")
    }
    pairs = _base_post_pairs(rows37)
    pairs = [(b, p) for b, p in pairs if not categorize_model(b)[0].startswith("Llama")]

    families = sorted({categorize_model(b)[0] for b, _ in pairs})
    settings = [
        ("Self-Verification", "self"),
        ("Intra-Family Verification", "intra"),
        ("Cross-Family Verification", "cross"),
    ]
    metrics = [
        ("Verifier FPR ↓", "verifier_fpr"),
        ("Verifier FNR ↓", "verifier_fnr"),
        ("Verifier Gain ↑", "verifier_gain"),
    ]

    fig, axes = plt.subplots(
        len(metrics), len(settings), figsize=(3 * len(settings), 3 * len(metrics))
    )

    for r, (mlabel, mkey) in enumerate(metrics):
        per_setting = []  # list of (base_per_fam, post_per_fam) for each setting
        for _, setting in settings:
            base_per_fam, post_per_fam = [], []
            for fam in families:
                base_vals, post_vals = [], []
                for b, p in pairs:
                    if categorize_model(b)[0] != fam:
                        continue
                    base_vals.append(
                        per_verifier_avg(avg_mats[mkey], rows37, rows37, setting)[
                            rows37.index(b)
                        ]
                    )
                    post_vals.append(
                        per_verifier_avg(avg_mats[mkey], rows37, rows37, setting)[
                            rows37.index(p)
                        ]
                    )
                base_per_fam.append(np.mean(base_vals))
                post_per_fam.append(np.mean(post_vals))
            per_setting.append((base_per_fam, post_per_fam))

        flat = [v for b, p in per_setting for v in b + p]
        ymin, ymax = min(flat), max(flat)
        ymargin = 0.4 if r == 0 else 0.1

        for c, ((slabel, _), (base_v, post_v)) in enumerate(
            zip(settings, per_setting, strict=False)
        ):
            ax = axes[r, c]
            x = np.arange(len(families))
            w = 0.4
            bb = ax.bar(x - w / 2, base_v, w, alpha=0.6, color="#d62728")
            bp = ax.bar(x + w / 2, post_v, w, alpha=0.6, color="#1f77b4")
            if r == 0:
                title = slabel.replace("-Family Verification", "-Family\nVerification")
                ax.set_title(title, fontsize=18, fontweight="bold", pad=5)
            if c == 0:
                ax.set_ylabel(mlabel, fontsize=18, fontweight="bold", labelpad=10)
                format_y_2dp(ax)
            else:
                ax.set_yticklabels([])

            # Hide bottom y-tick on FPR row to avoid overlap.
            if r == 0:
                ax.yaxis.set_major_locator(MaxNLocator(prune="lower"))

            for label in ax.get_yticklabels():
                label.set_fontweight("bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.tick_params(axis="both", which="major", labelsize=13.2)
            ax.set_xticks(x)
            if r == len(metrics) - 1:
                ax.set_xticklabels(families, fontsize=18, fontweight="bold")
            else:
                ax.set_xticklabels([])
            if ymin < 0:
                ax.set_ylim(ymin - ymargin, ymax + ymargin)
            else:
                ax.set_ylim(max(0, ymin - ymargin), ymax + ymargin)

            if r == 0 and c == 0:
                legend = ax.legend(
                    handles=[
                        Patch(facecolor="#d62728", alpha=0.6, label="Base"),
                        Patch(facecolor="#1f77b4", alpha=0.6, label="Post-trained"),
                    ],
                    fontsize=11,
                    loc="upper left",
                )
                bold_legend(legend)

            for bar_set, vs in [(bb, base_v), (bp, post_v)]:
                for bar, val in zip(bar_set, vs, strict=False):
                    y_text = val + 0.02 if val >= 0 else val - 0.02
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y_text,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=11,
                        fontweight="bold",
                    )
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    save_fig(fig, "figure_06_base_vs_posttrained_verifiers")


def figure_7():
    """Figure 7: verifier metrics per task, one point per dataset, for each verification setting."""
    rows37, _, _ = load_matrix("verifier_gain", DATASETS[0])
    post_idx = [
        i for i, m in enumerate(rows37) if categorize_model(m)[1] == "PostTrained"
    ]
    post_names = [rows37[i] for i in post_idx]

    metric_configs = [
        ("Verifier Accuracy ↑", "verifier_accuracy"),
        ("Verifier Gain ↑", "verifier_gain"),
    ]
    settings = [
        ("Self-Verification", "self"),
        ("Intra-Family Verification", "intra"),
        ("Cross-Family Verification", "cross"),
    ]

    fig, axes = plt.subplots(
        len(metric_configs),
        len(settings),
        figsize=(9 * len(settings), 9 * 2 / 3 * len(metric_configs)),
    )
    if len(metric_configs) == 1:
        axes = axes.reshape(1, -1)

    per_ds_solver_acc = {}
    for ds in DATASETS:
        _, vals = load_scalar("solver_accuracy", ds)
        per_ds_solver_acc[ds] = vals[post_idx].mean()

    for r, (mlabel, mkey) in enumerate(metric_configs):
        # For each (dataset, setting), the y-value is mean(over verifiers v of
        # mean(over solvers s matching setting of metric[s, v])).
        per_setting_data = {}
        for _, setting in settings:
            xs, ys, tasks = [], [], []
            for ds in DATASETS:
                _, _, full = load_matrix(mkey, ds)
                mat = full[np.ix_(post_idx, post_idx)]
                per_v = per_verifier_avg(mat, post_names, post_names, setting)
                avg = float(np.nanmean(per_v))
                xs.append(per_ds_solver_acc[ds])
                ys.append(avg)
                tasks.append(DATASET_LABEL[ds])
            per_setting_data[setting] = (xs, ys, tasks)

        all_y = np.concatenate([np.array(per_setting_data[s][1]) for _, s in settings])
        all_x = np.concatenate([np.array(per_setting_data[s][0]) for _, s in settings])
        x_margin = (all_x.max() - all_x.min()) * 0.2
        y_margin = (all_y.max() - all_y.min()) * 0.1 or 0.1

        for c, (slabel, setting) in enumerate(settings):
            ax = axes[r, c]
            xs, ys, tasks = per_setting_data[setting]
            colors = [
                PRIMARY_COLOR if t in PRIMARY_TASKS else OTHER_COLOR for t in tasks
            ]
            ax.scatter(xs, ys, s=150, alpha=0.7, c=colors)
            # Label offsets that keep the task names from overlapping.
            for x, y, t in zip(xs, ys, tasks, strict=False):
                xytext, va = (5, 5), "bottom"
                if (
                    (r == 0 and c == 1 and t == "MMLU (SS)")
                    or (r == 0 and c == 2 and t in ("MMLU (SS)", "Matrix Mult."))
                    or (r == 1 and c == 0 and t in ("GPQA", "MMLU (STEM)"))
                ):
                    xytext, va = (5, -7.5), "top"
                elif r == 1 and c == 0 and t == "MMLU (SS)":
                    xytext, va = (7, 0), "top"
                elif r == 1 and c == 1 and t in ("CSQA", "MMLU (STEM)"):
                    xytext, va = (5, -7.5), "top"
                elif r == 1 and c == 1 and t == "MMLU (SS)":
                    xytext, va = (5, -10), "top"
                elif r == 1 and c == 2 and t == "CSQA":
                    xytext, va = (5, -7.5), "top"
                ax.annotate(
                    t,
                    (x, y),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=18,
                    ha="left",
                    va=va,
                    fontweight="bold",
                )

            if c == 0:
                ax.set_ylabel(
                    mlabel, fontsize=32, fontweight="bold", rotation=90, labelpad=20
                )
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(slabel, fontsize=32, fontweight="bold", pad=15)
            if r == len(metric_configs) - 1:
                ax.set_xlabel(
                    "Solver Accuracy", fontsize=32, fontweight="bold", labelpad=20
                )
            else:
                ax.set_xticklabels([])

            ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
            ax.set_ylim(all_y.min() - y_margin, all_y.max() + y_margin)
            ax.grid(True, alpha=0.3)
            bold_ticks(ax, labelsize=24)

            r_val = np.corrcoef(xs, ys)[0, 1]
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.linspace(min(xs), max(xs), 100)
            ax.plot(
                x_line,
                slope * x_line + intercept,
                "--",
                color="gray",
                alpha=0.9,
                linewidth=5,
            )
            ax.text(
                0.02,
                0.98,
                f"r = {r_val:.3f}\nslope = {slope:.3f}",
                transform=ax.transAxes,
                fontsize=24,
                fontweight="bold",
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.4},
            )

    plt.tight_layout()
    save_fig(fig, "figure_07_verifier_metrics_by_task")


def figure_8():
    """Figure 8: solver filter rate of all 37 models, averaged over the 9 datasets."""
    labels, vals = scalar_avg_across_datasets("solver_filter_rate")
    colors = [color_for_model(m) for m in labels]

    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    ax.bar(range(len(labels)), vals, color=colors, alpha=0.65)
    ax.set_ylabel(
        "Solver\nFilter Ratio ↓",
        fontsize=22,
        fontweight="bold",
        rotation=90,
        labelpad=20,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, 1.2, 0.1))
    ax.set_ylim(0, max(vals) + 0.05)
    ax.grid(True, alpha=0.3, axis="y")
    bold_ticks(ax, labelsize=15, which="y")
    format_y_2dp(ax)

    legend = fig.legend(
        handles=family_legend(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=7,
        fontsize=18.7,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    save_fig(fig, "figure_08_solver_filter_rate_by_model")


def figure_9():
    """Figure 9: solver accuracy against model size on each of the 9 datasets."""
    fig, axes = plt.subplots(3, 3, figsize=(7 * 3, 7 * 3))

    for d_idx, ds in enumerate(DATASETS):
        ax = axes[d_idx // 3, d_idx % 3]
        labels, vals = load_scalar("solver_accuracy", ds)

        # Group by (family, training_type) for trend lines.
        category_points = {}
        for nick, v in zip(labels, vals, strict=False):
            fam, tt = categorize_model(nick)
            key = (fam, tt)
            category_points.setdefault(key, []).append((MODEL_SIZES[nick], v))

        sizes = [MODEL_SIZES[m] for m in labels]
        colors = [color_for_model(m) for m in labels]
        ax.scatter(sizes, vals, s=125, alpha=0.8, c=colors, edgecolors="none")
        for points in category_points.values():
            if len(points) > 1:
                xs, ys = zip(*sorted(points), strict=False)
                fam_key = list(category_points.keys())[
                    list(category_points.values()).index(points)
                ]
                color = (
                    FAMILY_COLORS[f"{fam_key[0]}-Base"]
                    if fam_key[1] == "Base"
                    else FAMILY_COLORS[fam_key[0]]
                )
                ax.plot(xs, ys, "--", color=color, alpha=0.6, linewidth=2)

        ax.set_xlabel("Model Size (Billions of Parameters)", labelpad=10, fontsize=18)
        ax.set_title(
            f"{DATASET_LABEL[ds]} Solver Accuracy ↑",
            fontweight="bold",
            pad=15,
            fontsize=17.5,
        )
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="both", which="major", labelsize=15)
        if d_idx % 3 != 0:
            ax.set_yticklabels([])

    legend = fig.legend(
        handles=family_legend(labels, alpha=1.0),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=7,
        fontsize=20,
        frameon=False,
        handlelength=2,
        handletextpad=0.5,
        markerscale=2,
    )
    bold_legend(legend)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.35)
    save_fig(fig, "figure_09_solver_accuracy_by_task")


def main():
    """Render all 12 paper figures into figures/."""
    print("Plotting figures from", DATA_DIR)
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    figure_9()
    figure_10()
    figure_11()
    figure_12()
    print("All 12 paper figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
