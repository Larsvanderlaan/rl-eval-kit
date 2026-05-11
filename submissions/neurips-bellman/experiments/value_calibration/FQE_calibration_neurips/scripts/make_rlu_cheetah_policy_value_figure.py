"""Make the compact RL Unplugged cheetah_run policy-value calibration figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FILES = [
    ROOT / "results" / "rlu_cheetah_policy_value_main" / "rlu_cheetah_policy_value_summary.csv",
    ROOT / "results" / "rlu_cheetah_policy_value_curve2" / "rlu_cheetah_policy_value_summary.csv",
    ROOT / "results" / "rlu_cheetah_policy_value_curve3" / "rlu_cheetah_policy_value_summary.csv",
]
OUT_DIR = ROOT / "paper_import_bundle" / "figures"


def main() -> None:
    frames = []
    for curve_id, path in enumerate(SUMMARY_FILES, start=1):
        frame = pd.read_csv(path)
        frame["curve"] = curve_id
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(["linear", "isotonic"])].copy()

    split_order = ["early_to_late", "alternating", "early5_to_late3"]
    split_labels = {
        "early_to_late": "cal 0-3 / eval 4-7",
        "alternating": "cal even / eval odd",
        "early5_to_late3": "cal 0-4 / eval 5-7",
    }
    method_order = ["linear", "isotonic"]
    method_labels = {"linear": "Linear", "isotonic": "Isotonic"}
    colors = {"linear": "#3B6EA8", "isotonic": "#C45A2C"}

    mean = (
        df.groupby(["split", "method"], as_index=False)["relative_absolute_ope_error"]
        .mean()
        .pivot(index="split", columns="method", values="relative_absolute_ope_error")
        .loc[split_order]
    )

    fig, ax = plt.subplots(figsize=(4.2, 1.55))
    y_base = range(len(split_order))
    bar_h = 0.32
    offsets = {"linear": -bar_h / 1.8, "isotonic": bar_h / 1.8}

    for method in method_order:
        values = mean[method].to_numpy()
        ypos = [y + offsets[method] for y in y_base]
        ax.barh(
            ypos,
            values,
            height=bar_h,
            color=colors[method],
            label=method_labels[method],
            edgecolor="white",
            linewidth=0.5,
        )
        for y, value in zip(ypos, values):
            ax.text(value + 0.018, y, f"{value:.2f}", va="center", ha="left", fontsize=6.2)
        for split_i, split in enumerate(split_order):
            curve_values = df[(df["split"] == split) & (df["method"] == method)]["relative_absolute_ope_error"].to_numpy()
            lo = float(curve_values.min())
            hi = float(curve_values.max())
            center = float(curve_values.mean())
            ax.errorbar(
                center,
                split_i + offsets[method],
                xerr=[[center - lo], [hi - center]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.55,
                capsize=1.6,
                capthick=0.55,
                zorder=3,
            )

    ax.axvline(1.0, color="#444444", linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_yticks(list(y_base), [split_labels[s] for s in split_order])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute OPE error / raw FQE-L2 error", fontsize=6.8, labelpad=2)
    ax.set_xlim(0.0, 1.12)
    ax.set_title("Deep OPE cheetah_run official FQE-L2 scores", loc="left", fontsize=7.6, fontweight="bold", pad=2)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=6.1, handlelength=1.0, columnspacing=0.9)
    ax.tick_params(axis="both", labelsize=6.4)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT_DIR / "rlu_cheetah_policy_calibration_compact.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "rlu_cheetah_policy_calibration_compact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
