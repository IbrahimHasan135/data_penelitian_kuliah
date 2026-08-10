"""
Compare the three automation scenarios (0 / 1 / 2 visitors) for Energy,
Power, and Temperature.

Produces 3 PNG charts (one per metric), each overlaying the 3 visitor
scenarios in one figure with a different color per line.

Sessions were recorded at different clock times with different sample
counts, so all charts are aligned on "elapsed time since session start"
(minutes) rather than raw timestamps, and every chart is fit into a fixed
0-60 minute window (whatever runs past 60 minutes is simply not shown).
Energy is plotted as consumption relative to the start of each session
(not the raw cumulative meter reading), since each session starts from a
different baseline. The y-axis on each chart uses the same fixed range as
this project's original single-file plots (main2.py / main3.py / main4.py)
so scenarios are visually comparable on the same scale.
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed on Linux

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.ticker import FormatStrFormatter

DATA_DIR = Path(__file__).resolve().parent

# Fixed color per scenario -- kept identical in compare_automation_vs_normal.py
# so the same scenario always reads as the same color across both charts.
SCENARIOS = [
    {"key": "0", "label": "No Visitor (Automation)", "color": "#008300"},
    {"key": "1", "label": "One Visitor (Automation)", "color": "#e87ba4"},
    {"key": "2", "label": "Two Visitors (Automation)", "color": "#eda100"},
]

GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
FIG_SIZE = (16, 8)
LINE_WIDTH = 2.5
TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
X_TICKS = [0, 20, 40, 60]

# Fixed x-window: fit every chart into a common 0-60 minute view.
XLIM = (0, 60)

# Fixed y-ranges, carried over from the original single-file plots so
# every scenario reads on the same scale:
#   - Energy: main2.py/main3.py used (energy_min - 0.1, energy_min + 1.0);
#     here energy is plotted as a delta starting at 0, so that becomes (-0.1, 1.0).
#   - Power: main2.py/main3.py locked it to (0, 900).
#   - Temperature: main4.py locked it to (21, 26).
YLIM_ENERGY = (-0.1, 1.0)
YLIM_POWER = (0, 900)
YLIM_TEMPERATURE = (21, 26)


def load_series(path: Path) -> pd.DataFrame:
    """
    Load a sensor xlsx export. Handles both on-disk shapes found in this
    dataset:
      - 3 separate columns: (entity_id|mac_address, state|temperature, last_changed|time)
      - 1 combined column: "entity_id,state,last_changed" per row

    Returns a DataFrame with columns [time, value], sorted by time.
    """
    df = pd.read_excel(path)
    cols = [str(c).strip() for c in df.columns]

    if len(cols) == 1:
        parts = df.iloc[:, 0].astype(str).str.split(",", n=2, expand=True)
        parts.columns = ["entity", "value", "time"]
        value = pd.to_numeric(parts["value"], errors="coerce")
        time = pd.to_datetime(parts["time"], errors="coerce", utc=True)
    else:
        df.columns = cols
        value_col = "state" if "state" in cols else "temperature"
        time_col = "last_changed" if "last_changed" in cols else "time"
        value = pd.to_numeric(df[value_col], errors="coerce")
        time = pd.to_datetime(df[time_col], errors="coerce", utc=True)

    out = pd.DataFrame({"time": time, "value": value})
    out = out.dropna().sort_values("time").reset_index(drop=True)
    return out


def elapsed_minutes(df: pd.DataFrame) -> pd.Series:
    return (df["time"] - df["time"].iloc[0]).dt.total_seconds() / 60.0


def style_axes(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_xticks(X_TICKS)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=TICK_SIZE)


def y_tick_format(value_label: str) -> str:
    if "Energy" in value_label:
        return "%.2f"
    if "Power" in value_label:
        return "%.0f"
    return "%.1f"


def plot_metric(file_template: str, value_label: str, title: str,
                 output_name: str, ylim: tuple, as_delta: bool = False):
    loaded = []
    for sc in SCENARIOS:
        path = DATA_DIR / file_template.format(sc["key"])
        if not path.exists():
            print(f"Warning: {path.name} not found, skipping.")
            continue

        df = load_series(path)
        if df.empty:
            print(f"Warning: {path.name} has no usable data, skipping.")
            continue

        x = elapsed_minutes(df)
        y = (df["value"] - df["value"].iloc[0]) if as_delta else df["value"]
        loaded.append({"label": sc["label"], "color": sc["color"], "x": x, "y": y})

    if not loaded:
        print(f"No data available for {output_name}, skipping chart.")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for item in loaded:
        ax.plot(
            item["x"], item["y"],
            color=item["color"], linewidth=LINE_WIDTH,
            label=item["label"],
        )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Time (minutes)", fontsize=LABEL_SIZE, color=MUTED_COLOR)
    ax.set_ylabel(value_label, fontsize=LABEL_SIZE, color=MUTED_COLOR)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_tick_format(value_label)))
    style_axes(ax)
    ax.legend(loc="best", frameon=True, framealpha=0.9, fontsize=LEGEND_SIZE)

    fig.tight_layout()
    out_path = DATA_DIR / output_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    plot_metric(
        file_template="E_{}.xlsx",
        value_label="Energy Consumed (kWh)",
        title="Energy Consumption Comparison Across Automation Scenarios",
        output_name="comparison_energy_automation.png",
        ylim=YLIM_ENERGY,
        as_delta=True,
    )
    plot_metric(
        file_template="P_{}.xlsx",
        value_label="Power (W)",
        title="Power Usage Comparison Across Automation Scenarios",
        output_name="comparison_power_automation.png",
        ylim=YLIM_POWER,
        as_delta=False,
    )
    plot_metric(
        file_template="T_{}.xlsx",
        value_label="Temperature (°C)",
        title="Temperature Comparison Across Automation Scenarios",
        output_name="comparison_temperature_automation.png",
        ylim=YLIM_TEMPERATURE,
        as_delta=False,
    )


if __name__ == "__main__":
    main()
