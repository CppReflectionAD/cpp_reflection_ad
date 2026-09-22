from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


WIDTH = 1200
HEIGHT = 900
MARGIN_LEFT = 90
MARGIN_RIGHT = 320
MARGIN_TOP = 50
MARGIN_BOTTOM = 60
PANEL_GAP = 110
PANEL_HEIGHT = 260
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]
CLOSED_FORM_COLOR = "#444444"
FINITE_DIFFERENCE_COLOR = "#17becf"


def usage() -> str:
    return "Usage: python plot_black_scholes_convergence.py " "[input.csv] [output.svg]"


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def nice_ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if min_value == max_value:
        delta = 1.0 if min_value == 0.0 else abs(min_value) * 0.1
        return [min_value - delta, min_value, min_value + delta]

    span = max_value - min_value
    raw_step = span / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    candidates = [1.0, 2.0, 5.0, 10.0]
    step = candidates[-1] * magnitude
    for candidate in candidates:
        candidate_step = candidate * magnitude
        if raw_step <= candidate_step:
            step = candidate_step
            break

    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step
    ticks: list[float] = []
    value = start
    guard = 0
    while value <= end + 0.5 * step and guard < 100:
        ticks.append(value)
        value += step
        guard += 1
    return ticks


def format_tick(value: float) -> str:
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.1e}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6g}"


def polyline_points(xs: list[float], ys: list[float], x_map, y_map) -> str:
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in zip(xs, ys))


def add_text(
    parts: list[str],
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "start",
    weight: str = "normal",
    fill: str = "#111111",
    rotate: float | None = None,
) -> None:
    transform = ""
    if rotate is not None:
        transform = f' transform="rotate({rotate:.0f} {x:.2f} {y:.2f})"'
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="Helvetica, Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{transform}>'
        f"{escape_xml(text)}</text>"
    )


def add_panel(
    parts: list[str],
    x_values: list[float],
    x_tick_labels: list[str],
    series: list[tuple[str, list[float], str, str | None, float]],
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    y_values = [value for _, values, _, _, _ in series for value in values]
    y_min = min(y_values)
    y_max = max(y_values)
    padding = 0.08 * max(1e-12, y_max - y_min)
    if padding == 0.0:
        padding = 1.0
    y_min -= padding
    y_max += padding

    x_min = min(x_values)
    x_max = max(x_values)

    def x_map(value: float) -> float:
        if x_max == x_min:
            return plot_left + plot_width / 2.0
        return plot_left + (value - x_min) * plot_width / (x_max - x_min)

    def y_map(value: float) -> float:
        if y_max == y_min:
            return plot_top + plot_height / 2.0
        return plot_top + plot_height - (value - y_min) * plot_height / (y_max - y_min)

    parts.append(
        f'<rect x="{plot_left:.2f}" y="{plot_top:.2f}" '
        f'width="{plot_width:.2f}" height="{plot_height:.2f}" '
        f'fill="#ffffff" stroke="#d0d0d0" stroke-width="1"/>'
    )

    for tick in nice_ticks(y_min, y_max, 6):
        y = y_map(tick)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{y:.2f}" '
            f'x2="{plot_left + plot_width:.2f}" y2="{y:.2f}" '
            f'stroke="#e6e6e6" stroke-width="1"/>'
        )
        add_text(
            parts, plot_left - 10, y + 5, format_tick(tick), 12, "end", fill="#555555"
        )

    for tick, tick_label in zip(x_values, x_tick_labels):
        x = x_map(tick)
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_top:.2f}" '
            f'x2="{x:.2f}" y2="{plot_top + plot_height:.2f}" '
            f'stroke="#f1f1f1" stroke-width="1"/>'
        )
        add_text(
            parts,
            x,
            plot_top + plot_height + 22,
            tick_label,
            12,
            "middle",
            fill="#555555",
        )

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_height:.2f}" '
        f'x2="{plot_left + plot_width:.2f}" y2="{plot_top + plot_height:.2f}" '
        f'stroke="#333333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" '
        f'x2="{plot_left:.2f}" y2="{plot_top + plot_height:.2f}" '
        f'stroke="#333333" stroke-width="1.5"/>'
    )

    add_text(
        parts, plot_left + plot_width / 2.0, plot_top - 22, title, 16, "middle", "bold"
    )
    add_text(
        parts,
        plot_left + plot_width / 2.0,
        plot_top + plot_height + 46,
        x_label,
        13,
        "middle",
    )
    add_text(
        parts,
        plot_left - 62,
        plot_top + plot_height / 2.0,
        y_label,
        13,
        "middle",
        rotate=-90,
    )

    for label, values, color, dasharray, stroke_width in series:
        dash_attr = ""
        if dasharray is not None:
            dash_attr = f' stroke-dasharray="{dasharray}"'
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width:.1f}"{dash_attr} '
            f'points="{polyline_points(x_values, values, x_map, y_map)}"/>'
        )
        for x, y in zip(x_values, values):
            parts.append(
                f'<circle cx="{x_map(x):.2f}" cy="{y_map(y):.2f}" r="2.7" '
                f'fill="{color}"/>'
            )


def add_legend(
    parts: list[str], items: list[tuple[str, str]], x: float, y: float
) -> None:
    row_height = 28
    top_padding = 18
    bottom_padding = 30
    title_gap = 30
    right_padding = 24
    max_width = max(220, WIDTH - x - right_padding)
    legend_width = max(
        220,
        min(max_width, 90 + max(len(label) for label, _ in items) * 7),
    )
    box_height = top_padding + title_gap + row_height * len(items) + bottom_padding
    parts.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{legend_width:.2f}" height="{box_height:.2f}" '
        f'fill="#ffffff" fill-opacity="0.92" stroke="#d0d0d0" stroke-width="1" rx="8"/>'
    )
    add_text(parts, x + 12, y + 22, "Legend", 14, "start", "bold")
    for index, (label, color) in enumerate(items):
        row_y = y + top_padding + title_gap + row_height * index
        parts.append(
            f'<line x1="{x + 12:.2f}" y1="{row_y - 5:.2f}" '
            f'x2="{x + 40:.2f}" y2="{row_y - 5:.2f}" stroke="{color}" stroke-width="2.5"/>'
        )
        parts.append(
            f'<circle cx="{x + 26:.2f}" cy="{row_y - 5:.2f}" r="2.7" fill="{color}"/>'
        )
        add_text(parts, x + 48, row_y, label, 12)


def read_rows(csv_path: Path) -> tuple[list[dict[str, float]], list[str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {key: float(value) for key, value in row.items() if value is not None}
            )
        return rows, list(reader.fieldnames or [])


def main(argv: list[str]) -> int:
    if len(argv) > 3:
        print(usage(), file=sys.stderr)
        return 1

    default_csv = Path(__file__).with_name("black_scholes_simulated_convergence.csv")
    csv_path = Path(argv[1]) if len(argv) >= 2 else default_csv
    svg_path = Path(argv[2]) if len(argv) >= 3 else csv_path.with_suffix(".svg")

    if not csv_path.exists():
        print(f"Input CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows, fieldnames = read_rows(csv_path)
    if not rows:
        print(f"CSV is empty: {csv_path}", file=sys.stderr)
        return 1

    x_values = [row["log10_paths"] for row in rows]
    x_tick_labels = [format_tick(row["paths"]) for row in rows]
    sim_per_path_value = rows[0].get("sim_per_path")
    price_series = [
        (
            "Closed-form price",
            [row["closed_form_price"] for row in rows],
            COLORS[0],
            None,
            2.5,
        ),
        ("MC price", [row["mc_price"] for row in rows], COLORS[1], None, 2.5),
    ]

    delta_series: list[tuple[str, list[float], str, str | None, float]] = [
        (
            "Closed-form delta",
            [row["closed_form_delta"] for row in rows],
            CLOSED_FORM_COLOR,
            None,
            2.5,
        ),
        (
            "Corrected delta",
            [row["legacy_corrected_delta"] for row in rows],
            COLORS[0],
            None,
            2.5,
        ),
    ]
    legend_items = [
        ("Closed-form delta", CLOSED_FORM_COLOR),
        ("Corrected delta", COLORS[0]),
    ]

    finite_difference_series: tuple[str, list[float], str, str | None, float] | None = (
        None
    )
    if "finite_difference_delta" in fieldnames:
        finite_difference_series = (
            "Finite-difference delta",
            [row["finite_difference_delta"] for row in rows],
            FINITE_DIFFERENCE_COLOR,
            "8 5",
            3.0,
        )
        legend_items.append(("Finite-difference delta", FINITE_DIFFERENCE_COLOR))

    smoothed_columns = sorted(
        [column for column in fieldnames if column.startswith("smoothed_delta_w=")],
        key=lambda name: float(name.split("=", 1)[1]),
    )
    for index, column in enumerate(smoothed_columns, start=1):
        width = column.split("=", 1)[1]
        color = COLORS[index % len(COLORS)]
        label = f"Smoothed delta (w={width})"
        delta_series.append((label, [row[column] for row in rows], color, None, 2.5))
        legend_items.append((label, color))

    if finite_difference_series is not None:
        delta_series.append(finite_difference_series)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
    ]

    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    add_panel(
        svg_parts,
        x_values,
        x_tick_labels,
        price_series,
        MARGIN_LEFT,
        MARGIN_TOP,
        plot_width,
        PANEL_HEIGHT,
        "Price Convergence",
        "price",
        "paths (logscale)",
    )
    add_panel(
        svg_parts,
        x_values,
        x_tick_labels,
        delta_series,
        MARGIN_LEFT,
        MARGIN_TOP + PANEL_HEIGHT + PANEL_GAP,
        plot_width,
        PANEL_HEIGHT,
        "Delta Convergence",
        "delta",
        "paths (logscale)",
    )

    add_text(
        svg_parts,
        MARGIN_LEFT,
        HEIGHT - 18,
        f"Source: {csv_path.name}",
        11,
        "start",
        fill="#666666",
    )
    if sim_per_path_value is not None:
        add_text(
            svg_parts,
            MARGIN_LEFT + 260,
            HEIGHT - 18,
            f"sim-per-path: {format_tick(sim_per_path_value)}",
            11,
            "start",
            fill="#666666",
        )
    add_legend(
        svg_parts,
        [("Closed-form price", COLORS[0]), ("MC price", COLORS[1])] + legend_items,
        WIDTH - MARGIN_RIGHT + 20,
        90,
    )

    svg_parts.append("</svg>")
    svg_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
