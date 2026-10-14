"""Color scale for color-coded values, shared by the plots and the Google Earth export (no plotting dependencies)."""

import numpy as np

# Sequential colors for magnitudes (reversed ColorBrewer RdYlBu: blue = low, red = high).
VALUE_COLORS = (
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#e0f3f8",
    "#ffffbf",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
)


def color_range(values: np.ndarray, max_std: float) -> tuple[float, float]:
    """Range of the color scale: minimum to maximum, clipped to mean ± ``max_std`` standard deviations
    (0 = no clipping), so that outliers don't dominate the scale."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lower, upper = float(finite.min()), float(finite.max())
    if max_std > 0:
        mean, std = float(finite.mean()), float(finite.std())
        lower, upper = max(lower, mean - max_std * std), min(upper, mean + max_std * std)
    if upper <= lower:
        upper = lower + 1e-9
    return lower, upper


def value_color(fraction: float) -> str:
    """Color ("#rrggbb") at ``fraction`` (0 = low, 1 = high) of the color scale, linearly interpolated."""
    stops = np.array([[int(color[i : i + 2], 16) for i in (1, 3, 5)] for color in VALUE_COLORS], dtype=float)
    positions = np.linspace(0.0, 1.0, len(VALUE_COLORS))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    red, green, blue = (round(float(np.interp(fraction, positions, stops[:, channel]))) for channel in range(3))
    return f"#{red:02x}{green:02x}{blue:02x}"
