"""Summary tables comparing several evaluation results, e.g. several algorithms against one reference."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult

TABLE_FORMATS = ("csv", "markdown", "latex")


def ate_summary(ate_results: Sequence[ATEResult], confidence_intervals: bool = True) -> pd.DataFrame:
    """One row per ATE result with its main statistics.

    Args:
        ate_results (Sequence[ATEResult]): Results to compare.
        confidence_intervals (bool, optional): Add the 95% confidence intervals of the mean and RMS of the
            position deviations (see :meth:`ATEResult.confidence_interval`). Defaults to True.

    Returns:
        pd.DataFrame: Indexed by the result names. Position statistics in meters, rotation statistics in
        degrees (NaN without orientations).
    """
    rows = []
    for result in ate_results:
        row: dict[str, float | int] = {
            "poses": len(result.pos_dev_comb),
            "pos_mean [m]": float(result.pos_ate),
            "pos_median [m]": float(result.pos_dev_median),
            "pos_rms [m]": float(result.pos_dev_rms),
            "pos_std [m]": float(result.pos_dev_std),
            "pos_p95 [m]": result.pos_dev_p95,
            "pos_max [m]": float(result.pos_dev_max),
        }
        if confidence_intervals:
            row["pos_mean_ci_low [m]"], row["pos_mean_ci_high [m]"] = result.confidence_interval("position", "mean")
            row["pos_rms_ci_low [m]"], row["pos_rms_ci_high [m]"] = result.confidence_interval("position", "rms")
        if result.has_orientation:
            row |= {
                "rot_mean [deg]": float(np.rad2deg(result.rot_ate)),
                "rot_rms [deg]": float(np.rad2deg(result.rot_dev_rms)),
                "rot_p95 [deg]": float(np.rad2deg(result.rot_dev_p95)),
                "rot_max [deg]": float(np.rad2deg(result.rot_dev_max)),
            }
        else:
            row |= dict.fromkeys(["rot_mean [deg]", "rot_rms [deg]", "rot_p95 [deg]", "rot_max [deg]"], np.nan)
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(_unique_names([result.name for result in ate_results]), name="name"))


def rpe_summary(rpe_results: Sequence[RPEResult]) -> pd.DataFrame:
    """One row per RPE result with the average drift over all pair distances.

    Returns:
        pd.DataFrame: Indexed by the result names; drifts in % (pairs by distance) or m/s (pairs by time) and
        deg/100m or deg/s, see the ``pos_drift_unit`` and ``rot_drift_unit`` columns.
    """
    rows = []
    for result in rpe_results:
        rows.append(
            {
                "pose_pairs": len(result),
                "pos_drift": result.pos_rpe,
                "pos_drift_unit": result.pos_drift_unit,
                "rot_drift": float(np.rad2deg(result.rot_rpe)) if result.has_rot_dev else np.nan,
                "rot_drift_unit": result.rot_drift_unit,
            }
        )
    return pd.DataFrame(rows, index=pd.Index(_unique_names([result.name for result in rpe_results]), name="name"))


def summary_table(
    ate_results: Sequence[ATEResult] = (),
    rpe_results: Sequence[RPEResult] = (),
    confidence_intervals: bool = True,
) -> pd.DataFrame:
    """ATE and RPE summaries side by side, joined by the result names (outer join, so that results with only an
    ATE or only an RPE keep their row).

    Args:
        ate_results (Sequence[ATEResult], optional): ATE results.
        rpe_results (Sequence[RPEResult], optional): RPE results, named like the corresponding ATE results.
        confidence_intervals (bool, optional): See :func:`ate_summary`. Defaults to True.

    Returns:
        pd.DataFrame: One row per name.
    """
    ate = ate_summary(ate_results, confidence_intervals) if ate_results else pd.DataFrame()
    rpe = rpe_summary(rpe_results) if rpe_results else pd.DataFrame()
    if ate.empty:
        return rpe
    if rpe.empty:
        return ate
    table = ate.join(rpe, how="outer")
    # counts stay integers, also where the outer join left them empty
    for column in ("poses", "pose_pairs"):
        table[column] = table[column].astype("Int64")
    return table


def format_table(table: pd.DataFrame, table_format: str = "csv", precision: int = 4) -> str:
    """Formats a summary table as CSV, Markdown or LaTeX (without additional dependencies).

    Args:
        table (pd.DataFrame): Table, e.g. from :func:`summary_table`.
        table_format (str, optional): ``csv``, ``markdown`` or ``latex``. Defaults to ``csv``.
        precision (int, optional): Number of decimals of floats. Defaults to 4.

    Returns:
        str: The formatted table.
    """
    if table_format not in TABLE_FORMATS:
        raise ValueError(f"Unknown table format '{table_format}'. Options: {', '.join(TABLE_FORMATS)}.")
    if table_format == "csv":
        return table.to_csv(float_format=f"%.{precision}f")

    header = [str(table.index.name or ""), *map(str, table.columns)]
    rows = [[str(name), *values] for name, values in zip(table.index, formatted(table, precision).to_numpy())]

    if table_format == "markdown":
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines) + "\n"

    def escape(text: str) -> str:
        for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
            text = text.replace(char, "\\" + char)
        return text

    lines = [
        "\\begin{tabular}{l" + "r" * len(table.columns) + "}",
        "\\hline",
        " & ".join(escape(text) for text in header) + " \\\\",
        "\\hline",
        *(" & ".join(escape(text) for text in row) + " \\\\" for row in rows),
        "\\hline",
        "\\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def _cell(value, precision: int) -> str:
    if value is None or value is pd.NA:
        return "-"
    if isinstance(value, float | np.floating):
        return "-" if np.isnan(value) else f"{value:.{precision}f}"
    return str(value)


def formatted(table: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    """The table with all values as text: floats with the given number of decimals, missing values as ``-``."""
    return table.astype(object).apply(lambda column: column.map(lambda value: _cell(value, precision)))


def _unique_names(names: list[str]) -> list[str]:
    """Names with a running number appended to repeated ones, so that every row can be addressed."""
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return unique
