"""Self-contained HTML reports of evaluations, e.g. to archive or share the results of a paper.

A report holds everything needed to reproduce and judge an evaluation in one file without external
resources: the summary table of all results, the statistics of each result, the alignment parameters, the
processing settings, and the figures as embedded images (with the plotting extras installed).

Example:
    >>> import trajectopy as tpy
    >>> from trajectopy.report import write_report
    >>> ate_result = tpy.ate(estimate, reference)
    >>> write_report("report.html", ate_results=[ate_result], settings=tpy.ProcessingSettings())
"""

import base64
import html
import logging
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from trajectopy.__version__ import __version__
from trajectopy.core.settings import PlotSettings, ProcessingSettings
from trajectopy.results.alignment_result import AlignmentResult
from trajectopy.results.ate_result import ATEResult
from trajectopy.results.rpe_result import RPEResult
from trajectopy.results.summary import formatted, summary_table

logger = logging.getLogger(__name__)

_STYLE = """
:root { --fg: #1f2328; --muted: #59636e; --bg: #ffffff; --panel: #f6f8fa; --border: #d1d9e0; --accent: #0969da; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6edf3; --muted: #9198a1; --bg: #0d1117; --panel: #151b23; --border: #3d444d; --accent: #4493f8; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 1200px; padding: 24px 16px 64px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
h1 { margin: 0 0 4px; font-size: 1.8em; }
h2 { margin-top: 40px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
h3 { margin-top: 28px; }
.meta { color: var(--muted); margin-bottom: 24px; }
nav a { color: var(--accent); margin-right: 16px; text-decoration: none; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 0.9em; margin: 8px 0; }
th, td { border: 1px solid var(--border); padding: 4px 8px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: var(--panel); }
.properties td:last-child { font-variant-numeric: tabular-nums; }
figure { margin: 16px 0 28px; }
figure img { width: 100%; border: 1px solid var(--border); border-radius: 6px; background: #fff; }
figcaption { color: var(--muted); font-size: 0.9em; margin-top: 4px; }
pre { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; overflow-x: auto;
  font-size: 0.85em; }
"""


def _table(rows: list[list[str]], header: list[str] | None = None, css_class: str = "") -> str:
    head = ""
    if header:
        head = "<thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in header) + "</tr></thead>"
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table class="{css_class}">{head}<tbody>{body}</tbody></table></div>'


def _summary_section(ate_results: Sequence[ATEResult], rpe_results: Sequence[RPEResult]) -> str:
    table = summary_table(ate_results, rpe_results)
    header = [str(table.index.name or "name"), *map(str, table.columns)]
    rows = [[str(name), *values] for name, values in zip(table.index, formatted(table).to_numpy())]
    return "<h2 id='summary'>Summary</h2>" + _table(rows, header)


def _result_section(results: Sequence[ATEResult | RPEResult]) -> str:
    parts = ["<h2 id='results'>Results</h2>"]
    for result in results:
        parts.append(f"<h3>{html.escape(result.name)}</h3>")
        parts.append(_table([[key, value] for key, value in result.property_dict.items()], css_class="properties"))
    return "".join(parts)


def _alignment_section(alignment: AlignmentResult) -> str:
    rows = [
        [line]
        for line in alignment.position_parameters.to_string_list() + alignment.rotation_parameters.to_string_list()
    ]
    converged = "" if alignment.converged else "<p><strong>The estimation did not converge.</strong></p>"
    return f"<h2 id='alignment'>Alignment: {html.escape(alignment.name)}</h2>{converged}" + (
        _table(rows, ["Estimated parameter (value ± standard deviation)"]) if rows else "<p>No parameters.</p>"
    )


def _report_variants(figure, result_names: set[str]) -> list[str]:
    """The variants of a figure to show in the report: the first one, one per result for per-result figures, and
    the combined position deviations of maps (the other components are in the GUI and in exports)."""
    names = list(figure.variants)
    keep = [name for name in names if name in result_names or name == "Position" or name.endswith(": Position")]
    return [names[0], *[name for name in keep if name != names[0]]]


def _figure_section(
    ate_results: Sequence[ATEResult], rpe_results: Sequence[RPEResult], plot_settings: PlotSettings, width: int
) -> str:
    try:
        from trajectopy.visualization.plots import deviation_figures
    except ImportError:
        logger.info("Figures are not part of the report: the plotting extras are not installed.")
        return (
            "<h2 id='figures'>Figures</h2><p>Install the plotting extras to include figures: "
            '<code>pip install "trajectopy[plot]"</code></p>'
        )

    figures = deviation_figures(ate_results, rpe_results, settings=plot_settings)
    result_names = {result.name for result in ate_results} | {result.name for result in rpe_results}
    parts = ["<h2 id='figures'>Figures</h2>"]
    with tempfile.TemporaryDirectory() as directory:
        for index, figure in enumerate(figures):
            for number, variant in enumerate(_report_variants(figure, result_names)):
                path = Path(directory) / f"{index}_{number}.png"
                try:
                    figure.export(str(path), variant=variant, width=width)
                except Exception as error:  # a single failing figure must not prevent the report
                    logger.warning("Could not render the figure '%s' (%s): %s", figure.title, variant, error)
                    continue
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                caption = figure.title + (f" – {variant}" if variant else "")
                parts.append(
                    f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{data}">'
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>"
                )
    return "".join(parts)


def report_html(
    ate_results: Sequence[ATEResult] = (),
    rpe_results: Sequence[RPEResult] = (),
    alignment: AlignmentResult | None = None,
    settings: ProcessingSettings | None = None,
    plot_settings: PlotSettings | None = None,
    title: str = "Trajectory Evaluation Report",
    figures: bool = True,
    figure_width: int = 1200,
) -> str:
    """Builds a self-contained HTML report of an evaluation.

    Args:
        ate_results (Sequence[ATEResult], optional): ATE results.
        rpe_results (Sequence[RPEResult], optional): RPE results.
        alignment (AlignmentResult | None, optional): Alignment used before the evaluation.
        settings (ProcessingSettings | None, optional): Processing settings of the evaluation, listed for
            reproducibility.
        plot_settings (PlotSettings | None, optional): Settings of the figures.
        title (str, optional): Title of the report.
        figures (bool, optional): Include the figures (requires the plotting extras). Defaults to True.
        figure_width (int, optional): Width of the figures in pixels. Defaults to 1200.

    Returns:
        str: The HTML document.
    """
    if not ate_results and not rpe_results:
        raise ValueError("A report needs at least one ATE or RPE result.")

    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections = [_summary_section(ate_results, rpe_results), _result_section([*ate_results, *rpe_results])]
    nav = ["<a href='#summary'>Summary</a>", "<a href='#results'>Results</a>"]
    if alignment is not None:
        sections.append(_alignment_section(alignment))
        nav.append("<a href='#alignment'>Alignment</a>")
    if figures:
        sections.append(_figure_section(ate_results, rpe_results, plot_settings or PlotSettings(), figure_width))
        nav.append("<a href='#figures'>Figures</a>")
    if settings is not None:
        sections.append(f"<h2 id='settings'>Processing Settings</h2><pre>{html.escape(settings.to_json())}</pre>")
        nav.append("<a href='#settings'>Settings</a>")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='meta'>Created {created} with Trajectopy {html.escape(__version__)}</div>"
        f"<nav>{''.join(nav)}</nav>{''.join(sections)}</body></html>"
    )


def write_report(filename: str, *args, **kwargs) -> None:
    """Writes a self-contained HTML report of an evaluation; see :func:`report_html` for the arguments.

    Args:
        filename (str): Output path, e.g. ``report.html``.
    """
    Path(filename).write_text(report_html(*args, **kwargs), encoding="utf-8")
    logger.info("Wrote the report %s", filename)
