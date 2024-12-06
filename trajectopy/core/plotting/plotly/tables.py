import plotly.graph_objects as go
from plotly.offline import plot

from trajectopy.core.alignment.parameters import AlignmentParameters
from trajectopy.core.settings.report import ReportSettings


def render_alignment_table(
    alignment_parameters: AlignmentParameters, report_settings: ReportSettings = ReportSettings()
) -> str:
    """
    Render a heatmap plot.

    Args:
        alignment_parameters (AlignmentParameters): Alignment parameters.
        report_settings (ReportSettings, optional): Report settings. Defaults to ReportSettings().

    Returns:
        HTML string of the rendered report including the heatmap plot.
    """

    def extract_value(param_string: str) -> str:
        return param_string.split("=")[1].split("s")[0].strip()

    def extract_std(param_string: str) -> str:
        return param_string.split(":")[-1].strip()

    labels = alignment_parameters.params_labels(enabled_only=True, lower_case=False)
    alignment_data = [
        go.Table(
            header=dict(values=["Parameter", "Value", "Standard Deviation"]),
            cells=dict(
                values=[
                    labels,
                    [extract_value(param_string) for param_string in alignment_parameters.to_string_list()],
                    [extract_std(param_string) for param_string in alignment_parameters.to_string_list()],
                ],
            ),
            name="Alignment Parameters",
        )
    ]
    fig = go.Figure(data=alignment_data)
    return plot(fig, output_type="div", config=report_settings.single_plot_export.to_config())
