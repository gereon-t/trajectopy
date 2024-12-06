"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

from dataclasses import dataclass, field

from trajectopy.core.settings.base import Settings


@dataclass
class ExportSettings(Settings):
    format: str = "png"  # one of png, svg, jpeg, webp
    height: int = 500
    width: int = 800
    scale: int = 1

    def to_config(self) -> dict:
        return {
            "toImageButtonOptions": {
                "format": self.format,
                "height": self.height,
                "width": self.width,
                "scale": self.scale,
            }
        }


@dataclass
class ReportSettings(Settings):
    """
    ReportSettings class represents the settings for generating reports.

    - `single_plot_height` (int): The height of a single plot. Default value is 450.
    - `two_subplots_height` (int): The height of two subplots. Default value is 540.
    - `three_subplots_height` (int): The height of three subplots. Default value is 750.
    - `scatter_max_std` (float): The upper colorbar limit is set to the mean plus this value times the standard deviation of the data. This is useful to prevent outliers from dominating the colorbar. Default value is 4.0.
    - `ate_unit_is_mm` (bool): Indicates whether the unit of Absolute Trajectory Error (ATE) is millimeters. Default value is False.
    - `directed_ate` (bool): Indicates whether the ATE is split into along-, horizontal-cross- and vertical-cross-track direction. Default value is True.
    - `histogram_opacity` (float): The opacity of the histogram bars. Default value is 0.7.
    - `histogram_bargap` (float): The gap between histogram bars. Default value is 0.1.
    - `histogram_barmode` (str): The mode of displaying histogram bars. Default value is "overlay".
    - `histogram_yaxis_title` (str): The title of the y-axis in the histogram. Default value is "Count".
    - `plot_mode` (str): The mode of displaying plots. Default value is "lines+markers".
    - `scatter_mode` (str): The mode of displaying scatter plots. Default value is "markers".
    - `scatter_colorscale` (str): The colorscale for scatter plots. Default value is "RdYlBu_r".
    - `scatter_axis_order` (str): The order of the axes in scatter plots. Default value is "xy". If 3d plotting is desired, also specify "z".
    - `scatter_marker_size` (int): The size of markers in scatter plots. Default value is 5.
    - `scatter_detailed` (bool): Indicates whether to show scatter plots for each degree of freedom. Default value is False.
    - `scatter_mapbox` (bool): Indicates whether to use mapbox for scatter plots. Default value is False.
    - `scatter_mapbox_style` (str): The mapbox style for scatter plots. Default value is "open-street-map".
    - `scatter_mapbox_zoom` (int): The zoom level for scatter plots. Default value is 15.
    - `pos_x_name` (str): The name of the x-coordinate in position data. Default value is "x".
    - `pos_y_name` (str): The name of the y-coordinate in position data. Default value is "y".
    - `pos_z_name` (str): The name of the z-coordinate in position data. Default value is "z".
    - `pos_x_unit` (str): The unit of the x-coordinate in position data. Default value is "m".
    - `pos_y_unit` (str): The unit of the y-coordinate in position data. Default value is "m".
    - `pos_z_unit` (str): The unit of the z-coordinate in position data. Default value is "m".
    - `pos_dir_dev_x_name` (str): The name of the along-track direction deviation in position data. Default value is "along".
    - `pos_dir_dev_y_name` (str): The name of the horizontal-cross-track direction deviation in position data. Default value is "cross-h".
    - `pos_dir_dev_z_name` (str): The name of the vertical-cross-track direction deviation in position data. Default value is "cross-v".
    - `rot_x_name` (str): The name of the roll angle in rotation data. Default value is "roll".
    - `rot_y_name` (str): The name of the pitch angle in rotation data. Default value is "pitch".
    - `rot_z_name` (str): The name of the yaw angle in rotation data. Default value is "yaw".
    - `rot_unit` (str): The unit of rotation angles. Default value is "°".
    - `single_plot_export` (ExportSettings): The export settings for single plots. Default value is an instance of ExportSettings with width=800 and height=450.
    - `two_subplots_export` (ExportSettings): The export settings for two subplots. Default value is an instance of ExportSettings with width=800 and height=540.
    - `three_subplots_export` (ExportSettings): The export settings for three subplots. Default value is an instance of ExportSettings with width=800 and height=750.

    """

    single_plot_height: int = 640
    two_subplots_height: int = 750
    three_subplots_height: int = 860

    scatter_max_std: float = 4.0
    ate_unit_is_mm: bool = False
    directed_ate: bool = True

    histogram_opacity: float = 0.7
    histogram_bargap: float = 0.1
    histogram_barmode: str = "overlay"
    histogram_yaxis_title: str = "Count"

    plot_mode: str = "lines+markers"

    scatter_mode: str = "markers"
    scatter_colorscale: str = "RdYlBu_r"
    scatter_axis_order: str = "xy"
    scatter_marker_size: int = 5
    scatter_detailed: bool = False

    scatter_mapbox: bool = False
    scatter_mapbox_style: str = "open-street-map"
    scatter_mapbox_zoom: int = 15
    scatter_mapbox_token: str = ""

    pos_x_name: str = "x"
    pos_y_name: str = "y"
    pos_z_name: str = "z"
    pos_x_unit: str = "m"
    pos_y_unit: str = "m"
    pos_z_unit: str = "m"

    pos_dir_dev_x_name: str = "along"
    pos_dir_dev_y_name: str = "cross-h"
    pos_dir_dev_z_name: str = "cross-v"

    rot_x_name: str = "roll"
    rot_y_name: str = "pitch"
    rot_z_name: str = "yaw"
    rot_unit: str = "°"

    single_plot_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=540))
    two_subplots_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=540))
    three_subplots_export: ExportSettings = field(default_factory=lambda: ExportSettings(width=800, height=750))


if __name__ == "__main__":
    settings = ReportSettings()
    settings.to_file("report_settings.json")
    imported_settings = ReportSettings.from_file("report_settings.json")

    assert imported_settings == settings
    print(imported_settings)
