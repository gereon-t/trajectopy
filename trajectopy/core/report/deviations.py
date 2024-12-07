"""
Trajectopy - Trajectory Evaluation in Python

Gereon Tombrink, 2024
mail@gtombrink.de
"""

import logging
from typing import Union


from trajectopy.core.evaluation.ate_result import ATEResult
from trajectopy.core.evaluation.rpe_result import RPEResult
from trajectopy.core.report.multi import render_multi_report
from trajectopy.core.report.single import render_single_report
from trajectopy.core.settings.report import ReportSettings


logger = logging.getLogger("root")


def create_deviation_report(
    ate_result: Union[ATEResult, list[ATEResult]],
    rpe_result: Union[RPEResult, list[RPEResult], None],
    report_settings: ReportSettings = ReportSettings(),
):
    """
    Create a HTML deviation report containing the absolute trajectory error and relative pose error results.

    Args:
        ate_result (Union[ATEResult, list[ATEResult]]): The absolute trajectory error results (one or multiple)
        rpe_result (Union[RPEResult, list[RPEResult]]): The relative pose error results (one or multiple)
        report_settings (ReportSettings): The report settings, defaults to ReportSettings()

    Returns:
        str: The deviation report
    """
    if (ate_result is not None and isinstance(ate_result, list)) or (
        rpe_result is not None and isinstance(rpe_result, list)
    ):
        return render_multi_report(ate_results=ate_result, rpe_results=rpe_result, report_settings=report_settings)

    return render_single_report(ate_result=ate_result, rpe_result=rpe_result, report_settings=report_settings)
