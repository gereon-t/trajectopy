"""Headless subcommands of the trajectopy command and the HTML report."""

from pathlib import Path

import pytest

from trajectopy.cli.commands import run

REFERENCE = "example_data/KITTI_gt.traj"
ESTIMATE = "example_data/KITTI_ORB.traj"


def test_evaluate_writes_table_results_and_report(tmp_path, capsys):
    table = tmp_path / "summary.md"
    report = tmp_path / "report.html"
    code = run(
        [
            "evaluate",
            ESTIMATE,
            "--ref",
            REFERENCE,
            "--table",
            str(table),
            "--output-dir",
            str(tmp_path / "results"),
            "--report",
            str(report),
            "--no-report-figures",
            "--quiet",
        ]
    )
    assert code == 0
    assert table.read_text().startswith("| name |")
    assert sorted(path.name.rsplit("_", 1)[1] for path in (tmp_path / "results").iterdir()) == [
        "alignment.result",
        "ate.result",
        "rpe.result",
    ]
    html = report.read_text(encoding="utf-8")
    assert "<h2 id='summary'>Summary</h2>" in html and "KITTI ORB" in html and "Processing Settings" in html


def test_ate_prints_summary(capsys):
    assert run(["ate", ESTIMATE, "--ref", REFERENCE, "--no-ci"]) == 0
    output = capsys.readouterr().out
    assert "pos_rms [m]" in output and "KITTI ORB" in output


def test_convert_and_info(tmp_path, capsys):
    kitti = tmp_path / "poses.kitti"
    assert run(["convert", REFERENCE, "--to", "kitti", "-o", str(kitti), "-q"]) == 0
    assert len(kitti.read_text().splitlines()[0].split()) == 12
    assert run(["info", str(kitti)]) == 0
    assert "Number of poses:              4541" in capsys.readouterr().out


def test_align_writes_trajectory_and_parameters(tmp_path):
    output, parameters = tmp_path / "aligned.tum", tmp_path / "alignment.result"
    code = run(
        [
            "align",
            ESTIMATE,
            "--ref",
            REFERENCE,
            "-o",
            str(output),
            "--to",
            "tum",
            "--parameters",
            str(parameters),
            "-q",
        ]
    )
    assert code == 0
    assert len(output.read_text().splitlines()) == 4541
    assert parameters.exists()


def test_errors_are_reported_without_traceback(capsys):
    assert run(["ate", "missing.traj", "--ref", REFERENCE]) == 2
    assert "File not found" in capsys.readouterr().err


def test_report_with_figures(tmp_path):
    pytest.importorskip("pyqtgraph")
    import trajectopy as tpy
    from trajectopy.report import write_report

    result = tpy.ate(tpy.Trajectory.from_file(ESTIMATE), tpy.Trajectory.from_file(REFERENCE))
    path = Path(tmp_path / "report.html")
    write_report(str(path), ate_results=[result])
    html = path.read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") >= 5
    assert "ATE Box Plot" in html and "ATE Segments" not in html
