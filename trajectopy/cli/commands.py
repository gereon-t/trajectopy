"""Headless subcommands of the ``trajectopy`` command (no Qt required unless figures are exported).

Examples::

    trajectopy ate estimate.traj --ref reference.traj
    trajectopy evaluate orb.txt sptam.txt --ref gt.txt --settings settings.json --report report.html
    trajectopy rpe recording.mcap --topic /odom --ref gt.tum --table summary.md
    trajectopy align estimate.traj --ref reference.traj -o aligned.traj
    trajectopy convert poses.kitti --to tum -o poses.tum
    trajectopy info recording.bag
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger("trajectopy.cli")

COMMANDS = ("ate", "rpe", "evaluate", "align", "convert", "info")
FORMATS = ("auto", "trajectopy", "tum", "kitti", "euroc")


class CommandError(Exception):
    """An error caused by the input of a command (shown without a traceback)."""


# ------------------------------------------------------------------------------------------------ loading


def load_trajectories(path: str, topics: Sequence[str] | None = None, file_format: str = "auto") -> list:
    """Trajectories of a file: one for a trajectory file, one per topic for a ROS bag."""
    from trajectopy.io.formats import read_trajectory_file
    from trajectopy.io.rosbag import is_ros_bag, trajectories_from_rosbag

    if not Path(path).exists():
        raise CommandError(f"File not found: {path}")
    if is_ros_bag(path):
        trajectories = trajectories_from_rosbag(path, topics=topics or None)
        if not trajectories:
            raise CommandError(f"No trajectories of a supported message type found in {path}.")
        return trajectories
    return [read_trajectory_file(path, format=file_format)]


def load_reference(args: argparse.Namespace):
    references = load_trajectories(args.ref, [args.ref_topic] if args.ref_topic else None, args.ref_format)
    if len(references) > 1:
        topics = ", ".join(trajectory.name for trajectory in references)
        raise CommandError(f"The reference bag has several trajectories ({topics}); select one with --ref-topic.")
    return references[0]


def load_estimates(args: argparse.Namespace) -> list:
    return [trajectory for path in args.files for trajectory in load_trajectories(path, args.topic, args.format)]


def load_settings(path: str | None):
    from trajectopy.core.settings import ProcessingSettings

    if not path:
        return ProcessingSettings()
    if not Path(path).exists():
        raise CommandError(f"Settings file not found: {path}")
    return ProcessingSettings.from_file(path)


def load_plot_settings(path: str | None):
    from trajectopy.core.settings import PlotSettings

    return PlotSettings.from_file(path) if path else PlotSettings()


# ----------------------------------------------------------------------------------------------- commands


def run_evaluation(args: argparse.Namespace, with_ate: bool, with_rpe: bool) -> int:
    import trajectopy as tpy
    from trajectopy.results.summary import format_table, summary_table

    settings = load_settings(args.settings)
    reference = load_reference(args)
    estimates = load_estimates(args)

    ate_results, rpe_results, alignments = [], [], []
    for estimate in estimates:
        logger.info("Evaluating %s against %s", estimate.name, reference.name)
        trajectory = estimate
        if with_ate:
            ate_result, alignment = tpy.ate(
                estimate, reference, processing_settings=settings, return_alignment=True, align=not args.no_align
            )
            ate_results.append(ate_result)
            alignments.append(alignment)
            if not args.no_align:
                trajectory = tpy.apply_alignment(estimate, alignment_result=alignment, inplace=False)
        elif not args.no_align:
            trajectory = tpy.align(estimate, reference, settings.alignment, settings.matching, inplace=False)
        if with_rpe:
            rpe_results.append(tpy.rpe(trajectory, reference, processing_settings=settings))

    table = summary_table(ate_results, rpe_results, confidence_intervals=not args.no_ci)
    text = format_table(table, args.table_format)
    if args.table:
        Path(args.table).write_text(text, encoding="utf-8")
        logger.info("Wrote the summary table %s", args.table)
    if not args.quiet and not args.table:
        print(_console_table(table))

    if args.output_dir:
        _write_results(Path(args.output_dir), ate_results, rpe_results, alignments)
    if args.figures:
        _export_figures(Path(args.figures), ate_results, rpe_results, load_plot_settings(args.plot_settings))
    if args.report:
        from trajectopy.report import write_report

        write_report(
            args.report,
            ate_results=ate_results,
            rpe_results=rpe_results,
            alignment=alignments[0] if len(alignments) == 1 and not args.no_align else None,
            settings=settings,
            plot_settings=load_plot_settings(args.plot_settings),
            figures=not args.no_report_figures,
        )
    return 0


def _console_table(table) -> str:
    """A readable plain-text version of the summary for the terminal: one column per result."""
    from trajectopy.results.summary import formatted

    return formatted(table).T.to_string()


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name).strip("_") or "result"


def _write_results(directory: Path, ate_results, rpe_results, alignments) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for result in ate_results:
        path = directory / f"{_safe_name(result.name)}_ate.result"
        path.unlink(missing_ok=True)  # results are appended to their files
        result.to_file(str(path))
    for result in rpe_results:
        path = directory / f"{_safe_name(result.name)}_rpe.result"
        path.unlink(missing_ok=True)
        result.to_file(str(path))
    for result, alignment in zip(ate_results, alignments):
        path = directory / f"{_safe_name(result.name)}_alignment.result"
        path.unlink(missing_ok=True)
        alignment.to_file(str(path))
    logger.info("Wrote the results to %s", directory)


def _export_figures(directory: Path, ate_results, rpe_results, plot_settings) -> None:
    try:
        from trajectopy.visualization import deviation_figures, export_figures
    except ImportError as error:
        raise CommandError('Exporting figures requires the plotting extras: pip install "trajectopy[plot]"') from error
    paths = export_figures(
        deviation_figures(ate_results, rpe_results, settings=plot_settings),
        str(directory),
        export_format=plot_settings.export_format,
        width=plot_settings.export_width,
    )
    logger.info("Exported %d figures to %s", len(paths), directory)


def run_align(args: argparse.Namespace) -> int:
    import trajectopy as tpy
    from trajectopy.io.formats import write_trajectory_file

    settings = load_settings(args.settings)
    reference = load_reference(args)
    estimates = load_estimates(args)
    if len(estimates) > 1 and args.output and not Path(args.output).is_dir():
        raise CommandError("Several trajectories are aligned: give a directory as output.")

    for estimate in estimates:
        alignment = tpy.estimate_alignment(estimate, reference, settings.alignment, settings.matching)
        aligned = tpy.apply_alignment(estimate, alignment_result=alignment, inplace=False)
        if not args.quiet:
            print(f"{estimate.name}:")
            for line in (
                alignment.position_parameters.to_string_list() + alignment.rotation_parameters.to_string_list()
            ):
                print(f"  {line}")
        if args.output:
            output = Path(args.output)
            path = output / f"{_safe_name(estimate.name)}_aligned.traj" if output.is_dir() else output
            write_trajectory_file(aligned, str(path), format=args.to)
            logger.info("Wrote %s", path)
        if args.parameters:
            Path(args.parameters).unlink(missing_ok=True)
            alignment.to_file(args.parameters)
    return 0


def run_convert(args: argparse.Namespace) -> int:
    from trajectopy.io.formats import write_trajectory_file

    trajectories = [t for path in args.files for t in load_trajectories(path, args.topic, args.format)]
    output = Path(args.output)
    if len(trajectories) > 1 and not output.is_dir():
        raise CommandError("Several trajectories are converted: give an existing directory as output.")
    suffix = {"tum": ".tum", "kitti": ".kitti", "euroc": ".csv"}.get(args.to, ".traj")
    for trajectory in trajectories:
        if args.epsg is not None:
            trajectory.positions.to_epsg(args.epsg, inplace=True)
        path = output / f"{_safe_name(trajectory.name)}{suffix}" if output.is_dir() else output
        write_trajectory_file(trajectory, str(path), format=args.to)
        if not args.quiet:
            print(f"Wrote {path}")
    return 0


def run_info(args: argparse.Namespace) -> int:
    from trajectopy.io.rosbag import is_ros_bag, list_topics

    for path in args.files:
        if is_ros_bag(path) and Path(path).exists():
            topics = list_topics(path)
            print(f"{path}: {len(topics)} topic(s) with a supported message type")
            for topic, msgtype in topics.items():
                print(f"  {topic}  [{msgtype}]")
        for trajectory in load_trajectories(path, args.topic, args.format):
            print(trajectory)
    return 0


# ------------------------------------------------------------------------------------------------- parser


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="+", help="Trajectory files or ROS bags (.bag, .mcap, .db3, bag directory)")
    parser.add_argument(
        "--topic", action="append", help="Topic to read from ROS bags (repeatable). Defaults to all supported."
    )
    parser.add_argument("--format", choices=FORMATS, default="auto", help="Format of the trajectory files")


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ref", "-r", required=True, help="Reference (ground truth) trajectory file or ROS bag")
    parser.add_argument("--ref-topic", help="Topic of the reference in a ROS bag")
    parser.add_argument("--ref-format", choices=FORMATS, default="auto", help="Format of the reference file")
    parser.add_argument("--settings", "-s", help="JSON processing settings file (see the documentation)")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quiet", "-q", action="store_true", help="Print only warnings and errors")
    parser.add_argument("--verbose", action="store_true", help="Print debug messages")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trajectopy",
        description="Trajectopy - Trajectory Evaluation in Python. Without a command, the GUI is started.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    evaluation_help = {
        "ate": "Absolute trajectory error of one or more trajectories against a reference",
        "rpe": "Relative pose error (drift) of one or more trajectories against a reference",
        "evaluate": "ATE and RPE of one or more trajectories against a reference",
    }
    for name, help_text in evaluation_help.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        _add_input_arguments(sub)
        _add_reference_arguments(sub)
        sub.add_argument("--no-align", action="store_true", help="Evaluate without estimating an alignment first")
        sub.add_argument("--table", "-t", help="Write the summary table to this file")
        sub.add_argument(
            "--table-format", choices=("csv", "markdown", "latex"), default=None, help="Format of the summary table"
        )
        sub.add_argument("--no-ci", action="store_true", help="Leave out the confidence intervals (faster)")
        sub.add_argument("--output-dir", "-o", help="Write the result files (.result) to this directory")
        sub.add_argument("--figures", help="Export all figures to this directory (requires trajectopy[plot])")
        sub.add_argument("--plot-settings", help="JSON plot settings file for figures and the report")
        sub.add_argument("--report", help="Write a self-contained HTML report to this file")
        sub.add_argument("--no-report-figures", action="store_true", help="Leave the figures out of the report")
        _add_common_arguments(sub)

    sub = subparsers.add_parser("align", help="Align trajectories to a reference and print the parameters")
    _add_input_arguments(sub)
    _add_reference_arguments(sub)
    sub.add_argument("--output", "-o", help="Write the aligned trajectory to this file (or directory)")
    sub.add_argument("--to", choices=FORMATS[1:], default="trajectopy", help="Format of the aligned trajectory")
    sub.add_argument("--parameters", help="Write the alignment parameters to this .result file")
    _add_common_arguments(sub)

    sub = subparsers.add_parser("convert", help="Convert trajectories between formats (and coordinate systems)")
    _add_input_arguments(sub)
    sub.add_argument("--to", choices=FORMATS[1:], default="trajectopy", help="Output format")
    sub.add_argument("--output", "-o", required=True, help="Output file (or directory for several trajectories)")
    sub.add_argument("--epsg", type=int, help="Transform the positions to this EPSG code")
    _add_common_arguments(sub)

    sub = subparsers.add_parser("info", help="Show a summary of trajectory files and the topics of ROS bags")
    _add_input_arguments(sub)
    _add_common_arguments(sub)
    return parser


def _table_format(args: argparse.Namespace) -> str:
    if args.table_format:
        return args.table_format
    suffix = Path(args.table).suffix.lower() if args.table else ""
    return {".md": "markdown", ".tex": "latex"}.get(suffix, "csv")


def run(argv: Sequence[str]) -> int:
    """Runs a subcommand. Returns the exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not args.verbose:
        # the library logs every processing step; the command reports only its own progress
        logging.getLogger("trajectopy").setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING if args.quiet else logging.INFO)

    try:
        if args.command in ("ate", "rpe", "evaluate"):
            args.table_format = _table_format(args)
            return run_evaluation(args, with_ate=args.command != "rpe", with_rpe=args.command != "ate")
        if args.command == "align":
            return run_align(args)
        if args.command == "convert":
            return run_convert(args)
        if args.command == "info":
            return run_info(args)
    except CommandError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        if args.verbose:
            raise
        print(f"Error: {error} (run with --verbose for details)", file=sys.stderr)
        return 1
    build_parser().print_help()
    return 2
