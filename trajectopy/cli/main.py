"""CLI entry point for trajectopy."""

import argparse
import sys

from trajectopy.__version__ import __version__ as VERSION


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Trajectopy - Trajectory Evaluation in Python")
    parser.add_argument("--version", "-v", action="store_true")
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Disable multithreading",
        default=getattr(sys, "gettrace", None)(),
    )
    parser.add_argument(
        "--report-settings",
        type=str,
        help="Path to JSON report settings file that will override the default settings.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--mpl-settings",
        type=str,
        help="Path to JSON matplotlib plot settings file that will override the default settings.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--report-path",
        "-o",
        type=str,
        help="Output directory for all reports of one session. If not specified, a temporary directory will be used.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--mapbox-token",
        type=str,
        help="Mapbox token to use Mapbox map styles in trajectory plots.",
        required=False,
        default="",
    )
    args = parser.parse_args()

    if args.version:
        print(f"Trajectopy {VERSION}")
        return

    try:
        from trajectopy.gui.app import main as gui_main

        gui_main(
            single_thread=args.single_thread,
            report_output_path=args.report_path,
            report_settings_path=args.report_settings,
            mpl_plot_settings_path=args.mpl_settings,
            mapbox_token=args.mapbox_token,
        )
    except ImportError:
        print("GUI dependencies not installed. Install with: pip install 'trajectopy[gui]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
