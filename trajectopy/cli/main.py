"""CLI entry point for trajectopy."""

import argparse
import sys

from trajectopy.__version__ import __version__ as VERSION


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="trajectopy", description="Trajectopy - Trajectory Evaluation in Python")
    parser.add_argument(
        "files",
        nargs="*",
        help="Trajectory (.traj, .bag) or result (.result) files to open (also used when opening files from Windows).",
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show the version and exit")
    parser.add_argument(
        "--single-thread",
        action="store_true",
        help="Disable multithreading",
        default=sys.gettrace() is not None,
    )
    parser.add_argument(
        "--plot-settings",
        type=str,
        help="Path to a JSON plot settings file that overrides the default plot settings.",
        required=False,
        default="",
    )
    args = parser.parse_args()

    if args.version:
        print(f"Trajectopy {VERSION}")
        return

    try:
        from trajectopy.gui.app import main as gui_main

        exit_code = gui_main(
            single_thread=args.single_thread,
            plot_settings_path=args.plot_settings,
            files=args.files,
        )
        if exit_code is not None:
            sys.exit(exit_code)
    except ImportError:
        print("GUI dependencies not installed. Install with: pip install 'trajectopy[gui]'")
        sys.exit(1)


if __name__ == "__main__":
    main()
