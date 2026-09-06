"""Command-line interface for residual analysis."""

import argparse

from . import __version__


def main(argv=None):
    """Run a command and return its process exit status."""
    parser = argparse.ArgumentParser(
        prog="hera-systematics", description="Cylindrical power-spectrum analysis"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args(argv)
    parser.print_help()
    return 0
