"""Module entry point for `python -m rapid_llm`."""

import sys

from rapid_llm import cli


if __name__ == "__main__":
    raise SystemExit(cli.main(sys.argv[1:]))
