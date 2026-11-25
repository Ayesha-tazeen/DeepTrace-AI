"""CLI: validate (and optionally ingest) uploaded media files.

Usage (from the DeepTrace-AI folder):
    python scripts/validate_media.py data/samples/photo.jpg
    python scripts/validate_media.py clip.mp4 song.wav --ingest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deeptrace.config import get_settings  # noqa: E402
from deeptrace.exceptions import MediaValidationError  # noqa: E402
from deeptrace.ingestion import ingest_media, validate_media  # noqa: E402
from deeptrace.logging_utils import configure_logging  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate media files for DeepTrace AI.")
    parser.add_argument("files", nargs="+", help="One or more files to validate")
    parser.add_argument("--ingest", action="store_true", help="Copy valid files into data/raw/")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level, settings.log_dir)

    exit_code = 0
    for file in args.files:
        try:
            result = ingest_media(file, settings) if args.ingest else validate_media(file, settings)
        except MediaValidationError as exc:
            result = exc.result
        print(result.model_dump_json(indent=2))
        if not result.is_valid:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
