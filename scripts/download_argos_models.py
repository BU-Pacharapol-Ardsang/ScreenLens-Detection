from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from argostranslate import package


DEFAULT_PAIRS = (("en", "th"), ("th", "en"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download bundled Argos Translate model files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where .argosmodel files should be stored.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        metavar="SRC:TGT",
        help="Language pair to download, e.g. en:th. May be specified multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite model files even if they already exist.",
    )
    return parser.parse_args()


def parse_pairs(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_PAIRS)

    pairs: list[tuple[str, str]] = []
    for value in values:
        source, separator, target = value.partition(":")
        source = source.strip().lower()
        target = target.strip().lower()
        if separator != ":" or not source or not target:
            raise SystemExit(f"Invalid --pair value: {value!r}. Expected SRC:TGT.")
        pairs.append((source, target))
    return pairs


def find_available_package(source_code: str, target_code: str):
    available_packages = package.get_available_packages()
    for available_package in available_packages:
        if available_package.from_code == source_code and available_package.to_code == target_code:
            return available_package
    raise SystemExit(f"No Argos package found for {source_code}->{target_code}.")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    package.update_package_index()
    pairs = parse_pairs(args.pair)

    for source_code, target_code in pairs:
        available_package = find_available_package(source_code, target_code)
        download_path = Path(available_package.download()).resolve()
        destination_path = output_dir / download_path.name

        if destination_path.exists() and not args.force:
            print(f"Keeping existing {source_code}->{target_code}: {destination_path}")
            continue

        shutil.copy2(download_path, destination_path)
        print(f"Bundled {source_code}->{target_code}: {destination_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
