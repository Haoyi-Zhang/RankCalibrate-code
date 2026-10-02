#!/usr/bin/env python3
"""Byte-compare the fixed manifest of regenerable scientific outputs.

The verifier deterministically regenerates exactly 25 claim-relevant CSV/JSON files.
Host/resource observations, the clean-reproduction record, and the retained intake/pilot
measurement are intentionally outside that contract and are never copied into a fresh run.
"""
from __future__ import annotations

import argparse
from pathlib import Path


SCIENTIFIC_RESULT_PATHS: tuple[Path, ...] = tuple(
    sorted(
        (
            Path("coefficient_bounds.csv"),
            Path("decoding.csv"),
            Path("dimensions.csv"),
            Path("extended_geometry.csv"),
            Path("fee_deficiency.csv"),
            Path("fiber_actions.csv"),
            Path("fiber_certificate.json"),
            Path("fiber_curve.csv"),
            Path("fiber_dominance.json"),
            Path("parity.csv"),
            Path("parity_menus.csv"),
        )
        + tuple(
            Path("matrices") / f"n{n}_b{b}.csv"
            for n in range(2, 6)
            for b in range(1, n + 1)
        )
    )
)
SCIENTIFIC_RESULT_SET = frozenset(SCIENTIFIC_RESULT_PATHS)
NONREGENERATED_OBSERVATIONS = frozenset(
    {Path("intake_and_pilot.json"), Path("reproduction.json")}
)


def _is_run_observation(relative: Path) -> bool:
    """Return whether *relative* is retained metadata, not a scientific output."""
    return (
        relative in NONREGENERATED_OBSERVATIONS
        or (
            relative.parent == Path(".")
            and relative.name.startswith("resources")
            and relative.suffix == ".json"
        )
    )


def deterministic_files(root: Path) -> dict[Path, bytes]:
    """Load exactly the regenerable scientific-output manifest from *root*.

    Missing manifest entries and unexpected non-observation files fail closed.  Retained
    intake/pilot measurements and run-specific resource/reproduction records are ignored.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"result directory does not exist: {root}")

    present: dict[Path, bytes] = {}
    unexpected: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative in SCIENTIFIC_RESULT_SET:
            present[relative] = path.read_bytes()
        elif not _is_run_observation(relative):
            unexpected.append(relative)

    missing = sorted(SCIENTIFIC_RESULT_SET - set(present))
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append("missing: " + ", ".join(map(str, missing)))
        if unexpected:
            parts.append("unexpected: " + ", ".join(map(str, unexpected)))
        raise ValueError(
            f"scientific result contract mismatch in {root} (" + "; ".join(parts) + ")"
        )
    return {relative: present[relative] for relative in SCIENTIFIC_RESULT_PATHS}


def compare_results(expected: Path, actual: Path) -> list[Path]:
    expected_files = deterministic_files(expected)
    actual_files = deterministic_files(actual)
    changed = [
        name
        for name in SCIENTIFIC_RESULT_PATHS
        if expected_files[name] != actual_files[name]
    ]
    if changed:
        raise ValueError(
            "scientific result mismatch (changed: "
            + ", ".join(map(str, changed))
            + ")"
        )
    return list(SCIENTIFIC_RESULT_PATHS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    args = parser.parse_args()
    matched = compare_results(args.expected, args.actual)
    print(f"scientific result comparison passed: {len(matched)} files")


if __name__ == "__main__":
    main()
