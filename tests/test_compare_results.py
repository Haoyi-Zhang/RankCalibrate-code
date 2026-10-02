from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compare_results import (  # noqa: E402
    SCIENTIFIC_RESULT_PATHS,
    compare_results,
)


class ResultComparisonTests(unittest.TestCase):
    @staticmethod
    def _write(root: Path, relative: str | Path, content: str) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @classmethod
    def _populate_scientific_tree(cls, root: Path, prefix: str = "") -> None:
        for relative in SCIENTIFIC_RESULT_PATHS:
            cls._write(root, relative, f"{prefix}{relative.as_posix()}\n")

    def test_identical_scientific_trees_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, actual = root / "expected", root / "actual"
            self._populate_scientific_tree(expected)
            self._populate_scientific_tree(actual)
            matched = compare_results(expected, actual)
            self.assertEqual(matched, list(SCIENTIFIC_RESULT_PATHS))
            self.assertEqual(len(matched), 25)

    def test_content_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, actual = root / "expected", root / "actual"
            self._populate_scientific_tree(expected)
            self._populate_scientific_tree(actual)
            self._write(actual, "dimensions.csv", "changed\n")
            with self.assertRaisesRegex(ValueError, "changed.*dimensions.csv"):
                compare_results(expected, actual)

    def test_missing_and_unexpected_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, actual = root / "expected", root / "actual"
            self._populate_scientific_tree(expected)
            self._populate_scientific_tree(actual)
            (actual / "dimensions.csv").unlink()
            self._write(actual, "unexpected.csv", "not in the contract\n")
            with self.assertRaisesRegex(ValueError, "missing.*unexpected"):
                compare_results(expected, actual)

    def test_host_and_retained_observation_records_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, actual = root / "expected", root / "actual"
            self._populate_scientific_tree(expected)
            self._populate_scientific_tree(actual)
            self._write(expected, "resources.json", '{"cpu": 1}\n')
            self._write(actual, "resources.json", '{"cpu": 999}\n')
            self._write(expected, "resources_decoding.json", '{"cpu": 2}\n')
            self._write(actual, "resources_decoding.json", '{"cpu": 888}\n')
            self._write(expected, "reproduction.json", '{"host": "a"}\n')
            self._write(actual, "reproduction.json", '{"host": "b"}\n')
            self._write(expected, "intake_and_pilot.json", '{"retained": true}\n')
            self.assertFalse((actual / "intake_and_pilot.json").exists())
            self.assertEqual(compare_results(expected, actual), list(SCIENTIFIC_RESULT_PATHS))

    def test_real_retained_observation_matches_fresh_scientific_tree(self):
        """Regression for comparing retained results with a truly fresh output tree.

        The fixture starts empty and receives only the 25 regenerable scientific files.
        The retained tree contains the genuine intake/pilot observation, which must neither
        be required nor copied into the fresh tree.
        """
        retained = ROOT / "results"
        self.assertTrue((retained / "intake_and_pilot.json").is_file())
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory)
            self.assertEqual(list(fresh.iterdir()), [])
            for relative in SCIENTIFIC_RESULT_PATHS:
                source = retained / relative
                target = fresh / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            self.assertFalse((fresh / "intake_and_pilot.json").exists())
            matched = compare_results(retained, fresh)
            self.assertEqual(matched, list(SCIENTIFIC_RESULT_PATHS))
            self.assertEqual(len(matched), 25)


if __name__ == "__main__":
    unittest.main()
