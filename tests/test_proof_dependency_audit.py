from __future__ import annotations

import csv
import unittest
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


class ProofDependencyAuditTests(unittest.TestCase):
    def test_dependency_rows_are_complete_and_unique(self):
        path = ROOT / "proof_dependency_audit.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["dependency_id"] for row in rows], ["D1", "D2", "D3"])
        for row in rows:
            for field, value in row.items():
                self.assertTrue(value.strip(), f"blank {field} in {row['dependency_id']}")
            checked = date.fromisoformat(row["last_checked"])
            self.assertLessEqual(checked, date.today())

    def test_imported_lower_bound_matches_reference_ledger(self):
        with (ROOT / "proof_dependency_audit.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            dependencies = {
                row["dependency_id"]: row for row in csv.DictReader(handle)
            }
        lower_bound = dependencies["D1"]
        self.assertIn("Theorem 16", lower_bound["external_or_standard_result"])
        self.assertIn("ramaswamy2016", lower_bound["source_or_status"])
        self.assertIn("full support", lower_bound["local_discharge"].lower())
        self.assertIn("centered contrasts", lower_bound["local_discharge"].lower())

        with (ROOT / "reference_audit.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            references = {row["bib_key"]: row for row in csv.DictReader(handle)}
        source = references["ramaswamy2016"]
        self.assertEqual(
            source["primary_url"],
            "https://jmlr.org/papers/volume17/14-316/14-316.pdf",
        )
        self.assertEqual(source["content_check"], "full_text_or_claim_sections_checked")


if __name__ == "__main__":
    unittest.main()
