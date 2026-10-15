from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference_audit import audit_references, parse_bibtex  # noqa: E402


class ReferenceAuditTests(unittest.TestCase):
    def test_consistent_wrong_abstain_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ('bibliography.bib', 'reference_audit.csv', 'external_resources.csv')
            for name in names:
                shutil.copyfile(ROOT / name, directory / name)
            for name in names[1:]:
                path = directory / name
                text = path.read_text(encoding='utf-8')
                correct = '10.1214/17-EJS1388'
                self.assertIn(correct, text)
                path.write_text(text.replace(correct, '10.1214/17-EJS1386'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'abstain-option identity mismatch'):
                audit_references(*(directory / name for name in names))

    def test_consistent_wrong_smoothap_identifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ('bibliography.bib', 'reference_audit.csv', 'external_resources.csv')
            for name in names:
                shutil.copyfile(ROOT / name, directory / name)
            correct = '10.1007/978-3-030-58545-7_39'
            wrong = '10.1007/978-3-030-58548-8_40'
            for name in names[1:]:
                path = directory / name
                text = path.read_text(encoding='utf-8')
                self.assertIn(correct, text)
                path.write_text(text.replace(correct, wrong), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Smooth-AP identity mismatch'):
                audit_references(*(directory / name for name in names))

    def test_wrong_smoothap_persistent_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ('bibliography.bib', 'reference_audit.csv', 'external_resources.csv')
            for name in names:
                shutil.copyfile(ROOT / name, directory / name)
            path = directory / names[1]
            with path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            next(r for r in rows if r['bib_key'] == 'brown2020')['persistent_id'] = (
                'https://doi.org/10.1007/978-3-030-58548-8_40')
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'Smooth-AP identity mismatch'):
                audit_references(*(directory / name for name in names))

    def test_checked_in_ledgers_match(self):
        summary = audit_references(
            ROOT / 'bibliography.bib',
            ROOT / 'reference_audit.csv',
            ROOT / 'external_resources.csv',
        )
        self.assertGreaterEqual(summary['entries'], 55)
        self.assertEqual(summary['entries'], len(parse_bibtex(ROOT / 'bibliography.bib')))
        self.assertGreater(summary['full_text_or_claim_sections'], 0)

    def test_title_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bib = directory / 'bibliography.bib'
            audit = directory / 'reference_audit.csv'
            resources = directory / 'external_resources.csv'
            shutil.copyfile(ROOT / 'bibliography.bib', bib)
            shutil.copyfile(ROOT / 'external_resources.csv', resources)
            with (ROOT / 'reference_audit.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            rows[0]['title'] += ' [mutated]'
            with audit.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                audit_references(bib, audit, resources)

    def test_missing_resource_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bib = directory / 'bibliography.bib'
            audit = directory / 'reference_audit.csv'
            resources = directory / 'external_resources.csv'
            shutil.copyfile(ROOT / 'bibliography.bib', bib)
            shutil.copyfile(ROOT / 'reference_audit.csv', audit)
            with (ROOT / 'external_resources.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0])
            removed = next(row['supported_claim'] for row in rows if row['supported_claim'] == 'yue2007')
            rows = [row for row in rows if row['supported_claim'] != removed]
            with resources.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                audit_references(bib, audit, resources)


if __name__ == '__main__':
    unittest.main()
