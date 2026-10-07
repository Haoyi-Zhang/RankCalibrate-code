"""Portable exact tests of full-moment inversion, not a timing campaign."""
from decimal import Decimal
from fractions import Fraction as F
from itertools import combinations, product
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ranking


def definition_report(law, n):
    """Normalized moments from their probability definition, including zeros."""
    weighted = [(y, mass / y.bit_count())
                for y, mass in enumerate(law, 1) if mass]
    return {term: sum((mass for y, mass in weighted if y & term == term), F(0))
            for term in range(1, 1 << n)}


def alternating_law(mu, n):
    """Independent displayed inverse: enumerate subsets of the missing bits."""
    out = []
    for support in range(1, 1 << n):
        missing = [1 << i for i in range(n) if not support & (1 << i)]
        value = F(0)
        for size in range(len(missing) + 1):
            for extra in combinations(missing, size):
                value += (-1) ** size * F(mu[support | sum(extra)])
        out.append(support.bit_count() * value)
    return out


class FullMomentInversionTests(unittest.TestCase):
    def assert_law(self, law, n):
        report = definition_report(law, n)
        saved = dict(report)
        self.assertEqual(alternating_law(report, n), law)
        self.assertEqual(ranking.moments(law, n, n), report)
        actual = ranking.law_from_full_moments(report, n)
        self.assertEqual(actual, law)
        self.assertTrue(all(type(value) is F for value in actual))
        self.assertEqual(report, saved)

    def test_all_point_masses_through_six_items(self):
        for n in range(2, 7):
            for mask in range(1, 1 << n):
                with self.subTest(n=n, mask=mask):
                    law = [F(int(y == mask)) for y in range(1, 1 << n)]
                    self.assert_law(law, n)

    def test_exact_mixtures_through_eight_items(self):
        for n in range(2, 9):
            full = (1 << n) - 1
            weights = list(range(1, full + 1))
            sparse = [F(0)] * full
            sparse[0], sparse[1], sparse[-1] = F(1, 3), F(1, 2), F(1, 6)
            laws = ([F(1, full)] * full,
                    [F(w, sum(weights)) for w in weights], sparse,
                    [F(y.bit_count(), n * (1 << (n - 1)))
                     for y in range(1, full + 1)])
            for case, law in enumerate(laws):
                with self.subTest(n=n, case=case):
                    self.assert_law(law, n)

    def test_complete_report_grid_and_infeasible_laws(self):
        reports = [dict(zip((1, 2, 3), row))
                   for row in product((F(-1), F(0), F(1, 2), F(1)), repeat=3)]
        reports += [definition_report(law, 2) for law in
                    ([F(-1, 4), F(1, 2), F(3, 4)],
                     [F(1), F(1), F(0)], [F(0)] * 3)]
        for report in reports:
            expected = alternating_law(report, 2)
            with self.subTest(report=report):
                if min(expected) < 0 or sum(expected) != 1:
                    with self.assertRaisesRegex(ValueError, "do not define a probability law"):
                        ranking.law_from_full_moments(report, 2)
                else:
                    self.assertEqual(ranking.law_from_full_moments(report, 2), expected)

    def test_invalid_incomplete_and_zero_mask_reports(self):
        incomplete = "every nonempty normalized moment is required"
        keys = "moment keys must be positive Boolean masks"
        exact = "moment value must be an exact rational value"
        cases = [(None, "moment report must be a mapping"),
                 ([], "moment report must be a mapping"), ({}, incomplete),
                 ({1: F(1), 2: F(0)}, incomplete),
                 ({1: F(1), 2: F(0), 3: F(0), 4: F(0)}, incomplete),
                 ({0: F(0)}, keys), ({-1: F(0)}, keys),
                 ({True: F(0)}, keys), ({"1": F(0)}, keys)]
        cases += [({1: value}, exact)
                  for value in (True, 0.5, 1j, "bad", "1/0", None)]
        # Value validation precedes the report's completeness check.
        cases.append(({1: F(1), 2: F(0), 3: F(0), 4: 0.5}, exact))
        for report, message in cases:
            with self.subTest(report=report):
                with self.assertRaises(ValueError) as caught:
                    ranking.law_from_full_moments(report, 2)
                self.assertEqual(str(caught.exception), message)
        for n, message in ((True, "n must be an integer"),
                           (2.0, "n must be an integer"),
                           (1, "n must be at least 2"),
                           (-1, "n must be at least 2")):
            with self.subTest(n=n):
                with self.assertRaises(ValueError) as caught:
                    ranking.law_from_full_moments(None, n)
                self.assertEqual(str(caught.exception), message)

    def test_exact_scalar_forms_and_mapping_order(self):
        law = [F(1, 4), F(1, 4), F(1, 2)]
        expected = definition_report(law, 2)
        for report in ({3: "1/4", 2: Decimal("0.5"), 1: F(1, 2)},
                       {1: "0.5", 2: F(1, 2), 3: Decimal("0.25")}):
            saved = dict(report)
            self.assertEqual({key: F(value) for key, value in report.items()}, expected)
            actual = ranking.law_from_full_moments(report, 2)
            self.assertEqual(actual, law)
            self.assertTrue(all(type(value) is F for value in actual))
            self.assertEqual(report, saved)
        self.assert_law([F(1), F(0), F(0)], 2)
        self.assertEqual(ranking.law_from_full_moments({3: 0, 2: 0, 1: 1}, 2),
                         [F(1), F(0), F(0)])

    def test_single_validation_and_fresh_output(self):
        law = [F(1, 3), F(1, 2), F(1, 6)]
        report = definition_report(law, 2)
        saved = dict(report)
        with mock.patch.object(ranking, "_validated_moments",
                               wraps=ranking._validated_moments) as validate:
            first = ranking.law_from_full_moments(report, 2)
            validate.assert_called_once_with(report)
        second = ranking.law_from_full_moments(report, 2)
        self.assertIsNot(first, second)
        first[0] = F(99)
        self.assertEqual(second, law)
        self.assertEqual(report, saved)
        self.assertEqual(len(second), 3)  # no public empty-outcome slot


if __name__ == "__main__":
    unittest.main()
