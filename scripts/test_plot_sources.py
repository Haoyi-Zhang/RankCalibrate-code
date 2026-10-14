"""Exact plotted fiber/fee coverage, including negative and zero endpoints."""
from pathlib import Path
from fractions import Fraction as Q
import importlib.util
import json
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('plot_retained', ROOT / 'scripts/plot_retained.py')
ROUTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTE)

class PlotSources(unittest.TestCase):
    def test_entire_fiber_retains_negative_zero_and_positive_parameters(self):
        fiber, _ = ROUTE.validate_inputs(ROOT / 'results')
        fixture = json.loads((ROOT / 'fixtures/fiber.json').read_text())
        parameters = [Q(r['t']) for r in fiber]
        self.assertEqual([parameters[0], parameters[-1]], list(map(Q, fixture['parameter_interval'])))
        self.assertIn(Q(0), parameters)
        self.assertTrue(any(t < 0 for t in parameters))
        self.assertTrue(any(t > 0 for t in parameters))
        for row in fiber:
            t = Q(row['t'])
            p = [Q(a, fixture['base_denominator']) + t*b
                 for a, b in zip(fixture['base_numerators'], fixture['direction'])]
            self.assertEqual(sum(p), 1)
            self.assertGreaterEqual(min(p), 0)

    def test_deficiency_retains_both_zero_endpoints_and_peak(self):
        _, fee = ROUTE.validate_inputs(ROOT / 'results')
        self.assertEqual([Q(fee[i]['pair_fee']) for i in (0, -1)], [Q(3, 80), Q(1, 20)])
        for row in (fee[0], fee[-1]):
            self.assertEqual(Q(row['deterministic']), 0)
            self.assertEqual(Q(row['randomized']), 0)
        peak = fee[10]
        self.assertEqual(Q(peak['pair_fee']), Q(7, 160))
        self.assertEqual(Q(peak['deterministic']), Q(1, 160))
        self.assertEqual(Q(peak['randomized']), Q(1, 320))
        for row in fee[1:-1]:
            self.assertGreater(Q(row['randomized']), 0)
            self.assertLess(Q(row['randomized']), Q(row['deterministic']))

    def test_positive_only_input_and_changed_float_are_rejected(self):
        fiber, fee = ROUTE.validate_inputs(ROOT / 'results')
        with patch.object(ROUTE, 'read_rows', return_value=[r for r in fiber if Q(r['t']) > 0]):
            with self.assertRaisesRegex(ValueError, 'thirty-one'):
                ROUTE.validate_inputs(ROOT / 'results')
        changed = [dict(r) for r in fiber]
        changed[0]['t_float'] = '0'
        with patch.object(ROUTE, 'read_rows', side_effect=[changed, fee]):
            with self.assertRaisesRegex(ValueError, 'coordinate mismatch'):
                ROUTE.validate_inputs(ROOT / 'results')

    def test_output_cannot_overwrite_paper_or_retained_figures(self):
        for destination in (ROOT.parent / 'paper/figures', ROOT / 'figures', ROOT / 'results'):
            with self.assertRaises(ValueError):
                ROUTE.isolated_output(destination, ROOT / 'results')

if __name__ == '__main__':
    unittest.main()
