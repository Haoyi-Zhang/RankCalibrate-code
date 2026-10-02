from __future__ import annotations

import sys
import unittest
from fractions import Fraction as F
from itertools import islice, permutations
from math import comb
from pathlib import Path
from unittest import mock

# Resolve the implementation relative to this file, so discovery works both
# from the standalone repository and from the complete-project root.
HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import ranking as ranking_module  # noqa: E402
from ranking import (  # noqa: E402
    action_fee,
    block_reward,
    average_precision,
    contrast_basis,
    contrast_coefficients,
    contrast_features,
    decode,
    dimension,
    exact_rank,
    gamma,
    harmonic,
    law_from_full_moments,
    mobius_coefficients,
    moments,
    oracle_utility,
    newton_coefficients,
    numerator,
    ordered_partitions,
    reference_distribution,
    refinements,
    reference_fee,
    sorted_utility,
    validate_action,
)


class ExactUnits(unittest.TestCase):
    def test_average_precision_and_validation(self):
        self.assertEqual(average_precision((0, 1, 2), 5), F(5, 6))
        self.assertEqual(average_precision((2, 1, 0), 1), F(1, 3))
        with self.assertRaises(ValueError):
            average_precision((0, 1), 0)
        with self.assertRaises(ValueError):
            average_precision((0, 0), 1)

    def test_ordered_partition_counts(self):
        self.assertEqual(len(list(ordered_partitions((0, 1, 2), 2))), 12)
        self.assertEqual(len(list(ordered_partitions((0, 1, 2), 3))), 13)
        self.assertEqual(len(list(ordered_partitions((0, 1, 2, 3), 1))), 24)

    def test_dimension_saturation_boundary(self):
        self.assertEqual(dimension(2, 1), 1)
        self.assertEqual(dimension(2, 2), 2)
        self.assertEqual(dimension(3, 2), 6)
        self.assertNotEqual(dimension(2, 1), (1 << 2) - 2)
        for n in range(2, 9):
            for b in range(1, n + 1):
                self.assertEqual(
                    dimension(n, b) == (1 << n) - 2,
                    b >= max(2, n - 1),
                )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            list(ordered_partitions((0,), 0))
        with self.assertRaises(ValueError):
            validate_action(((0,), (0,)), 2)
        with self.assertRaises(ValueError):
            moments([F(1)], 2, 2)
        with self.assertRaises(ValueError):
            dimension(1, 1)


    def test_public_boundary_validation(self):
        with self.assertRaises(ValueError):
            list(ordered_partitions([0, 1], 1))
        with self.assertRaises(ValueError):
            list(ordered_partitions((0, 0), 1))
        with self.assertRaises(ValueError):
            validate_action(((0,), ("1",)), 2)
        with self.assertRaises(ValueError):
            harmonic(-1, 1)
        with self.assertRaises(ValueError):
            gamma(0, 0)
        with self.assertRaises(ValueError):
            reference_distribution(1)
        with self.assertRaises(ValueError):
            reference_fee(3, 2, 2)
        with self.assertRaises(ValueError):
            moments([F(1, 3), F(1, 3), F(1, 3)], 2, 3)
        with self.assertRaises(ValueError):
            exact_rank([[1, 0], [1]])
        with self.assertRaises(ValueError):
            action_fee(((0,),), lambda _t, _s: F(-1))

    def test_inexact_scalar_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            moments([0.25, 0.25, 0.5], 2, 2)
        with self.assertRaises(ValueError):
            moments([True, F(0), F(0)], 2, 2)
        with self.assertRaises(ValueError):
            exact_rank([[F(1), 0.5]])
        with self.assertRaises(ValueError):
            action_fee(((0,),), lambda _t, _s: 0.0)

        # Decimal strings remain exact and are intentionally accepted.
        self.assertEqual(moments(["1/4", "1/4", "1/2"], 2, 1)[1], F(1, 2))

    def test_definition_oracle_matches_sorted_completion(self):
        for n in range(2, 6):
            for b in range(1, n + 1):
                for action in ordered_partitions(tuple(range(n)), b):
                    self.assertTrue(list(refinements(action)))
                    for y in range(1, 1 << n):
                        self.assertEqual(oracle_utility(action, y), sorted_utility(action, y))

    def test_relabeling_equivariance(self):
        for n in range(2, 5):
            labelings = list(islice(permutations(range(n)), 4))
            actions = list(islice(ordered_partitions(tuple(range(n)), n), 18))
            for labels in labelings:
                for action in actions:
                    relabeled_action = tuple(
                        tuple(labels[item] for item in block) for block in action
                    )
                    for y in range(1, 1 << n):
                        relabeled_y = sum(
                            1 << labels[item] for item in range(n) if y >> item & 1
                        )
                        self.assertEqual(
                            sorted_utility(action, y),
                            sorted_utility(relabeled_action, relabeled_y),
                        )
                        self.assertEqual(
                            numerator(action, y),
                            numerator(relabeled_action, relabeled_y),
                        )

    def test_merging_adjacent_blocks_weakly_improves_oracle_utility(self):
        n = 4
        for action in ordered_partitions(tuple(range(n)), 2):
            for index in range(len(action) - 1):
                merged = (
                    action[:index]
                    + (tuple(sorted(action[index] + action[index + 1])),)
                    + action[index + 2:]
                )
                for y in range(1, 1 << n):
                    self.assertGreaterEqual(
                        oracle_utility(merged, y), oracle_utility(action, y)
                    )

    def test_report_and_fee_fail_closed(self):
        n, b = 3, 2
        law = reference_distribution(n)
        report = moments(law, n, b + 1)
        incomplete = dict(report)
        incomplete.pop(7)
        with self.assertRaises(ValueError):
            decode(n, b, incomplete, lambda _t, _s: F(0))
        with self.assertRaises(ValueError):
            decode(n, b, report, lambda _t, _s: F(-1))
        with self.assertRaises(ValueError):
            block_reward(1, 1, report)
        with self.assertRaises(ValueError):
            block_reward(1, 2, {1: F(1)})

    def test_harmonic_newton_identity(self):
        # gamma(t,s) is the s-th Newton coefficient of r -> H_t(r).
        for t in range(7):
            for s in range(1, 7):
                difference = sum(
                    ((-1) ** (s - j) * comb(s, j) * harmonic(t, j)
                     for j in range(s + 1)),
                    F(0),
                )
                self.assertEqual(difference, gamma(t, s))

    def test_reference_pair_price(self):
        for n in range(2, 13):
            for t in range(n - 1):
                self.assertEqual(reference_fee(n, t, 2), F(1, 4 * n * (t + 1)))
                self.assertEqual(reference_fee(n, t, 1), 0)

    def test_reference_prices_are_nonnegative(self):
        for n in range(2, 13):
            for t in range(n):
                for s in range(1, n - t + 1):
                    price = reference_fee(n, t, s)
                    self.assertGreaterEqual(price, 0)
                    if s >= 2:
                        self.assertGreater(price, 0)

    def test_reference_moments(self):
        for n in range(2, 8):
            mu = moments(reference_distribution(n), n, n)
            for term, value in mu.items():
                self.assertEqual(value, F(1, n * 2 ** (term.bit_count() - 1)))

    def test_reference_action_fee_is_expected_resolution_gain(self):
        for n in range(2, 6):
            law = reference_distribution(n)
            fee = lambda t, s, n=n: reference_fee(n, t, s)
            for b in range(1, n + 1):
                for action in ordered_partitions(tuple(range(n)), b):
                    fixed_refinement = tuple(i for block in action for i in block)
                    expected_gain = sum(
                        (
                            law[y - 1]
                            * (
                                sorted_utility(action, y)
                                - average_precision(fixed_refinement, y)
                            )
                            for y in range(1, 1 << n)
                        ),
                        F(0),
                    )
                    self.assertEqual(action_fee(action, fee), expected_gain)

    def test_cubic_sign_and_boolean_inversion(self):
        action = ((0,), (1, 2))
        coefficients = newton_coefficients(action, 3)
        self.assertEqual(coefficients[7], -F(1, 6))
        self.assertEqual(
            coefficients,
            mobius_coefficients([numerator(action, y) for y in range(8)], 3),
        )

    def test_degree_bound_for_all_small_actions(self):
        for n in range(2, 5):
            for b in range(1, n + 1):
                degree = min(n, b + 1)
                for action in ordered_partitions(tuple(range(n)), b):
                    coefficients = newton_coefficients(action, n)
                    for term, value in enumerate(coefficients):
                        if term.bit_count() > degree:
                            self.assertEqual(value, 0)

    def test_contrast_basis_dimension_and_rank(self):
        for n in range(2, 6):
            for b in range(1, n + 1):
                basis = contrast_basis(n, b)
                self.assertEqual(len(basis), dimension(n, b))
                rows = [contrast_features(n, b, y) for y in range(1, 1 << n)]
                rank, _ = exact_rank(rows)
                self.assertEqual(rank, dimension(n, b))

    def test_contrast_coefficients_reconstruct_every_small_action(self):
        for n in range(2, 5):
            for b in range(1, n + 1):
                actions = list(ordered_partitions(tuple(range(n)), b))
                reference = actions[0]
                reference_values = [
                    sorted_utility(reference, y) for y in range(1, 1 << n)
                ]
                features = [
                    contrast_features(n, b, y) for y in range(1, 1 << n)
                ]
                for action in actions:
                    coefficients = contrast_coefficients(action, reference, n, b)
                    for y, feature_row in enumerate(features, 1):
                        reconstructed = sum(
                            (coefficient * feature for coefficient, feature in zip(
                                coefficients, feature_row
                            )),
                            F(0),
                        )
                        self.assertEqual(
                            reconstructed,
                            sorted_utility(action, y) - reference_values[y - 1],
                        )

    def test_reference_tie_has_expected_active_face_rank(self):
        for n in range(2, 5):
            reference_law = reference_distribution(n)
            for b in range(1, n + 1):
                actions = list(ordered_partitions(tuple(range(n)), b))
                matrix = [
                    [sorted_utility(action, y) for y in range(1, 1 << n)]
                    for action in actions
                ]
                contrasts = [
                    [value - base for value, base in zip(row, matrix[0])]
                    for row in matrix[1:]
                ]
                centered = [
                    [
                        value - sum(
                            (probability * coordinate for probability, coordinate in zip(
                                reference_law, row
                            )),
                            F(0),
                        )
                        for value in row
                    ]
                    for row in contrasts
                ]
                rank, _ = exact_rank(centered)
                augmented_rank, _ = exact_rank(
                    [[F(1)] * ((1 << n) - 1)] + centered
                )
                self.assertEqual(rank, dimension(n, b))
                self.assertEqual(augmented_rank, dimension(n, b) + 1)

    def test_parity_laws_match_low_order_moments(self):
        for b in range(2, 7):
            n = b + 1
            normalizer = n * 2 ** (n - 2)
            plus = [
                F(y.bit_count(), normalizer)
                if (n - y.bit_count()) % 2 == 0
                else F(0)
                for y in range(1, 1 << n)
            ]
            minus = [
                F(y.bit_count(), normalizer)
                if (n - y.bit_count()) % 2 == 1
                else F(0)
                for y in range(1, 1 << n)
            ]
            self.assertEqual(sum(plus), 1)
            self.assertEqual(sum(minus), 1)
            self.assertEqual(moments(plus, n, b), moments(minus, n, b))

    def test_decoder_matches_bruteforce_and_transition_count(self):
        n, b = 4, 2
        weights = [y % 5 + 1 for y in range(1, 1 << n)]
        law = [F(weight, sum(weights)) for weight in weights]
        report = moments(law, n, b + 1)
        fee = lambda t, s: F(comb(s, 2), 24)
        best, action, transitions = decode(n, b, report, fee)
        brute = min(
            1
            - sum(
                (law[y - 1] * sorted_utility(candidate, y)
                 for y in range(1, 1 << n)),
                F(0),
            )
            + action_fee(candidate, fee)
            for candidate in ordered_partitions(tuple(range(n)), b)
        )
        self.assertEqual(best, brute)
        self.assertEqual(
            best,
            1
            - sum(
                (law[y - 1] * sorted_utility(action, y)
                 for y in range(1, 1 << n)),
                F(0),
            )
            + action_fee(action, fee),
        )
        self.assertEqual(
            transitions,
            sum(comb(n, s) * 2 ** (n - s) for s in range(1, b + 1)),
        )

    def test_decoder_validates_moment_report_once(self):
        n, b = 4, 2
        report = moments(reference_distribution(n), n, b + 1)
        original = ranking_module._validated_moments
        with mock.patch.object(
            ranking_module, "_validated_moments", wraps=original
        ) as validator:
            decode(n, b, report, lambda _t, _s: F(0))
        self.assertEqual(validator.call_count, 1)


class MomentInversionTests(unittest.TestCase):
    def test_full_moment_inversion(self):
        for n in range(2, 7):
            weights = [y % 7 + 1 for y in range(1, 1 << n)]
            law = [F(weight, sum(weights)) for weight in weights]
            self.assertEqual(law_from_full_moments(moments(law, n, n), n), law)
        invalid = {1: F(1, 2), 2: F(1, 2), 3: F(3, 4)}
        with self.assertRaises(ValueError):
            law_from_full_moments(invalid, 2)


if __name__ == '__main__':
    unittest.main()
