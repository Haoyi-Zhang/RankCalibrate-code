"""Exact arithmetic for AP ranking with oracle-resolved ordered blocks.

Only the Python standard library is required. Items are 0-based integer indices;
outcome masks are integers 1,...,2**n-1. Every public boundary validates its
finite-domain inputs and every scientific quantity uses ``fractions.Fraction``.
"""
from __future__ import annotations

from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import comb, factorial
from typing import Callable, Iterable, Iterator, Mapping, Sequence

Action = tuple[tuple[int, ...], ...]
Fee = Callable[[int, int], F]


def _integer(name: str, value: object, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _fraction(name: str, value: object) -> F:
    # ``Fraction(float)`` is exact only for the binary floating-point value,
    # not for the decimal quantity a caller usually intended.  Scientific
    # inputs therefore fail closed on inexact scalar types.  Integers,
    # Fraction instances, decimal/rational strings, and Decimal-like exact
    # values remain accepted through Fraction's constructor.
    if isinstance(value, (bool, float, complex)):
        raise ValueError(f"{name} must be an exact rational value")
    try:
        return F(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be an exact rational value") from error


def ordered_partitions(items: tuple[int, ...], b: int) -> Iterator[Action]:
    """Yield every ordered set partition of ``items`` with block size at most b."""
    b = _integer("block cap", b, 1)
    if not isinstance(items, tuple):
        raise ValueError("items must be a tuple of distinct nonnegative integers")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in items):
        raise ValueError("items must be distinct nonnegative integers")
    if len(set(items)) != len(items):
        raise ValueError("items must be distinct")
    if not items:
        yield ()
        return
    for s in range(1, min(b, len(items)) + 1):
        for block in combinations(items, s):
            remaining = tuple(i for i in items if i not in block)
            for tail in ordered_partitions(remaining, b):
                yield (block,) + tail


def validate_action(action: Action, n: int) -> None:
    """Require an ordered partition of exactly ``range(n)`` into nonempty blocks."""
    n = _integer("n", n, 0)
    if not isinstance(action, tuple) or any(not isinstance(block, tuple) for block in action):
        raise ValueError("an action must be a tuple of tuple blocks")
    if any(not block for block in action):
        raise ValueError("action blocks must be nonempty")
    flat = [i for block in action for i in block]
    if any(isinstance(i, bool) or not isinstance(i, int) for i in flat):
        raise ValueError("action items must be integer indices")
    if sorted(flat) != list(range(n)):
        raise ValueError("an action must partition all n items exactly once")


def _action_size(action: Action) -> int:
    if not isinstance(action, tuple):
        raise ValueError("an action must be a tuple of tuple blocks")
    return sum(len(block) if isinstance(block, tuple) else 0 for block in action)


def average_precision(order: Sequence[int], y: int) -> F:
    n = len(order)
    if any(isinstance(i, bool) or not isinstance(i, int) for i in order):
        raise ValueError("order must contain integer item indices")
    y = _integer("relevance mask", y, 1)
    if sorted(order) != list(range(n)) or y >= 1 << n:
        raise ValueError("invalid permutation or nonempty relevance mask")
    seen = 0
    numerator_value = F(0)
    for k, i in enumerate(order, 1):
        if y >> i & 1:
            seen += 1
            numerator_value += F(seen, k)
    return numerator_value / y.bit_count()


def refinements(action: Action) -> Iterator[tuple[int, ...]]:
    n = _action_size(action)
    validate_action(action, n)
    for blocks in product(*(permutations(block) for block in action)):
        yield tuple(i for block in blocks for i in block)


def oracle_utility(action: Action, y: int) -> F:
    """Independent definition-based oracle: enumerate every block refinement."""
    n = _action_size(action)
    validate_action(action, n)
    y = _integer("relevance mask", y, 1)
    if y >= 1 << n:
        raise ValueError("relevance mask is outside the action's outcome space")
    return max(average_precision(order, y) for order in refinements(action))


def sorted_utility(action: Action, y: int) -> F:
    n = _action_size(action)
    validate_action(action, n)
    y = _integer("relevance mask", y, 1)
    if y >= 1 << n:
        raise ValueError("relevance mask is outside the action's outcome space")
    order = tuple(
        i
        for block in action
        for i in sorted(block, key=lambda i: (-(y >> i & 1), i))
    )
    return average_precision(order, y)


def harmonic(t: int, r: int) -> F:
    t = _integer("t", t, 0)
    r = _integer("r", r, 0)
    return sum((F(1, t + k) for k in range(1, r + 1)), F(0))


def gamma(t: int, s: int) -> F:
    t = _integer("t", t, 0)
    s = _integer("s", s, 1)
    return F((-1) ** (s - 1) * factorial(s - 1) * factorial(t), factorial(t + s))


def numerator(action: Action, y: int) -> F:
    """Closed-form AP numerator, including the algebraically useful mask y=0."""
    n = _action_size(action)
    validate_action(action, n)
    y = _integer("relevance mask", y, 0)
    if y >= 1 << n:
        raise ValueError("relevance mask is outside the action's Boolean cube")
    t = a = 0
    total = F(0)
    for block in action:
        r = sum(y >> i & 1 for i in block)
        total += r + (a - t) * harmonic(t, r)
        t += len(block)
        a += r
    return total


def mobius_coefficients(values: Sequence[F], n: int) -> list[F]:
    n = _integer("n", n, 0)
    if len(values) != 1 << n:
        raise ValueError("one value per Boolean mask is required")
    out = [_fraction("Boolean table value", value) for value in values]
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                out[mask] -= out[mask ^ (1 << i)]
    return out


def submasks(mask: int) -> Iterator[int]:
    mask = _integer("mask", mask, 0)
    term = mask
    while term:
        yield term
        term = (term - 1) & mask


def newton_coefficients(action: Action, n: int) -> list[F]:
    """Build the numerator polynomial by Newton expansion, not table inversion."""
    n = _integer("n", n, 0)
    validate_action(action, n)
    out = [F(0) for _ in range(1 << n)]
    prefix: list[int] = []
    for block in action:
        for i in block:
            out[1 << i] += 1
        mask = sum(1 << i for i in block)
        for term in submasks(mask):
            coefficient = gamma(len(prefix), term.bit_count())
            out[term] -= len(prefix) * coefficient
            for i in prefix:
                out[term | 1 << i] += coefficient
        prefix.extend(block)
    return out


def exact_rank(rows: Iterable[Sequence[F]]) -> tuple[int, int]:
    """Fraction Gaussian elimination; also return largest observed bit length."""
    pivots: dict[int, list[F]] = {}
    width: int | None = None
    bits = 1
    for source in rows:
        row = [_fraction("matrix entry", value) for value in source]
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("all matrix rows must have the same length")
        for j, pivot in sorted(pivots.items()):
            if row[j]:
                coefficient = row[j]
                row = [x - coefficient * z for x, z in zip(row, pivot)]
                bits = max(
                    bits,
                    *(max(abs(x.numerator).bit_length(), x.denominator.bit_length())
                      for x in row),
                )
        j = next((i for i, value in enumerate(row) if value), None)
        if j is not None:
            coefficient = row[j]
            row = [value / coefficient for value in row]
            pivots[j] = row
            bits = max(
                bits,
                *(max(abs(x.numerator).bit_length(), x.denominator.bit_length())
                  for x in row),
            )
    return len(pivots), bits


def dimension(n: int, b: int) -> int:
    n = _integer("n", n, 2)
    b = _integer("block cap", b, 1)
    if b > n:
        raise ValueError("require 1 <= b <= n")
    if b == 1:
        return n + comb(n, 2) - 2
    return sum(comb(n, s) for s in range(1, min(n, b + 1) + 1)) - 1


def reference_distribution(n: int) -> list[F]:
    n = _integer("n", n, 2)
    return [F(y.bit_count(), n * 2 ** (n - 1)) for y in range(1, 1 << n)]


def reference_fee(n: int, t: int, s: int) -> F:
    """Fixed reference-value fee. No true conditional distribution is used."""
    n = _integer("n", n, 2)
    t = _integer("prefix size", t, 0)
    s = _integer("block size", s, 1)
    if t + s > n:
        raise ValueError("the prefix and block must fit within n items")
    expected_harmonic = sum(
        (F(comb(s, r), 2 ** s) * harmonic(t, r) for r in range(s + 1)),
        F(0),
    )
    return (s - harmonic(t, s) - 2 * t * expected_harmonic) / (2 * n)


def action_fee(action: Action, fee: Fee) -> F:
    n = _action_size(action)
    validate_action(action, n)
    if not callable(fee):
        raise ValueError("fee must be callable")
    t = 0
    total = F(0)
    for block in action:
        value = _fraction("fee value", fee(t, len(block)))
        if value < 0:
            raise ValueError("fees must be nonnegative")
        total += value
        t += len(block)
    return total


def moments(p: Sequence[F], n: int, d: int) -> dict[int, F]:
    n = _integer("n", n, 2)
    d = _integer("moment degree", d, 0)
    if d > n:
        raise ValueError("moment degree cannot exceed n")
    values = [_fraction("probability", value) for value in p]
    if len(values) != (1 << n) - 1 or any(value < 0 for value in values) or sum(values) != 1:
        raise ValueError("p must be a distribution on the nonzero Boolean cube")
    return {
        term: sum(
            (values[y - 1] / y.bit_count()
             for y in range(1, 1 << n) if y & term == term),
            F(0),
        )
        for term in range(1, 1 << n)
        if term.bit_count() <= d
    }


def _validated_moments(mu: Mapping[int, F]) -> dict[int, F]:
    if not isinstance(mu, Mapping):
        raise ValueError("moment report must be a mapping")
    out: dict[int, F] = {}
    for key, value in mu.items():
        if isinstance(key, bool) or not isinstance(key, int) or key <= 0:
            raise ValueError("moment keys must be positive Boolean masks")
        out[key] = _fraction("moment value", value)
    return out


def _block_reward_validated(prefix: int, block: int, values: Mapping[int, F]) -> F:
    """Evaluate one DP transition from an already validated moment mapping."""
    prefix = _integer("prefix mask", prefix, 0)
    block = _integer("block mask", block, 1)
    if prefix & block:
        raise ValueError("prefix and block masks must be disjoint")
    ids = [i for i in range(prefix.bit_length()) if prefix >> i & 1]
    required = {1 << i for i in range(block.bit_length()) if block >> i & 1}
    if ids:
        for term in submasks(block):
            required.add(term)
            required.update(term | 1 << i for i in ids)
    # Probe just this transition's masks. Copying all report keys here adds
    # an O(m_d) traversal to every edge of the subset graph.
    missing = sorted(term for term in required if term not in values)
    if missing:
        raise ValueError(f"moment report is missing masks {missing}")
    value = sum(
        (values[1 << i] for i in range(block.bit_length()) if block >> i & 1),
        F(0),
    )
    t = len(ids)
    if t:
        for term in submasks(block):
            value += gamma(t, term.bit_count()) * (
                sum((values[term | 1 << i] for i in ids), F(0)) - t * values[term]
            )
    return value


def block_reward(prefix: int, block: int, mu: Mapping[int, F]) -> F:
    """Expected normalized AP utility contribution of one block transition."""
    return _block_reward_validated(prefix, block, _validated_moments(mu))


def decode(n: int, b: int, mu: Mapping[int, F], fee: Fee) -> tuple[F, Action, int]:
    """Exact subset DP. Return minimum loss, one optimal action, transitions."""
    n = _integer("n", n, 2)
    b = _integer("block cap", b, 1)
    if b > n:
        raise ValueError("require 1 <= b <= n")
    if not callable(fee):
        raise ValueError("fee must be callable")
    values = _validated_moments(mu)
    degree = min(n, b + 1)
    required = {
        term for term in range(1, 1 << n) if term.bit_count() <= degree
    }
    missing = sorted(term for term in required if term not in values)
    if missing:
        raise ValueError(
            f"decoder requires every normalized moment through degree {degree}; "
            f"missing masks {missing}"
        )
    if any(key >= 1 << n for key in values):
        raise ValueError("moment report contains a mask outside n items")

    full = (1 << n) - 1
    best: list[F | None] = [None] * (1 << n)
    # One prefix mask and one block mask per state, rather than a tuple of
    # up to b item indices per predecessor: O(2**n) stored numbers.
    parent: dict[int, tuple[int, int]] = {}
    best[0] = F(0)
    transitions = 0
    for prefix in range(1 << n):
        current = best[prefix]
        if current is None:
            raise RuntimeError(f"unreachable subset-DP state {prefix}")
        remaining = tuple(i for i in range(n) if not prefix >> i & 1)
        for s in range(1, min(b, len(remaining)) + 1):
            fee_value = _fraction("fee value", fee(prefix.bit_count(), s))
            if fee_value < 0:
                raise ValueError("fees must be nonnegative")
            for ids in combinations(remaining, s):
                block = sum(1 << i for i in ids)
                nxt = prefix | block
                candidate = current + _block_reward_validated(prefix, block, values) - fee_value
                transitions += 1
                if best[nxt] is None or candidate > best[nxt]:
                    best[nxt] = candidate
                    parent[nxt] = (prefix, block)
    if best[full] is None:
        raise RuntimeError("subset decoder failed to reach the full set")
    action: list[tuple[int, ...]] = []
    position = full
    while position:
        if position not in parent:
            raise RuntimeError(f"subset decoder has no parent for state {position}")
        previous, block = parent[position]
        action.append(tuple(i for i in range(n) if block >> i & 1))
        position = previous
    action.reverse()
    result = tuple(action)
    validate_action(result, n)
    return F(1) - best[full], result, transitions


def action_name(action: Action) -> str:
    n = _action_size(action)
    validate_action(action, n)
    return '|'.join(','.join(str(i + 1) for i in block) for block in action)


def contrast_basis(n: int, b: int) -> list[tuple[int, int]]:
    """Explicit D-dimensional features ``phi_term - phi_anchor``."""
    dimension(n, b)  # validate
    degree = min(n, b + 1)
    terms = [term for term in range(1, 1 << n) if term.bit_count() <= degree]
    if b == 1:
        return [
            (term, 1 if term.bit_count() == 1 else 3)
            for term in terms
            if term not in (1, 3)
        ]
    return [(term, 1) for term in terms if term != 1]


def contrast_features(n: int, b: int, y: int) -> list[F]:
    n = _integer("n", n, 2)
    y = _integer("relevance mask", y, 1)
    if y >= 1 << n:
        raise ValueError("the relevance vector must be a nonempty n-bit mask")
    return [
        F(int(y & term == term) - int(y & anchor == anchor), y.bit_count())
        for term, anchor in contrast_basis(n, b)
    ]


def contrast_coefficients(
    action: Action,
    reference: Action,
    n: int,
    b: int,
) -> list[F]:
    """Coefficients of U_action - U_reference in the contrast basis."""
    n = _integer("n", n, 2)
    b = _integer("block cap", b, 1)
    if b > n:
        raise ValueError("require 1 <= b <= n")
    validate_action(action, n)
    validate_action(reference, n)
    if any(len(block) > b for candidate in (action, reference) for block in candidate):
        raise ValueError("action exceeds the block cap")
    coefficients = newton_coefficients(action, n)
    reference_coefficients = newton_coefficients(reference, n)
    return [
        coefficients[term] - reference_coefficients[term]
        for term, _anchor in contrast_basis(n, b)
    ]


def law_from_full_moments(mu: Mapping[int, F], n: int) -> list[F]:
    """Invert all normalized moments; reject infeasible or incomplete reports."""
    n = _integer("n", n, 2)
    full = (1 << n) - 1
    values = _validated_moments(mu)
    if set(values) != set(range(1, full + 1)):
        raise ValueError("every nonempty normalized moment is required")
    # Upper Boolean Mobius inversion: subtract each bit-present partner.
    # Mask zero is only a working slot; it cannot feed a nonempty output.
    weights = [F(0)] + [values[mask] for mask in range(1, full + 1)]
    for i in range(n):
        bit = 1 << i
        for mask in range(full + 1):
            if not mask & bit:
                weights[mask] -= weights[mask | bit]
    probabilities = [
        support.bit_count() * weights[support]
        for support in range(1, full + 1)
    ]
    if min(probabilities) < 0 or sum(probabilities) != 1:
        raise ValueError("normalized moments do not define a probability law")
    return probabilities
