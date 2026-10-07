# Calibration-complete structured ranking

Exact rational validation for **Calibration Dimension of Average-Precision Ranking with Oracle Block Abstention**. An action is an ordered partition of a fixed item set. After the binary relevance outcome is revealed, an ideal resolver orders each selected block relevance-first. The target is average precision plus a declared, outcome-independent action fee. No retrieval corpus, trained model, external resolver, human study, or deployment result is included or implied.

## Reproduce

Use Python 3.10 or later on a POSIX system with the standard-library `resource` module. No third-party Python package, network access, solver, random seed, account, or data download is required. From this repository root:

```sh
python -m unittest discover -s tests -v
python verify.py
```

The location-independent suite contains **44 tests**, including six dedicated full-moment inversion tests. It covers mathematical identities, malformed and inexact-input rejection, relabeling equivariance, monotonicity under block merging, contrast reconstruction, the active-face rank certificate, exact round trips, single-pass report validation, transition-local moment access, two-mask predecessor storage, bibliography and proof-dependency ledgers, deterministic-output comparison, and deliberate audit-failure cases. The verifier reruns the tests and regenerates the scientific CSV/JSON outputs under `results/` with one worker. Do not use `python -O`: the runner rejects optimized execution because scientific assertions must remain active. Timing and memory are observations, not reproduction targets.

For bounded or resumable checks:

```sh
python verify.py --part exhaustive
python verify.py --part extended
python verify.py --part fiber
python verify.py --part parity
python verify.py --part decoding
```

A failed assertion, unit test, bibliography audit, or proof-dependency audit returns a nonzero exit status. Append `--output /path/to/empty-output` to regenerate results without touching the retained `results/` directory. The destination must be genuinely empty. Inputs resolve relative to the source files rather than the shell working directory. The five parts together cover the same scientific computations as `--part all`; resource observations differ across separate processes. A complete clean comparison is:

```sh
fresh=$(mktemp -d)
python verify.py --output "$fresh"
python compare_results.py results "$fresh"
```

The comparator uses an explicit manifest of **25 regenerable scientific files** and checks each byte-for-byte. It fails on any missing, changed, or unexpected scientific file. `results/intake_and_pilot.json` is a retained observation of the original host and early pilot; the verifier does not regenerate or copy it, and the comparator excludes it. Run-specific `resources*.json` files and `reproduction.json` are likewise metadata rather than scientific equality targets.

## What is proved and what is checked

`theory.pdf` is the accompanying article and contains the general proofs. The runner does **not** mechanize those proofs or certify novelty. It validates bounded identities and instances using `fractions.Fraction`, without floating-point tolerances or random sampling.

The exhaustive suite covers all 14 `(n,b)` cases with `2 <= n <= 5` and every block cap: 71,428 action–outcome entries and 205,074 complete-refinement AP evaluations. It checks the oracle definition against alternative evaluations, the Boolean polynomial against inversion, the explicit contrast basis and exact rank, the reference Bayes tie, and coefficient bounds. Nested caps repeat actions; these are not independent samples.

The extended geometry suite covers all six caps at `n=6`, with menus up to **4,683 actions**. For every action it checks the predicted degree, linear constraints, zero constant coefficient, all-relevant numerator, and numerator coefficient bound: 22,995 coefficient-bound checks across the six caps. Modular elimination over a fixed prime selects candidate independent rows; exact rational elimination then certifies a 299-row basis across the six cases. The modular step is only row selection, not the final rank certificate. Twelve definition-level action/outcome samples per cap independently compare the implementation with direct relevance-aware refinement.

The three-item fiber certificate covers all 13 actions, exact feasible endpoints, affine dominance over the entire parameter rectangle, and two full-support witness laws. The parity construction checks normalized moments through `b=8`; complete action menus are enumerated only through `b=5` (`n=6`). Decoder checks contain 168 exhaustive optimum comparisons and six larger returned-action risk/transition checks through `n=8`; the latter are not brute-force optimum comparisons.

The three negative controls are quadratic truncation of the oracle polynomial, outcome-independent random completion, and an incorrectly position-independent reference pair price. Their failures/successes distinguish outcome-aware higher-order interactions from bookkeeping changes.

## Bibliography audit

`bibliography.bib` is a byte-for-byte mirror of the article bibliography. `reference_audit.csv` records, for every entry, canonical title, author sequence, year, authoritative URL, source tier, publication status, and verification depth. `reference_audit.py` parses balanced-brace BibTeX, enforces at least 55 entries, matches title/author/year exactly, cross-checks URLs against `external_resources.csv`, rejects duplicate persistent identifiers, and distinguishes formal publications from official preprints. `proof_dependency_audit.csv` separately records the imported lower-bound theorem and the standard convex/linear-programming facts used in the proof, together with their assumptions, local discharge, and residual boundaries.

The current audit covers **65 cited scholarly works**: 61 primary publications and four official preprints. Twenty-six closest works or theorem dependencies were checked at full-text or claim-section depth; the remaining relevant works were checked against official publication metadata and the cited abstract/section. This is a documented source check, not a claim that every auxiliary proof in every cited work was independently rederived.

## File roles

`ranking.py` contains validated exact definitions, ordered-partition enumeration, harmonic/Newton coefficients, rational rank, explicit contrast features, reference prices, the subset decoder, and normalized-moment inversion. Public scientific inputs reject binary floating-point and Boolean values rather than silently converting them; use integers, `Fraction`, exact decimal/rational strings, or another exact value accepted by `Fraction`. `verify.py` orchestrates the five finite campaigns, and `compare_results.py` detects deterministic evidence drift. `tests/` contains the 44 tests. `fixtures/fiber.json` is an exact analytic fixture, not sampled or fitted data.

`claim_evidence_ledger.csv` maps every material manuscript claim to theorem/lemma locations, proof or checker, exact output, maturity, scope, and latest recheck. `external_resources.csv` records the 65 scholarly sources and two official venue/style resources, including attribution and integration boundaries. `proof_dependency_audit.csv` prevents an imported theorem from being mistaken for a locally proved result. No external implementation or dataset is imported.

`results/matrices/` stores definition-based utility tables for the 14 small cases. `dimensions.csv`, `extended_geometry.csv`, and `coefficient_bounds.csv` retain dimension and coefficient evidence. `fiber_actions.csv`, `fiber_dominance.json`, `fiber_certificate.json`, `fiber_curve.csv`, and `fee_deficiency.csv` retain the exact three-item analysis. `parity.csv` and `parity_menus.csv` separate moment checks from full-menu checks. `decoding.csv` labels exhaustive optimum comparisons separately from returned-action checks. These are the 25 files in the regenerable scientific manifest. `intake_and_pilot.json` preserves a one-time resource/pilot observation and its accounting boundary; it is not a regeneration target. `resources*.json` records particular runs, while `reproduction.json` records clean-copy reproduction and its limits.

## Conventions and boundaries

Code items are indexed `0,...,n-1`; the article uses `1,...,n`. An outcome is an integer mask `1,...,2**n-1`, with bit `i` marking item `i` relevant. In the article's printed coordinate order, `(100)` has mask 1. The all-zero outcome is excluded. A block is an unordered tuple of item indices; an action is a tuple of blocks in precedence order. Scientific rational values are serialized as reduced fraction strings; decimal columns are display conversions only.

The executable decoder accepts position-and-size additive fees, a subclass of the article's set-dependent additive-fee theorem. Its state count is exponential in `n`; no polynomial-time inference or bit-complexity claim is made. Convex dimension is exact for the stated reference schedule and for a common surrogate over all admissible fees when the link may depend on the fee; it is not claimed for every fixed uniform pair fee. Moment-fiber counterexamples concern insufficient optimal mean reports, not every conceivable convex surrogate. Random completion is not oracle resolution.

The decoder validates and copies the moment report once. Each transition then probes only its required masks; it does not traverse the full report. Predecessors hold two integer subset masks per state, and item tuples are constructed only when backtracking the returned action. These conventions implement the article's storage bound in numbers, not bits.

Full normalized-moment inversion uses the appendix's alternating-superset formula via an exact upper Boolean Möbius transform. The loop visits `n * 2**n` masks and performs exactly `n * 2**(n-1)` Fraction subtractions, followed by linear output construction. This is O(n 2^n) arithmetic operations and O(2^n) stored numbers, not a bit-complexity bound or a measured speedup. The zero-mask slot is internal only: every nonempty moment is still required and only nonempty-outcome probabilities are returned, with unchanged exact nonnegativity and unit-total checks. The six new tests use independent definition/alternating-sum oracles on all point masses through n=6, four mixtures per n=2…8, a 64-report two-item grid and directed invalid inputs. They are discovered by the existing verifier and scientific CI without a workflow change; no hosted run is claimed.

The retained `results/resources*.json` files describe historical host runs and their test counts, not a measurement of the current source. Their timings are not updated by a prose or code revision. Scientific result files remain the deterministic comparison targets.

## Automated exact checks

For the flat standalone artifact repository, `.github/workflows/scientific-checks.yml` runs on pushes to `main`, pull requests, and manual dispatch. It retains the material-integrity gate, executes the full verifier, and compares all 25 scientific files. The Ubuntu 24.04/Python 3.12 scientific step has a 480-second whole-run wall deadline, a 450-second per-process CPU limit, a 2 GiB virtual-memory limit, and bounded file size. Raw logs and generated outputs are uploaded even when a gate fails. Workflow configuration alone is not evidence that a hosted run has succeeded.

## License and attribution

The MIT license applies to the original Python code, tests, analytic fixtures, and generated numeric tables. It does not license the accompanying article. No scholarly PDF, external baseline implementation, dataset, or publisher style file is included in this standalone repository. External resources were not modified to improve a comparison.
