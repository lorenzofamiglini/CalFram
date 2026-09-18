# Tie-safe binning

Branch `tie-safe-binning`, on top of `13f004b`. One file changed (`calfram/calibration_framework.py`), one test
module added (`tests/test_tie_safe.py`). Default behaviour is unchanged, with one exception that is a bug fix
(section A).

## The problem

Some models do not emit continuous scores. The case that triggered this patch emits probabilities on a 0.01 grid,
with large tied blocks at 0.00 and 1.00 (in a one-vs-rest view of a 77-class task most of each column is 0.00).
The binning code assumes continuous scores in three places:

1. **`adaptive=True`**. `compute_equal_mass_bin_heights` sorts the rows by score and cuts bins of equal *count*, so a
   tied block is split across several bins. Those bins hold the same score and their heights differ only by
   noise (and by the order of the rows, since the sort is stable), so `is_monotonic` fails almost at once and the
   sweep stops at 1-3 bins. The result also changes when the rows are shuffled.
2. **`strategy=<int>`**. The edges are quantiles of the scores, so with ties an edge *is* a tied value. The whole
   block then goes to the bin on one side only because `np.digitize` is left-closed. `ec_dir` normalises each side
   of the diagonal by its own weight, so where a block that holds 20-40% of the data goes moves `ec_dir` a lot.
3. **`strategy=<str>`** (one bin per unique value, the default `'doane'` of `calibrationdiagnosis`). When 1.0 is one
   of the values `bin_edges[-1] == 1.0`, no closing edge is appended, and `bin_edges[1:-1]` drops 1.0 from the inner
   edges given to `np.digitize`: the two largest values share a bin (`n_unique - 1` bins).

### Reproducible example

```python
import warnings
import numpy as np
from calfram.calibration_framework import CalibrationFramework

warnings.simplefilter("ignore")
cf = CalibrationFramework()


def quantised_scores(n, shift, seed):
    """Scores on a 0.01 grid, about 45% of them at 0.00 and 20% at 1.00. The label is drawn from
    sigmoid(logit), the score is sigmoid(logit + shift): shift > 0 is over-confident, shift < 0 under-confident."""
    rng = np.random.default_rng(seed)
    component = rng.choice(3, size=n, p=[0.45, 0.20, 0.35])
    logit = np.where(component == 0, rng.normal(-8, 1, n), np.where(component == 1, rng.normal(8, 1, n), rng.normal(0, 2, n)))
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.round(1 / (1 + np.exp(-(logit + shift))), 2), y


def diagnose(score, y, **kwargs):
    proba = np.column_stack([1 - score, score])
    classes_scores = cf.select_probability(y, proba, (score >= 0.5).astype(int))
    measures, bins = cf.calibrationdiagnosis(classes_scores, **kwargs)
    return "%3d bins  ec_g %.3f  ec_dir %+.3f  ece_fp %.4f" % (len(bins['1']['binfr']), measures['1']['ec_g'], measures['1']['ec_dir'], measures['1']['ece_fp'])


score, y = quantised_scores(20000, shift=1.0, seed=0)          # over-confident
neg_first, pos_first = np.lexsort((y, score)), np.lexsort((-y, score))  # same data, ties reordered
for name, order in (("rows as drawn", slice(None)), ("ties: negatives first", neg_first), ("ties: positives first", pos_first)):
    print("%-22s adaptive=True               " % name, diagnose(score[order], y[order], adaptive=True))
    print("%-22s adaptive=True, tie_safe=True" % name, diagnose(score[order], y[order], adaptive=True, tie_safe=True))
print("%-22s strategy=15                 " % "", diagnose(score, y, strategy=15))
print("%-22s strategy=15, tie_safe=True  " % "", diagnose(score, y, strategy=15, tie_safe=True))
print("%-22s strategy='doane'            " % "", diagnose(score, y, strategy='doane'), " (%d unique values)" % len(np.unique(score)))
```

Output on this branch (the lines without `tie_safe` are identical on `13f004b`, except the last one, which gives
100 bins there):

```
rows as drawn          adaptive=True                  3 bins  ec_g 0.923  ec_dir +0.077  ece_fp 0.0519
rows as drawn          adaptive=True, tie_safe=True  21 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
ties: negatives first  adaptive=True                 15 bins  ec_g 0.926  ec_dir +0.128  ece_fp 0.0524
ties: negatives first  adaptive=True, tie_safe=True  21 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
ties: positives first  adaptive=True                  3 bins  ec_g 0.928  ec_dir +0.072  ece_fp 0.0519
ties: positives first  adaptive=True, tie_safe=True  21 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
                       strategy=15                    6 bins  ec_g 0.926  ec_dir +0.074  ece_fp 0.0519
                       strategy=15, tie_safe=True     8 bins  ec_g 0.926  ec_dir +0.128  ece_fp 0.0524
                       strategy='doane'             101 bins  ec_g 0.926  ec_dir +0.128  ece_fp 0.0527  (101 unique values)
```

The same 20000 rows give 3 or 15 bins and `ec_dir` +0.072 or +0.128 with the existing sweep, depending only on how
the tied rows are ordered. The tie-safe sweep gives 21 bins and +0.127 in every order, in line with the
one-bin-per-value reference (+0.128).

## The change

### A. One bin per unique value, 1.0 included (bug fix, always on)

`binning_schema` now appends a closing edge when the edges are the unique values themselves and the largest one
is >= 1, so the largest value is one of the inner edges passed to `np.digitize` and gets its own bin. This applies
to the `str` strategy always, and to the `int` strategy when there are fewer unique values than bins ("Only N
unique probability values found, using these instead of b bins") **only with `tie_safe=True`**. Without
`tie_safe` that `int` fallback keeps the bins of `13f004b` bit for bit (the two largest values still share a bin
when 1.0 is one of them), so benchmark numbers produced with an `int` strategy do not move on upgrade. The quirk
is the same bug; whether to fix it by default too is a maintainer decision (it is a one-line change,
`per_value = tie_safe` -> `per_value = True`). Example on `13f004b`: p in {0, 0.3, 0.6, 1.0}, n = 400,
`strategy=15` gives 3 bins, `ec_dir` +0.142; with `tie_safe=True` 4 bins, `ec_dir` +0.130.

Nothing changes when the previous code already gave one bin per value (largest value < 1): `bins`, `binids`
(including their offset when the smallest value is > 0) and `binfr` are the same arrays as before.

This (the `str` strategy) is the only change of default behaviour. It is not neutral for `ec_dir`: in the under-confident version of
the example (`shift=-1.0`), `strategy='doane'` gives `ec_dir` -0.060 on `13f004b` (1.00 shares a bin with 0.99) and
-0.094 with the fix.

### B. `tie_safe=True`

New keyword, default `False`, on `binning_schema`, `calibrationcurve`, `calibrationdiagnosis` and `reliabilityplot`
(which also gains `adaptive`, it could not plot the adaptive bins before).

The data are first aggregated per distinct score (count, positives). Bins are then made of *whole* distinct scores
by `compute_tie_safe_cuts`, which keeps the targets of `compute_equal_mass_bin_heights` (a cut after every `n // b`
samples, the last bin takes the remainder) with one constraint, a bin can only end where a tied block ends:

1. each target `k * (n // b)`, k = 1..b-1, moves to the tied-block boundary nearest to it (the upper one on an
   exact tie); targets that land on the same boundary give one cut. The targets are global, so the rounding error
   does not pile up from bin to bin;
2. a distinct score that holds `n // b` samples or more alone (a heavy block, typically 0.00 or 1.00) gets a cut on
   both sides;
3. a bin with fewer than `n // (2b)` samples (a sliver) joins its smaller neighbour, and while there are more than
   `b` bins the smallest bin does too.

Consequences: a block is never split; each cut is within one tied block of its equal-mass target, so a grid
without heavy blocks gives `b` bins of about `n / b` (also for `b = 2`); no bin is a sliver whose noisy height would
stop the sweep; a block larger than `n // b` takes the place of several bins (so there can be fewer than `b` bins);
a heavy block is a bin of its own except when a sliver next to it has no other neighbour (the rows before the
first heavy block, or between two heavy blocks), in which case the sliver joins it; and the bins depend only on the
counts per distinct score, not on the order of the rows. `b < 1` is treated as `b = 1` (one bin), as the default
path does.

`monotonic_sweep_calibration_tie_safe` runs the usual sweep (b = 2, 3, ... until the bin heights stop being
monotonic) over these bins and returns the last monotonic binning itself, not only its number of bins: the bins
that `calibrationdiagnosis` uses are exactly the bins that were checked. `tie_safe_bin_edges` puts each edge
half-way between the two distinct scores it separates, so no score lies on an edge and `np.digitize` gives the same
assignment left- or right-closed.

The monotonicity check is `is_monotonic_tie_safe(counts, positives, robust)`. With all-distinct scores it is
`is_monotonic` exactly (`robust=False`). With ties (`robust=True`) one relaxation applies, only to two adjacent
bins whose sizes differ by a factor of 2 or more: a decrease counts only if it survives changing one label in the
smaller bin. Reason: next to a large block at 0.00 (one-vs-rest columns, 90-95% of the rows) that block almost
always holds a positive (1/2713), while the small bin after it (138 rows, about one positive expected) holds none
about a third of the time; `1/2713 > 0/138` stopped the sweep at 2-3 bins in 53 of 100 runs, and flipping that one
label gave 9 bins. Bins of similar size are still checked exactly. So the selected bins are monotonic up to one
label between bins of very different size.

With all-distinct scores the bins are exactly those of `compute_equal_mass_bin_heights` for every `b`, hence the
sweep selects the same `b` as `monotonic_sweep_calibration` (tested, and brute-forced on random data). The final
bins still differ slightly from `adaptive=True`, because the existing code checks equal-count bins but then
builds the final bins from `np.quantile` edges, which also puts the largest score alone in an extra bin (b + 1
bins). The differences are small, see the numbers below.

| call | bins |
|---|---|
| `adaptive=True` | unchanged |
| `adaptive=True, tie_safe=True` | tie-safe sweep |
| `strategy=<int>` | unchanged (including the fewer-values-than-bins fallback) |
| `strategy=<int>, tie_safe=True` | `compute_tie_safe_cuts` with b = strategy (no sweep). Fixes problem 2; fallback with fewer values than bins: one bin per value, with fix A |
| `strategy=<str>`, with or without `tie_safe` | one bin per unique value (already tie-safe), with fix A |
| fewer than 2 unique values | unchanged (artificial bins and warning) |

How the bins reach the metrics: `calibrationdiagnosis` -> `calibrationcurve` -> `binning_schema`. Everything
downstream reads `bins_dict['binids']`: `calibrationcurve` computes the points `x`, `y` of the ECI measures from it,
`binfr` gives their weights, and `compute_eces` receives the same `binids` (`bins` is only used as a `minlength`).
`tie_safe` is passed along that chain, and a test recomputes `x`, `y`, the weights and `ece_fp` from the returned
`binids`.

Why a keyword rather than `adaptive="tie_safe"`: the existing code tests `if adaptive:`, so on any older
installation `adaptive="tie_safe"` would silently run the old sweep, while an unknown keyword raises a `TypeError`.
A keyword also extends naturally to the `int` strategy.

Cost: the sweep works on per-score counts with numpy, it does not re-sort the rows for every `b` (100000
continuous scores: 0.11 s against 1.2 s for the existing sweep; 100000 quantised scores: 0.03 s against 1.0 s).

## Before / after

Binary task of the example, n = 20000, seed 0 ("before" = `13f004b`; per-value = `strategy='doane'` with fix A, used
as the reference for the direction):

| data | mode | bins | ec_g | ec_dir | ece_fp |
|---|---|---|---|---|---|
| over-confident (shift +1) | before, `adaptive=True` | 3 | 0.923 | +0.077 | 0.0519 |
| | before, same rows shuffled 20 times | 2-5 | | +0.056 .. +0.119 | |
| | after, `adaptive=True, tie_safe=True` (any row order) | 21 | 0.926 | +0.127 | 0.0524 |
| | before, `strategy=15` | 6 | 0.926 | +0.074 | 0.0519 |
| | after, `strategy=15, tie_safe=True` | 8 | 0.926 | +0.128 | 0.0524 |
| | per-value: before / after | 100 / 101 | 0.926 | +0.128 / +0.128 | 0.0527 |
| under-confident (shift -1) | before, `adaptive=True` | 3 | 0.924 | -0.076 | 0.0538 |
| | after, `adaptive=True, tie_safe=True` | 19 | 0.923 | -0.094 | 0.0539 |
| | before, `strategy=15`, left-closed (as in the code) / same edges right-closed | 7 | 0.923 / 0.924 | -0.077 / -0.094 | 0.0538 / 0.0539 |
| | after, `strategy=15, tie_safe=True`, left- or right-closed | 7 | 0.924 | -0.094 | 0.0539 |
| | per-value: before / after | 100 / 101 | 0.924 / 0.923 | -0.060 / -0.094 | 0.0539 / 0.0540 |
| calibrated (shift 0) | before, `adaptive=True` | 3 | 0.999 | -0.001 | 0.0009 |
| | after, `adaptive=True, tie_safe=True` | 20 | 0.994 | +0.003 | 0.0043 |
| | per-value: before / after | 100 / 101 | 0.984 | +0.008 | 0.0111 |

Across seeds 0-7 the tie-safe sweep returns 17-24 bins at n = 20000 and 8-14 at n = 3000 (existing sweep: 2-3 and
3-13), and `ec_dir` has the right sign in every run at n = 3000, 8000 and 20000 (over: +0.13 .. +0.22, under:
-0.08 .. -0.21; calibrated: at most 0.043 in absolute value at n = 3000 and 0.009 at n = 20000).

`strategy=<int>, tie_safe=True` with a small b on the same data (shift +1): `strategy=3` gives 3 bins, `ec_dir`
+0.147 (before: 2 bins, +0.060, the tied block at 1.00 falls on a quantile edge); `strategy=2` gives 2 bins in both.
On a uniform 0.01 grid (n = 20000, no heavy block) `strategy` = 2, 3, 4, 5, 10, 15 gives exactly that many bins of
about `n / b`.

Dominant block at 0.00 (binary, 95% at 0.00, 1% at 1.00, shift +1, 100 seeds): runs that end with <= 3 bins, 53/100
before the robust check and 0/100 with it at n = 3000 (median 7 bins), 0/100 at n = 20000 (median 12). With 85% at
0.00: 0/100 at both sizes (median 10 and 14).

One-vs-rest, simulated multi-class tasks with probabilities rounded to 0.01 and an over-confident (sharpened) model;
class-wise means from `classwise_calibration`:

| task | mode | bins per class (median, min-max) | ec_g | ec_dir | ece_freq |
|---|---|---|---|---|---|
| 77 classes, n = 4000, 89.5% of the entries at 0.00 | before, `adaptive=True` | 1 (1-1) | 0.999 | -0.000 | 0.001 |
| | after, `adaptive=True, tie_safe=True` | 4 (3-6); 8/77 classes <= 3 | 0.990 | +0.151 | 0.008 |
| | per-value (after) | 41 (32-53) | 0.989 | +0.164 | 0.010 |
| 77 classes, n = 20000, 89.4% at 0.00 | before, `adaptive=True` | 1 (1-1) | 0.999 | -0.000 | 0.001 |
| | after, `adaptive=True, tie_safe=True` | 5 (3-9); 9/77 classes <= 3 | 0.990 | +0.193 | 0.008 |
| | per-value (after) | 81 (73-87) | 0.990 | +0.167 | 0.009 |
| 20 classes, n = 6000, 80.6% at 0.00 | before, `adaptive=True` | 1 (1-2) | 0.998 | +0.000 | 0.002 |
| | after, `adaptive=True, tie_safe=True` | 6 (3-9) | 0.978 | +0.189 | 0.018 |
| | per-value (after) | 83.5 (75-93) | 0.975 | +0.190 | 0.022 |

With one bin per class the existing sweep reports an over-confident model as perfectly calibrated and balanced.
The 77-class generator is `multiclass(n, 77, power=2)` (true softmax with a latent class, reported probabilities
`p**2` renormalised and rounded to 0.01); the 20-class one is the one in `test_multiclass_with_rare_classes`. Before
the robust check, 23 of the 77 classes at n = 4000 ended with <= 3 bins; the remaining ones are stops on a real
decrease between small bins, the same fragility the existing sweep has on continuous scores.

Continuous scores (all distinct), n = 4000, `adaptive=True` against `adaptive=True, tie_safe=True`:

| data | before | after |
|---|---|---|
| shift +1 | 17 bins, ec_g 0.790, ec_dir +0.209, ece_fp 0.1482 | 16 bins, ec_g 0.790, ec_dir +0.210, ece_fp 0.1482 |
| shift -1 | 17 bins, ec_g 0.794, ec_dir -0.206, ece_fp 0.1460 | 16 bins, ec_g 0.794, ec_dir -0.206, ece_fp 0.1460 |
| shift 0 | 17 bins, ec_g 0.972, ec_dir -0.007, ece_fp 0.0203 | 16 bins, ec_g 0.972, ec_dir -0.007, ece_fp 0.0203 |

The sweep selects the same b (16); the 17th bin of the existing mode is the largest score alone. Over 90 runs
(n = 500, 2000, 10000; shift 0, +1, -1; 10 seeds; logits N(0, 2)) the selected b was identical in all of them and the
largest differences were 0.008 on `ec_g`, 0.006 on `ece_fp` and, on `ec_dir`, 0.075 at n = 500, 0.013 at n = 2000 and
0.005 at n = 10000 (at n = 500 the extra bin of the existing mode holds a single score, which weighs on `ec_dir`).
With all-distinct scores the tie-safe bins are unaffected by the review changes (bit-identical `binids` in these
90 runs against the first version of this branch).

## Backward compatibility

- New keywords are last in each signature and default to `False`.
- 12000 random `binning_schema` calls without `tie_safe` (n from 2 to 900; continuous, rounded to 0.1/0.01/0.001,
  mass at exactly 0 and 1; `int` and `str` strategies; `adaptive` on and off) compared with `13f004b`: 10236
  bit-identical; the 1764 others are all `str`-strategy binnings with 1.0 among the values (fix A). The `int`
  fallback with fewer values than bins: 0 differences. Warnings, exceptions and default `reliabilityplot` output are
  unchanged.
- The existing tests (`tests/test.py`) pass unchanged.

## Things to know

- `strategy=<int>, tie_safe=True` never returns a bin with fewer than `n // (2b)` samples. If a block holds most of
  the data and what is left is smaller than that, the remainder joins a neighbouring bin, so a small `b` can return
  1-2 bins when one block dominates; ask for more bins or use the sweep. Without a dominant block it returns `b`
  bins.
- The robust check lets the sweep accept a one-label decrease between bins of very different size, so
  `measures['y']` of the tie-safe sweep is not always non-decreasing. Between bins of similar size it is.
- With heavy blocks at 0.00 and 1.00 `ec_dir` stays sensitive by construction: a block at 0.00 can only lie on or
  above the diagonal and a block at 1.00 on or below it, at a distance close to 0, and each side is normalised by
  its own weight. Tie-safe bins make that contribution deterministic, they do not remove it.
- Not touched: the existing `adaptive=True` / `int` paths put the largest score alone in a last bin when it is
  below 1 (the quantile edges include the maximum), and the sweep still returns `n + 1` before the `min` with the
  number of unique values when every `b` is monotonic.

## Tests

`tests/test_tie_safe.py` (30 tests), by requirement:

- (i) quantised scores with heavy mass at 0 and 1 and a known calibration: at least 10 bins at n = 20000, the
  selected bins pass `is_monotonic_tie_safe`, `ec_dir` > 0 for over-confidence and < 0 for under-confidence (n = 3000
  and 20000, three seeds), close to 0 when calibrated; blocks at 0.00 and 1.00 are bins of their own; a dominant
  block at 0.00 (95%, n = 3000, 40 seeds) gives <= 3 bins in at most 2 runs and flipping its one positive changes
  the bin count by at most 2; a 20-class one-vs-rest task through `select_probability` / `classwise_calibration`.
- (ii) invariance: identical `bins`, `binfr`, `binids` and measures under row shuffles and when the ties are sorted
  negatives-first / positives-first (with a control showing that this reordering does change the existing sweep);
  no score lies on an inner edge and `np.digitize(..., right=True)` gives the same `binids`.
- (iii) continuous scores: same b as `monotonic_sweep_calibration`, same bin heights as
  `compute_equal_mass_bin_heights` for every b, measures within 0.01 (`ec_g`, `ece`) and 0.02 (`ec_dir`) of
  `adaptive=True`; continuous scores rounded to 4 decimals (ties of a few rows) keep the 16 bins of `b = 16`.
- (iv) per-value strategy: exactly `n_unique` bins with 0.0 and 1.0 present, unchanged output when 1.0 is not a
  value; the `int` fallback keeps the `13f004b` bins by default and gives one bin per value with `tie_safe`.
- (v) `tests/test.py` untouched.
- Also: unit tests of the binning rule (no sliver, never more than b bins, heavy blocks apart, `b` bins on a grid
  without heavy block, `b <= 0`), of `is_monotonic_tie_safe` (exact for similar sizes, one-label tolerance
  otherwise), `strategy=0, tie_safe=True` through `calibrationdiagnosis` (one bin, as without `tie_safe`), the
  diagnosis is computed on the tie-safe bins, defaults unchanged, degenerate inputs, `reliabilityplot` keywords.

```
PYTHONPATH=. python -m pytest tests/test.py tests/test_tie_safe.py -q
41 passed
```

## Revision after review

- `compute_tie_safe_cuts` anchored each bin to its own start and filled it to at least `n // b`, so on tied data the
  overshoot piled up and the last bin was merged: `b - 1` bins on any grid, 1 bin for `strategy=2`, a double-size
  top bin, and a few small ties were enough to lose a bin against all-distinct scores. Cuts now go to the block
  boundary nearest to the global targets (rule above).
- The heavy-block rule was skipped once the cut budget was used; it is now applied independently of the targets.
- `strategy=0` (or negative) with `tie_safe=True` divided by zero; `b` is now clamped to at least 1.
- The sweep stopped at 2-3 bins next to a dominant 0.00 block on a one-label violation; robust check above.
- The `int` fallback with fewer values than bins no longer changes by default (section A).
- The b-skipping shortcut in the sweep was removed: with global targets, b values that share `n // b` no longer
  give the same bins.
