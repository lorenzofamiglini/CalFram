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
rows as drawn          adaptive=True, tie_safe=True  23 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
ties: negatives first  adaptive=True                 15 bins  ec_g 0.926  ec_dir +0.128  ece_fp 0.0524
ties: negatives first  adaptive=True, tie_safe=True  23 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
ties: positives first  adaptive=True                  3 bins  ec_g 0.928  ec_dir +0.072  ece_fp 0.0519
ties: positives first  adaptive=True, tie_safe=True  23 bins  ec_g 0.926  ec_dir +0.127  ece_fp 0.0524
                       strategy=15                    6 bins  ec_g 0.926  ec_dir +0.074  ece_fp 0.0519
                       strategy=15, tie_safe=True     7 bins  ec_g 0.927  ec_dir +0.125  ece_fp 0.0524
                       strategy='doane'             101 bins  ec_g 0.926  ec_dir +0.128  ece_fp 0.0527  (101 unique values)
```

The same 20000 rows give 3 or 15 bins and `ec_dir` +0.072 or +0.128 with the existing sweep, depending only on how
the tied rows are ordered. The tie-safe sweep gives 23 bins and +0.127 in every order, in line with the
one-bin-per-value reference (+0.128).

## The change

### A. One bin per unique value, 1.0 included (bug fix, always on)

`binning_schema` now appends a closing edge when the edges are the unique values themselves and the largest one
is >= 1, so the largest value is one of the inner edges passed to `np.digitize` and gets its own bin. This applies
to the two code paths that use the unique values as edges: the `str` strategy and the `int` strategy when there
are fewer unique values than bins ("Only N unique probability values found, using these instead of b bins").

Nothing changes when the previous code already gave one bin per value (largest value < 1): `bins`, `binids`
(including their offset when the smallest value is > 0) and `binfr` are the same arrays as before.

This is the only change of default behaviour. It is not neutral for `ec_dir`: in the under-confident version of
the example (`shift=-1.0`), `strategy='doane'` gives `ec_dir` -0.060 on `13f004b` (1.00 shares a bin with 0.99) and
-0.094 with the fix.

### B. `tie_safe=True`

New keyword, default `False`, on `binning_schema`, `calibrationcurve`, `calibrationdiagnosis` and `reliabilityplot`
(which also gains `adaptive`, it could not plot the adaptive bins before).

The data are first aggregated per distinct score (count, positives). Bins are then made of *whole* distinct scores
by `compute_tie_safe_cuts`, which follows the rule of `compute_equal_mass_bin_heights` (bins of `n // b` samples,
the last one takes the remainder) with one constraint, a bin can only end where a tied block ends:

- the distinct scores are taken in increasing order and a bin is filled until it holds at least `n // b` samples;
- a distinct score that holds `n // b` samples or more alone is a bin of its own; the unfinished bin before it,
  like the unfinished bin at the end, joins the previous bin.

Consequences: every bin holds at least `n // b` samples (no sliver bin whose noisy height would stop the sweep), a
block larger than `n // b` takes the place of several bins (so there can be fewer than `b` bins), the blocks at
0.00 and 1.00 are not mixed with their neighbours once they are heavy enough, and the bins depend only on the
counts per distinct score, not on the order of the rows.

`monotonic_sweep_calibration_tie_safe` runs the usual sweep (b = 2, 3, ... until the bin heights stop being
monotonic) over these bins and returns the last monotonic binning itself, not only its number of bins: the bins
that `calibrationdiagnosis` uses are exactly the bins that were checked. `tie_safe_bin_edges` puts each edge
half-way between the two distinct scores it separates, so no score lies on an edge and `np.digitize` gives the same
assignment left- or right-closed.

With all-distinct scores the bins are exactly those of `compute_equal_mass_bin_heights` for every `b`, hence the
sweep selects the same `b` as `monotonic_sweep_calibration` (tested, and brute-forced on random data). The final
bins still differ slightly from `adaptive=True`, because the existing code checks equal-count bins but then
builds the final bins from `np.quantile` edges, which also puts the largest score alone in an extra bin (b + 1
bins). The differences are small, see the numbers below.

| call | bins |
|---|---|
| `adaptive=True` | unchanged |
| `adaptive=True, tie_safe=True` | tie-safe sweep |
| `strategy=<int>` | unchanged |
| `strategy=<int>, tie_safe=True` | `compute_tie_safe_cuts` with b = strategy (no sweep). Fixes problem 2 |
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
continuous scores: 0.06 s against 1.2 s for the existing sweep; 100000 quantised scores: 0.03 s). When tied
blocks make consecutive `b` give the same bins, those `b` are skipped.

## Before / after

Binary task of the example, n = 20000, seed 0 ("before" = `13f004b`; per-value = `strategy='doane'` with fix A, used
as the reference for the direction):

| data | mode | bins | ec_g | ec_dir | ece_fp |
|---|---|---|---|---|---|
| over-confident (shift +1) | before, `adaptive=True` | 3 | 0.923 | +0.077 | 0.0519 |
| | before, same rows shuffled 20 times | 2-5 | | +0.056 .. +0.119 | |
| | after, `adaptive=True, tie_safe=True` (any row order) | 23 | 0.926 | +0.127 | 0.0524 |
| | before, `strategy=15` | 6 | 0.926 | +0.074 | 0.0519 |
| | after, `strategy=15, tie_safe=True` | 7 | 0.927 | +0.125 | 0.0524 |
| | per-value: before / after | 100 / 101 | 0.926 | +0.128 / +0.128 | 0.0527 |
| under-confident (shift -1) | before, `adaptive=True` | 3 | 0.924 | -0.076 | 0.0538 |
| | after, `adaptive=True, tie_safe=True` | 20 | 0.924 | -0.094 | 0.0539 |
| | before, `strategy=15`, left-closed (as in the code) / same edges right-closed | 7 | 0.923 / 0.924 | -0.077 / -0.094 | 0.0538 / 0.0539 |
| | after, `strategy=15, tie_safe=True`, left- or right-closed | 7 | 0.922 | -0.096 | 0.0539 |
| | per-value: before / after | 100 / 101 | 0.924 / 0.923 | -0.060 / -0.094 | 0.0539 / 0.0540 |
| calibrated (shift 0) | before, `adaptive=True` | 3 | 0.999 | -0.001 | 0.0009 |
| | after, `adaptive=True, tie_safe=True` | 20 | 0.993 | +0.004 | 0.0049 |
| | per-value: before / after | 100 / 101 | 0.984 | +0.008 | 0.0111 |

Across seeds 0-7 the tie-safe sweep returns 16-23 bins at n = 20000 and 9-14 at n = 3000 (existing sweep: 2-3 and
3-12), and `ec_dir` has the right sign in every run at n = 3000, 8000 and 20000 (over: +0.12 .. +0.22, under:
-0.08 .. -0.22; calibrated: at most 0.054 in absolute value at n = 3000 and 0.008 at n = 20000).

One-vs-rest, simulated multi-class tasks with probabilities rounded to 0.01 and an over-confident (sharpened) model;
class-wise means from `classwise_calibration`:

| task | mode | bins per class (median, min-max) | ec_g | ec_dir | ece_freq |
|---|---|---|---|---|---|
| 77 classes, n = 4000, 89.5% of the entries at 0.00 | before, `adaptive=True` | 1 (1-1) | 0.999 | -0.000 | 0.001 |
| | after, `adaptive=True, tie_safe=True` | 4 (2-7) | 0.990 | +0.163 | 0.008 |
| | per-value (after) | 41 (33-49) | 0.989 | +0.167 | 0.010 |
| 20 classes, n = 6000, 80.6% at 0.00 | before, `adaptive=True` | 1 (1-2) | 0.998 | +0.000 | 0.002 |
| | after, `adaptive=True, tie_safe=True` | 6 (3-10) | 0.978 | +0.167 | 0.019 |
| | per-value (after) | 83.5 (75-93) | 0.975 | +0.190 | 0.022 |

With one bin per class the existing sweep reports an over-confident model as perfectly calibrated and balanced.

Continuous scores (all distinct), n = 4000, `adaptive=True` against `adaptive=True, tie_safe=True`:

| data | before | after |
|---|---|---|
| shift +1 | 17 bins, ec_g 0.790, ec_dir +0.209, ece_fp 0.1482 | 16 bins, ec_g 0.790, ec_dir +0.210, ece_fp 0.1482 |
| shift -1 | 17 bins, ec_g 0.794, ec_dir -0.206, ece_fp 0.1460 | 16 bins, ec_g 0.794, ec_dir -0.206, ece_fp 0.1460 |
| shift 0 | 17 bins, ec_g 0.972, ec_dir -0.007, ece_fp 0.0203 | 16 bins, ec_g 0.972, ec_dir -0.007, ece_fp 0.0203 |

The sweep selects the same b (16); the 17th bin of the existing mode is the largest score alone. Over 90 runs
(n = 500, 2000, 10000; shift 0, +1, -1; 10 seeds) the selected b was identical in all of them and the largest
differences were 0.005 on `ec_g`, 0.004 on `ece_fp`, 0.006 on `ece_acc` and, on `ec_dir`, 0.033 at n = 500, 0.012 at
n = 2000 and 0.007 at n = 10000.

## Backward compatibility

- New keywords are last in each signature and default to `False`.
- Outputs compared with `13f004b` on 300 random multi-class datasets (continuous, rounded to 0.01, rounded to 0.1)
  with the default strategy, `strategy=10/15/40` and `adaptive=True`: 3303 class diagnoses bit-identical (`measures`
  and `binning_dict`); the 1032 remaining ones are per-value binnings with 1.0 among the values, where the new code
  returns `n_unique` bins instead of `n_unique - 1` (fix A).
- The existing tests (`tests/test.py`) pass unchanged.

## Things to know

- `strategy=<int>, tie_safe=True` never returns a bin with fewer than `n // b` samples. If a block holds most of the
  data and what is left is smaller than `n // b`, that remainder joins a neighbouring bin, so a small `b` can return
  1-2 bins; ask for more bins or use the sweep.
- With heavy blocks at 0.00 and 1.00 `ec_dir` stays sensitive by construction: a block at 0.00 can only lie on or
  above the diagonal and a block at 1.00 on or below it, at a distance close to 0, and each side is normalised by
  its own weight. Tie-safe bins make that contribution deterministic, they do not remove it.
- Not touched: the existing `adaptive=True` / `int` paths put the largest score alone in a last bin when it is
  below 1 (the quantile edges include the maximum), and the sweep still returns `n + 1` before the `min` with the
  number of unique values when every `b` is monotonic.

## Tests

`tests/test_tie_safe.py` (25 tests), by requirement:

- (i) quantised scores with heavy mass at 0 and 1 and a known calibration: at least 10 bins at n = 20000, the
  selected bins are monotonic, `ec_dir` > 0 for over-confidence and < 0 for under-confidence (n = 3000 and 20000,
  three seeds), close to 0 when calibrated; blocks at 0.00 and 1.00 are bins of their own; a 20-class one-vs-rest
  task through `select_probability` / `classwise_calibration`.
- (ii) invariance: identical `bins`, `binfr`, `binids` and measures under row shuffles and when the ties are sorted
  negatives-first / positives-first (with a control showing that this reordering does change the existing sweep);
  no score lies on an inner edge and `np.digitize(..., right=True)` gives the same `binids`.
- (iii) continuous scores: same b as `monotonic_sweep_calibration`, same bin heights as
  `compute_equal_mass_bin_heights` for every b, measures within 0.01 (`ec_g`, `ece`) and 0.02 (`ec_dir`) of
  `adaptive=True`.
- (iv) per-value strategy: exactly `n_unique` bins with 0.0 and 1.0 present (also for the `int` fallback), and
  unchanged output when 1.0 is not a value.
- (v) `tests/test.py` untouched.
- Also: unit tests of the binning rule (every bin >= `n // b`, never more than b bins, heavy blocks apart), the
  sweep gives the same cuts as a sweep that checks every b, the diagnosis is computed on the tie-safe bins, defaults
  unchanged, degenerate inputs, `reliabilityplot` keywords.

```
PYTHONPATH=. python -m pytest tests/test.py tests/test_tie_safe.py -q
36 passed
```
