import unittest
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np

from calfram.calibration_framework import CalibrationFramework


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def quantised_scores(n, shift, seed, p_zero=0.45, p_one=0.20):
    """
    Binary task whose scores lie on a 0.01 grid, with large tied blocks at 0.00 and 1.00.

    The label is drawn from sigmoid(logit) and the reported score is sigmoid(logit + shift) rounded to two
    decimals, so the calibration is known: shift = 0 is calibrated (up to the rounding), shift > 0 reports
    more than the true probability (over-confidence, ec_dir > 0), shift < 0 less (under-confidence, ec_dir < 0).
    """
    rng = np.random.default_rng(seed)
    component = rng.choice(3, size=n, p=[p_zero, p_one, 1 - p_zero - p_one])
    logit = np.where(component == 0, rng.normal(-8, 1, n), np.where(component == 1, rng.normal(8, 1, n), rng.normal(0, 2, n)))
    y = (rng.random(n) < sigmoid(logit)).astype(int)
    return np.round(sigmoid(logit + shift), 2), y


def continuous_scores(n, shift, seed):
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 2, n)
    y = (rng.random(n) < sigmoid(logit)).astype(int)
    return sigmoid(logit + shift), y


def diagnose(cf, score, y, **kwargs):
    """Measures and bins of the positive class of a binary task."""
    proba = np.column_stack([1 - score, score])
    classes_scores = cf.select_probability(y, proba, (score >= 0.5).astype(int))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measures, binning_dict = cf.calibrationdiagnosis(classes_scores, **kwargs)
    # calibrationdiagnosis turns an exception into a warning and NaN measures: make it a failure here
    errors = [str(w.message) for w in caught if str(w.message).startswith("Error")]
    assert not errors, errors
    return measures['1'], binning_dict['1']


def bins_of_values(score, binids):
    """The set of bin ids that each distinct score is sent to."""
    return {value: set(binids[score == value].tolist()) for value in np.unique(score)}


class TestPerValueBinning(unittest.TestCase):
    """str strategy (and the int strategy when there are fewer unique values than bins): one bin per unique value."""

    def setUp(self):
        self.cf = CalibrationFramework()

    def test_one_bin_per_value_with_zero_and_one(self):
        rng = np.random.default_rng(0)
        values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        score = rng.choice(values, size=500, p=[0.4, 0.1, 0.1, 0.1, 0.3])
        y = (rng.random(500) < score).astype(int)
        proba = np.column_stack([1 - score, score])

        prob_true, prob_pred, bins_dict = self.cf.calibrationcurve(y, proba, strategy='doane')

        self.assertEqual(len(bins_dict['binfr']), len(values))
        self.assertEqual(len(bins_dict['bins']), len(values))
        np.testing.assert_allclose(prob_pred, values)  # Each bin holds a single value, 1.0 and 0.0 included
        np.testing.assert_allclose(bins_dict['binfr'], [np.mean(score == v) for v in values])
        np.testing.assert_allclose(prob_true, [np.mean(y[score == v]) for v in values])
        for ids in bins_of_values(score, bins_dict['binids']).values():
            self.assertEqual(len(ids), 1)

    def test_one_bin_per_value_on_a_grid(self):
        score, y = quantised_scores(5000, 0.0, 0)
        self.assertEqual(score.min(), 0.0)
        self.assertEqual(score.max(), 1.0)
        measures, bins_dict = diagnose(self.cf, score, y, strategy='doane')
        self.assertEqual(len(bins_dict['binfr']), len(np.unique(score)))
        np.testing.assert_allclose(measures['x'], np.unique(score))

    def test_int_strategy_with_fewer_values_than_bins(self):
        rng = np.random.default_rng(1)
        score = rng.choice([0.0, 0.3, 0.6, 1.0], size=400)
        y = (rng.random(400) < score).astype(int)
        proba = np.column_stack([1 - score, score])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bins_dict = self.cf.binning_schema(proba, y, method=15)
            bins_dict_t = self.cf.binning_schema(proba, y, method=15, tie_safe=True)
        # Default: the bins of the previous versions, where 0.6 and 1.0 share a bin
        np.testing.assert_array_equal(bins_dict['binids'], np.digitize(score, [0.3, 0.6]))
        self.assertEqual(len(bins_dict['binfr']), 3)
        # tie_safe: one bin per value
        self.assertEqual(len(bins_dict_t['binfr']), 4)
        for ids in bins_of_values(score, bins_dict_t['binids']).values():
            self.assertEqual(len(ids), 1)

    def test_unchanged_when_one_is_not_a_value(self):
        # The previous rule already gave one bin per value when the largest value is below 1: same output
        rng = np.random.default_rng(2)
        for low in (0.0, 0.05):
            score = np.round(rng.uniform(low, 0.99, 800), 2)
            y = (rng.random(800) < score).astype(int)
            proba = np.column_stack([1 - score, score])

            edges = np.array(sorted(set(score)))
            if edges[0] > 0:
                edges = np.concatenate([[0.0], edges])
            edges = np.concatenate([edges, [1.0]])
            expected_binids = np.digitize(score, edges[1:-1])
            expected_counts = np.bincount(expected_binids, minlength=len(edges) - 1)

            bins_dict = self.cf.binning_schema(proba, y, method='doane')
            np.testing.assert_array_equal(bins_dict['binids'], expected_binids)
            np.testing.assert_array_equal(bins_dict['bins'], edges[:-1][expected_counts > 0])
            np.testing.assert_array_equal(bins_dict['binfr'], (expected_counts / len(score))[expected_counts > 0])
            self.assertEqual(len(bins_dict['binfr']), len(np.unique(score)))


class TestTieSafeCuts(unittest.TestCase):
    """compute_tie_safe_cuts: equal-mass bins made of whole distinct scores."""

    def setUp(self):
        self.cf = CalibrationFramework()

    @staticmethod
    def cum(counts):
        return np.concatenate([[0], np.cumsum(counts)])

    def test_heavy_blocks_are_bins_of_their_own(self):
        # 100 samples, 5 bins of 20: the block of 50 is one bin, 8 + 7 + 5 = 20 the next one, the 5 samples
        # left before the block of 25 join it and the block of 25 is a bin of its own
        counts = [50, 8, 7, 5, 5, 25]
        cuts = self.cf.compute_tie_safe_cuts(self.cum(counts), 5)
        np.testing.assert_array_equal(cuts, [1, 5])  # {50}, {8, 7, 5, 5}, {25}
        # With 2 bins of 50 the block of 50 is one bin and all the rest the other
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum(counts), 2), [1])
        # No previous bin to join: the 3 samples before the block of 60 stay with it; 17 >= 20 / 2 is a bin
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum([3, 60, 20, 17]), 5), [2, 3])  # {3, 60}, {20}, {17}
        # A heavy block at the end is a bin of its own even when the targets are used up: the single sample
        # before it joins the previous bin
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum([4, 1, 4]), 2), [2])  # {4, 1}, {4}
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum([2, 1, 1, 4, 4, 3, 3, 2, 4]), 5), [3, 4, 6, 8])

    def test_small_last_run_joins_the_previous_bin(self):
        counts = [10, 10, 10, 2]
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum(counts), 3), [1, 2])  # {10}, {10}, {10, 2}
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum(counts), 4), [1, 2])  # 2 samples cannot make a bin of 8
        counts = [90, 4, 3, 3]
        np.testing.assert_array_equal(self.cf.compute_tie_safe_cuts(self.cum(counts), 2), [])  # 10 samples cannot make a bin of 50

    def test_no_sliver_and_at_most_b_bins(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            n_unique = int(rng.integers(2, 40))
            counts = np.where(rng.random(n_unique) < 0.2, rng.integers(50, 500, n_unique), rng.integers(1, 10, n_unique))
            cum_counts = self.cum(counts)
            n = int(cum_counts[-1])
            for b in range(1, min(n, 120) + 1):
                cuts = self.cf.compute_tie_safe_cuts(cum_counts, b)
                sizes = np.diff(cum_counts[np.concatenate([[0], cuts, [n_unique]])])
                self.assertEqual(sizes.sum(), n)
                self.assertLessEqual(len(sizes), b)
                self.assertTrue(np.all(np.diff(cuts) > 0) and np.all((cuts > 0) & (cuts < n_unique)))
                if len(sizes) > 1:
                    self.assertGreaterEqual(2 * sizes.min(), n // max(b, 1))  # No bin below half the nominal size

    def test_b_bins_on_a_grid_without_heavy_block(self):
        # Ties everywhere but no block near n // b: every b gives b bins close to n // b (no overshoot that
        # piles up and leaves b - 1 bins, or a single bin for b = 2)
        rng = np.random.default_rng(8)
        score = np.round(rng.random(20000), 2)
        counts = np.unique(score, return_counts=True)[1]
        self.assertLess(counts.max(), 0.015 * len(score))
        cum_counts = self.cum(counts)
        for b in (2, 3, 4, 5, 10, 15, 20):
            sizes = np.diff(cum_counts[np.concatenate([[0], self.cf.compute_tie_safe_cuts(cum_counts, b), [len(counts)]])])
            self.assertEqual(len(sizes), b)
            self.assertLess(np.max(np.abs(sizes - len(score) / b)), counts.max())  # Each cut within one block of its target

    def test_a_few_small_ties_keep_the_bins(self):
        # Continuous scores rounded to 4 decimals (ties of a few rows): the same b bins as the exact scores
        for seed in range(4):
            score, _ = continuous_scores(4000, 0.0, seed)
            for s in (score, np.round(score, 4)):
                counts = np.unique(s, return_counts=True)[1]
                cuts = self.cf.compute_tie_safe_cuts(self.cum(counts), 16)
                sizes = np.diff(self.cum(counts)[np.concatenate([[0], cuts, [len(counts)]])])
                self.assertEqual(len(sizes), 16)
                self.assertLessEqual(np.max(np.abs(sizes - 250)), counts.max())

    def test_zero_or_negative_b_gives_one_bin(self):
        for b in (0, -3):
            self.assertEqual(len(self.cf.compute_tie_safe_cuts(self.cum([5, 5, 5]), b)), 0)

    def test_all_distinct_scores_give_the_equal_mass_bins(self):
        rng = np.random.default_rng(4)
        for n in (7, 50, 333):
            score = rng.random(n)
            y = (rng.random(n) < score).astype(int)
            data = list(zip(score.tolist(), y.tolist()))
            cum_counts = np.arange(n + 1)
            cum_positives = np.concatenate([[0.0], np.cumsum(y[np.argsort(score)])])
            for b in range(1, n + 1):
                idx = np.concatenate([[0], self.cf.compute_tie_safe_cuts(cum_counts, b), [n]])
                heights = np.diff(cum_positives[idx]) / np.diff(cum_counts[idx])
                self.assertEqual(heights.tolist(), self.cf.compute_equal_mass_bin_heights(data, b))

    def test_edges_lie_between_the_values(self):
        values = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
        edges = self.cf.tie_safe_bin_edges(values, np.array([1, 4]))
        np.testing.assert_allclose(edges, [0.0, 0.005, 0.995, 1.0])
        np.testing.assert_array_equal(self.cf.tie_safe_bin_edges(values, np.array([], dtype=np.int64)), [0.0, 1.0])


class TestTieSafeAdaptive(unittest.TestCase):
    """adaptive=True, tie_safe=True: monotonic sweep over bins made of whole distinct scores."""

    def setUp(self):
        self.cf = CalibrationFramework()

    # (i) quantised scores with heavy mass at 0 and 1 and a known calibration
    def test_sensible_number_of_bins_on_quantised_scores(self):
        for shift in (0.0, 1.0, -1.0):
            for seed in (0, 1, 2):
                score, y = quantised_scores(20000, shift, seed)
                self.assertGreater(np.mean(score == 0.0), 0.35)
                self.assertGreater(np.mean(score == 1.0), 0.15)
                measures, bins_dict = diagnose(self.cf, score, y, adaptive=True, tie_safe=True)
                n_bins = len(bins_dict['binfr'])
                self.assertGreaterEqual(n_bins, 10)
                self.assertLessEqual(n_bins, len(np.unique(score)))
                # The selected bins are monotonic, up to one label between bins of very different size
                counts = np.bincount(bins_dict['binids'])
                positives = np.bincount(bins_dict['binids'], weights=y)
                self.assertTrue(self.cf.is_monotonic_tie_safe(counts, positives, robust=True))

    def test_ec_dir_has_the_sign_of_the_known_miscalibration(self):
        for n in (3000, 20000):
            for seed in (0, 1, 2):
                over, _ = diagnose(self.cf, *quantised_scores(n, 1.0, seed), adaptive=True, tie_safe=True)
                under, _ = diagnose(self.cf, *quantised_scores(n, -1.0, seed), adaptive=True, tie_safe=True)
                calibrated, _ = diagnose(self.cf, *quantised_scores(n, 0.0, seed), adaptive=True, tie_safe=True)
                self.assertGreater(over['ec_dir'], 0.08)
                self.assertLess(under['ec_dir'], -0.04)
                if n == 20000:
                    self.assertLess(abs(calibrated['ec_dir']), 0.05)
                self.assertGreater(calibrated['ec_g'], over['ec_g'])
                self.assertGreater(calibrated['ec_g'], under['ec_g'])
                self.assertLess(calibrated['ece_fp'], 0.015)
                self.assertGreater(over['ece_fp'], 0.03)
                self.assertGreater(under['ece_fp'], 0.03)

    def test_tied_blocks_are_never_split(self):
        score, y = quantised_scores(20000, 1.0, 0)
        for kwargs in (dict(adaptive=True), dict(strategy=15), dict(strategy=40)):
            _, bins_dict = diagnose(self.cf, score, y, tie_safe=True, **kwargs)
            for ids in bins_of_values(score, bins_dict['binids']).values():
                self.assertEqual(len(ids), 1)
            # No empty bin, and the bins follow the order of the scores
            self.assertEqual(len(np.unique(bins_dict['binids'])), len(bins_dict['binfr']))
            self.assertTrue(np.all(np.diff(bins_dict['binids'][np.argsort(score, kind='stable')]) >= 0))

    def test_heavy_blocks_at_zero_and_one_are_bins_of_their_own(self):
        for shift in (0.0, 1.0, -1.0):
            score, y = quantised_scores(20000, shift, 0)
            for kwargs in (dict(adaptive=True), dict(strategy=15)):
                measures, bins_dict = diagnose(self.cf, score, y, tie_safe=True, **kwargs)
                self.assertEqual(measures['x'][0], 0.0)
                self.assertEqual(measures['x'][-1], 1.0)
                self.assertEqual(bins_dict['binfr'][0], np.mean(score == 0.0))
                self.assertEqual(bins_dict['binfr'][-1], np.mean(score == 1.0))

    # (ii) invariance
    def test_invariant_to_row_order(self):
        score, y = quantised_scores(20000, 1.0, 0)
        for kwargs in (dict(adaptive=True), dict(strategy=15)):
            measures, bins_dict = diagnose(self.cf, score, y, tie_safe=True, **kwargs)
            orders = [np.random.default_rng(seed).permutation(len(score)) for seed in range(3)]
            orders += [np.lexsort((y, score)), np.lexsort((-y, score))]  # Ties reordered: negatives first, positives first
            for order in orders:
                measures_o, bins_dict_o = diagnose(self.cf, score[order], y[order], tie_safe=True, **kwargs)
                np.testing.assert_array_equal(bins_dict_o['bins'], bins_dict['bins'])
                np.testing.assert_array_equal(bins_dict_o['binfr'], bins_dict['binfr'])
                np.testing.assert_array_equal(bins_dict_o['binids'], bins_dict['binids'][order])
                for key in ('ec_g', 'ec_dir', 'ec_underconf', 'ec_overconf', 'ece_fp', 'ece_acc', 'brier_loss', 'x', 'y'):
                    np.testing.assert_allclose(measures_o[key], measures[key], rtol=0, atol=1e-12)

    def test_reordering_ties_is_a_real_perturbation(self):
        # Control for the test above: the same reordering changes the bins of the sweep that is not tie-safe
        score, y = quantised_scores(20000, 1.0, 0)
        n_bins = []
        for order in (np.lexsort((y, score)), np.lexsort((-y, score))):
            _, bins_dict = diagnose(self.cf, score[order], y[order], adaptive=True)
            n_bins.append(len(bins_dict['binfr']))
        self.assertNotEqual(n_bins[0], n_bins[1])

    def test_independent_of_left_or_right_closed_bins(self):
        score, y = quantised_scores(20000, -1.0, 1)
        for kwargs in (dict(adaptive=True), dict(strategy=15)):
            _, bins_dict = diagnose(self.cf, score, y, tie_safe=True, **kwargs)
            self.assertFalse(np.any(np.isin(bins_dict['bins'][1:], score)))  # No score lies on an inner edge
            inner_edges = bins_dict['bins'][1:]  # No empty bin: 'bins' holds all the left edges
            np.testing.assert_array_equal(np.digitize(score, inner_edges, right=False), bins_dict['binids'])
            np.testing.assert_array_equal(np.digitize(score, inner_edges, right=True), bins_dict['binids'])

    # (iii) continuous scores: same sweep as the existing adaptive mode
    def test_agrees_with_adaptive_on_continuous_scores(self):
        for shift in (0.0, 1.0, -1.0):
            for seed in (0, 1, 2):
                score, y = continuous_scores(4000, shift, seed)
                self.assertEqual(len(np.unique(score)), len(score))
                legacy, legacy_bins = diagnose(self.cf, score, y, adaptive=True)
                tie_safe, tie_safe_bins = diagnose(self.cf, score, y, adaptive=True, tie_safe=True)

                # Same number of bins selected by the sweep (the existing mode then adds a bin for the largest score alone)
                b = self.cf.monotonic_sweep_calibration(list(zip(score.tolist(), y.tolist())), len(score))
                self.assertEqual(len(tie_safe_bins['binfr']), b)
                self.assertIn(len(legacy_bins['binfr']), (b, b + 1))

                self.assertAlmostEqual(tie_safe['ec_g'], legacy['ec_g'], delta=0.01)
                self.assertAlmostEqual(tie_safe['ece_fp'], legacy['ece_fp'], delta=0.01)
                self.assertAlmostEqual(tie_safe['ece_acc'], legacy['ece_acc'], delta=0.01)
                self.assertAlmostEqual(tie_safe['ec_dir'], legacy['ec_dir'], delta=0.02)
                self.assertEqual(tie_safe['brier_loss'], legacy['brier_loss'])
                if shift != 0:
                    self.assertEqual(np.sign(tie_safe['ec_dir']), np.sign(shift))

    def test_sweep_selects_the_same_b_as_the_existing_sweep(self):
        rng = np.random.default_rng(5)
        for n in (40, 300, 1500):
            for _ in range(5):
                score = rng.random(n)
                y = (rng.random(n) < score).astype(int)
                b = self.cf.monotonic_sweep_calibration(list(zip(score.tolist(), y.tolist())), n)
                cum_positives = np.concatenate([[0.0], np.cumsum(y[np.argsort(score)])])
                cuts = self.cf.monotonic_sweep_calibration_tie_safe(np.arange(n + 1), cum_positives)
                self.assertEqual(len(cuts) + 1, b)

    def test_robust_monotonicity_check(self):
        counts = np.array([2700, 140])
        # 1/2700 > 0/140 is a decrease, but one positive more in the small bin removes it
        self.assertFalse(self.cf.is_monotonic_tie_safe(counts, np.array([1.0, 0.0])))
        self.assertTrue(self.cf.is_monotonic_tie_safe(counts, np.array([1.0, 0.0]), robust=True))
        # A decrease that survives one label still counts
        self.assertFalse(self.cf.is_monotonic_tie_safe(counts, np.array([500.0, 10.0]), robust=True))
        self.assertFalse(self.cf.is_monotonic_tie_safe(counts[::-1], np.array([10.0, 1.0]), robust=True))
        # Bins of similar size: exactly is_monotonic
        rng = np.random.default_rng(9)
        for _ in range(500):
            counts = rng.integers(50, 99, 6)
            positives = rng.binomial(counts, 0.3).astype(float)
            expected = self.cf.is_monotonic((positives / counts).tolist())
            self.assertEqual(self.cf.is_monotonic_tie_safe(counts, positives), expected)
            self.assertEqual(self.cf.is_monotonic_tie_safe(counts, positives, robust=True), expected)

    def test_dominant_block_at_zero(self):
        # One-vs-rest-like column: 95% of the rows at 0.00 with a handful of positives. A single label in the
        # small bin next to the block must not stop the sweep at 2-3 bins
        n_bins = []
        for seed in range(40):
            score, y = quantised_scores(3000, 1.0, seed, p_zero=0.95, p_one=0.01)
            proba = np.column_stack([1 - score, score])
            n_bins.append(len(self.cf.binning_schema(proba, y, adaptive=True, tie_safe=True)['binfr']))
        self.assertLessEqual(sum(b <= 3 for b in n_bins), 2)
        self.assertGreaterEqual(np.median(n_bins), 5)

        # Flipping the one positive of the block at 0.00 barely changes the bins
        score, y = quantised_scores(3000, 1.0, 0, p_zero=0.95, p_one=0.01)
        self.assertEqual(y[score == 0.0].sum(), 1)
        flipped = y.copy()
        flipped[(score == 0.0) & (y == 1)] = 0
        proba = np.column_stack([1 - score, score])
        n_before = len(self.cf.binning_schema(proba, y, adaptive=True, tie_safe=True)['binfr'])
        n_after = len(self.cf.binning_schema(proba, flipped, adaptive=True, tie_safe=True)['binfr'])
        self.assertGreaterEqual(min(n_before, n_after), 5)
        self.assertLessEqual(abs(n_before - n_after), 2)

    # calibrationdiagnosis really uses the tie-safe bins
    def test_diagnosis_is_computed_on_the_tie_safe_bins(self):
        score, y = quantised_scores(20000, 1.0, 2)
        measures, bins_dict = diagnose(self.cf, score, y, adaptive=True, tie_safe=True)
        proba = np.column_stack([1 - score, score])
        expected = self.cf.binning_schema(proba, y, adaptive=True, tie_safe=True)
        np.testing.assert_array_equal(bins_dict['binids'], expected['binids'])

        binids = bins_dict['binids']
        counts = np.bincount(binids)
        x = np.bincount(binids, weights=score) / counts
        freq = np.bincount(binids, weights=y) / counts
        np.testing.assert_allclose(measures['x'], x)
        np.testing.assert_allclose(measures['y'], freq)
        np.testing.assert_allclose(measures['relative-freq'], counts / len(score))
        self.assertAlmostEqual(measures['ece_fp'], np.sum(counts / len(score) * np.abs(freq - x)), places=12)
        self.assertEqual(len(measures['ec_l_all']), len(counts))

        _, legacy_bins = diagnose(self.cf, score, y, adaptive=True)
        self.assertGreater(len(counts), len(legacy_bins['binfr']))

    def test_multiclass_with_rare_classes(self):
        # One-vs-rest of a 20-class task: most of each column is a tied block at 0.00
        rng = np.random.default_rng(6)
        n, n_classes = 6000, 20
        latent = rng.integers(0, n_classes, n)
        logits = rng.normal(0, 1, (n, n_classes))
        logits[np.arange(n), latent] += rng.normal(5, 2.5, n)
        true_proba = np.exp(logits - logits.max(axis=1, keepdims=True))
        true_proba /= true_proba.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(n_classes, p=p) for p in true_proba])
        sharpened = true_proba ** 2  # Over-confident report, rounded to the grid
        proba = np.round(sharpened / sharpened.sum(axis=1, keepdims=True), 2)
        self.assertGreater(np.mean(proba == 0.0), 0.7)

        classes_scores = self.cf.select_probability(y, proba, proba.argmax(axis=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            measures, binning_dict = self.cf.calibrationdiagnosis(classes_scores, adaptive=True, tie_safe=True)
            reference, _ = self.cf.calibrationdiagnosis(classes_scores, strategy='doane')  # One bin per value
        n_bins = [len(binning_dict[key]['binfr']) for key in binning_dict]
        self.assertGreaterEqual(np.median(n_bins), 4)
        self.assertGreaterEqual(min(n_bins), 3)
        class_wise = self.cf.classwise_calibration(measures)
        self.assertGreater(class_wise['ec_dir'], 0.05)
        self.assertAlmostEqual(class_wise['ec_dir'], self.cf.classwise_calibration(reference)['ec_dir'], delta=0.06)

    # API
    def test_default_keeps_the_existing_behaviour(self):
        score, y = quantised_scores(3000, 1.0, 3)
        for kwargs in (dict(adaptive=True), dict(strategy=15), dict(strategy='doane')):
            measures, bins_dict = diagnose(self.cf, score, y, **kwargs)
            measures_f, bins_dict_f = diagnose(self.cf, score, y, tie_safe=False, **kwargs)
            np.testing.assert_array_equal(bins_dict['binids'], bins_dict_f['binids'])
            self.assertEqual(measures['ec_dir'], measures_f['ec_dir'])
        # One bin per value is tie-safe already: the keyword changes nothing
        measures, bins_dict = diagnose(self.cf, score, y, strategy='doane')
        measures_t, bins_dict_t = diagnose(self.cf, score, y, strategy='doane', tie_safe=True)
        np.testing.assert_array_equal(bins_dict['binids'], bins_dict_t['binids'])
        self.assertEqual(measures['ec_dir'], measures_t['ec_dir'])

    def test_int_strategy_tie_safe(self):
        score, y = quantised_scores(20000, 1.0, 4)
        for b in (5, 15, 40):
            _, bins_dict = diagnose(self.cf, score, y, strategy=b, tie_safe=True)
            counts = np.bincount(bins_dict['binids'])
            self.assertLessEqual(len(counts), b)
            self.assertGreaterEqual(2 * counts.min(), len(score) // b)
        self.assertEqual(len(diagnose(self.cf, score, y, strategy=2, tie_safe=True)[1]['binfr']), 2)
        # On all-distinct scores: b bins of n // b samples
        score, y = continuous_scores(3000, 0.0, 0)
        _, bins_dict = diagnose(self.cf, score, y, strategy=15, tie_safe=True)
        np.testing.assert_array_equal(np.bincount(bins_dict['binids']), [200] * 15)

    def test_strategy_zero(self):
        # Invalid but accepted by the default path (one bin): tie_safe must not fail on it
        score, y = quantised_scores(2000, 1.0, 6)
        measures, bins_dict = diagnose(self.cf, score, y, strategy=0)
        measures_t, bins_dict_t = diagnose(self.cf, score, y, strategy=0, tie_safe=True)
        self.assertEqual(len(bins_dict['binfr']), 1)
        self.assertEqual(len(bins_dict_t['binfr']), 1)
        self.assertAlmostEqual(measures_t['ec_g'], measures['ec_g'], places=12)

    def test_few_unique_values(self):
        y = np.array([0, 1] * 50)
        for score in (np.full(100, 0.3), np.where(y == 1, 1.0, 0.0), np.where(y == 1, 0.8, 0.2)):
            proba = np.column_stack([1 - score, score])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bins_dict = self.cf.binning_schema(proba, y, adaptive=True, tie_safe=True)
            self.assertEqual(len(bins_dict['binfr']), len(np.unique(score)))
            self.assertEqual(len(bins_dict['binids']), 100)

    def test_reliabilityplot_accepts_the_keywords(self):
        import matplotlib.pyplot as plt
        score, y = quantised_scores(2000, 1.0, 5)
        proba = np.column_stack([1 - score, score])
        classes_scores = self.cf.select_probability(y, proba, (score >= 0.5).astype(int))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cf.reliabilityplot(classes_scores, split=False, adaptive=True, tie_safe=True)
        n_points = [len(line.get_xdata()) for line in plt.gca().get_lines()[:2]]
        _, bins_dict = diagnose(self.cf, score, y, adaptive=True, tie_safe=True)
        self.assertEqual(n_points[1], len(bins_dict['binfr']))
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
