"""Follow-up fixes (see "Follow-up fixes" in PATCH_NOTES.md)."""
import unittest
import warnings

import numpy as np

from calfram.calibration_framework import HISTOGRAM_RULES, CalibrationFramework
from tests.test_mass_balance import blocks, diagnose
from tests.test_tie_safe import continuous_scores, quantised_scores


def triangle_distance(cf, x, y):
    """The normalised distances as calibrationdiagnosis computed them before: triangle heights (Heron's formula)
    of each point and of the farthest point in its column, relative to the previous point on the diagonal."""
    new_pts = cf.end_points(x, y)
    tilde = cf.add_tilde(new_pts)
    max_pts = new_pts.copy()
    for pt in range(1, len(new_pts)):
        max_pts[pt] = [new_pts[pt][0], 1] if new_pts[pt][0] <= 0.5 else [new_pts[pt][0], 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = cf.h_triangle_safe(new_pts, tilde) / cf.h_triangle_safe(max_pts, tilde)
    return np.nan_to_num(ratio, nan=0.0, posinf=1.0, neginf=0.0)


class TestNormalisedDistance(unittest.TestCase):

    def setUp(self):
        self.cf = CalibrationFramework()

    def test_values(self):
        x = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        y = np.array([0.3, 0.2, 1.0, 0.4, 0.0])
        np.testing.assert_allclose(self.cf.normalised_distance(x, y), [0.3, 0.0, 1.0, 0.5, 1.0])

    def test_matches_triangle_heights(self):
        rng = np.random.default_rng(0)
        for t in range(2000):
            x = np.sort(np.r_[rng.random(int(rng.integers(1, 30))), [0.5, 1.0][t % 2]])
            y = rng.random(len(x))
            on_line = rng.random(len(x)) < 0.2
            y[on_line] = x[on_line]
            # Heron's formula with the 1e-10 clamp is only approximate when a bin has (almost) the x of the previous one
            apart = np.diff(np.r_[0.0, x]) > 1e-6
            np.testing.assert_allclose(self.cf.normalised_distance(x, y)[apart], triangle_distance(self.cf, x, y)[apart],
                                       rtol=0, atol=1e-9)


class TestWeightCheck(unittest.TestCase):

    def test_mismatch_is_reported(self):
        cf = CalibrationFramework()
        original = cf.binning_schema

        def one_weight_too_many(*args, **kwargs):
            bins = original(*args, **kwargs)
            bins['binfr'] = np.r_[bins['binfr'], 0.0]
            return bins

        cf.binning_schema = one_weight_too_many
        score, y = blocks([(0.2, 100, 10), (0.7, 100, 80)])
        proba = np.column_stack([1 - score, score])
        classes_scores = cf.select_probability(y, proba, (score >= 0.5).astype(int))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            measures, _ = cf.calibrationdiagnosis(classes_scores, strategy=10)
        self.assertTrue(any("2 calibration points" in str(w.message) for w in caught))
        self.assertTrue(np.isnan(measures['1']['ec_g']))

    def test_aligned_weights(self):
        score, y = blocks([(0.2, 100, 10), (0.7, 100, 80)])
        m, bins = diagnose(CalibrationFramework(), score, y, strategy=10)
        self.assertEqual(len(bins['binfr']), len(m['where']))


class TestOnTheDiagonal(unittest.TestCase):

    def setUp(self):
        self.cf = CalibrationFramework()

    def test_underbelow_line(self):
        pts = np.array([[0.4, 0.4 + 1e-15], [0.4, 0.4 - 1e-15], [0.4, 0.41], [0.4, 0.39], [0.25, 0.25]])
        self.assertEqual(self.cf.underbelow_line(pts), ['lie', 'lie', 'left', 'right', 'lie'])

    def test_calibrated_bin_at_0_4(self):
        score, y = blocks([(0.4, 100, 40)])
        self.assertNotEqual(score.mean(), 0.4)   # the rounding this is about
        for balance in ('sides', 'mass'):
            m, _ = diagnose(self.cf, score, y, strategy=10, balance=balance)
            self.assertEqual(list(m['where']), ['lie'])
            self.assertAlmostEqual(m['ec_g'], 1.0, places=12)
            self.assertTrue(np.isnan(m['ec_underconf']) and np.isnan(m['ec_overconf']))
            if balance == 'mass':
                self.assertEqual(m['ec_dir'], 0.0)
                self.assertTrue(np.isnan(m['ec_dir_sides']))
            else:
                self.assertTrue(np.isnan(m['ec_dir']))   # the 'sides' balance has no side to average

    def test_calibrated_bin_among_others(self):
        # before, the bin at 0.4 was 'left' or 'right' by 1e-16 (here 'left': x = 0.4000000000000001), and counted as a bin of that side
        over = [(0.2, 100, 10), (0.7, 100, 50)]
        score, y = blocks(over + [(0.4, 100, 40)])
        m, _ = diagnose(self.cf, score, y, strategy=10, balance='mass')
        self.assertEqual(list(m['where']), ['right', 'lie', 'right'])
        d = np.array([0.1 / 0.8, 0.2 / 0.7])
        self.assertAlmostEqual(m['ec_dir_sides'], np.mean(d), places=12)
        self.assertAlmostEqual(m['ec_dir'], np.sum(d) / 3, places=12)
        self.assertAlmostEqual(m['ec_dir'], 1 - m['ec_g'], places=12)


class TestStrStrategies(unittest.TestCase):

    def setUp(self):
        self.cf = CalibrationFramework()

    def binning(self, score, y, method, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.cf.binning_schema(np.column_stack([1 - score, score]), y, method=method, **kwargs)

    def test_histogram_rules(self):
        for seed, n in ((0, 300), (1, 2000), (2, 20000)):
            score, y = continuous_scores(n, 0.5, seed)
            for rule in HISTOGRAM_RULES:
                counts, edges = np.histogram(score, bins=rule)
                bins = self.binning(score, y, rule)
                # the same bins as np.histogram: same number of edges, same assignment, empty bins dropped
                np.testing.assert_array_equal(np.bincount(bins['binids'], minlength=len(edges) - 1), counts)
                self.assertLessEqual(bins['binids'].max(), len(edges) - 2)
                np.testing.assert_array_equal(bins['bins'], edges[:-1][counts > 0])
                np.testing.assert_allclose(bins['binfr'], counts[counts > 0] / n)
                # tie_safe does not apply to fixed-width edges (a tied value always goes to one bin)
                bins_t = self.binning(score, y, rule, tie_safe=True)
                np.testing.assert_array_equal(bins_t['binids'], bins['binids'])

    def test_default_is_doane(self):
        score, y = continuous_scores(4000, 1.0, 0)
        counts, _ = np.histogram(score, bins='doane')
        m, bins = diagnose(self.cf, score, y)
        self.assertEqual(len(bins['binfr']), np.sum(counts > 0))
        self.assertEqual(len(m['x']), np.sum(counts > 0))
        self.assertLess(len(bins['binfr']), 30)    # before: one bin per row (4000)

    def test_unique(self):
        for gen, n in ((continuous_scores, 500), (quantised_scores, 5000)):
            score, y = gen(n, 0.5, 0)
            values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
            bins = self.binning(score, y, 'unique')
            # one bin per unique value, 1.0 included, in order
            self.assertEqual(len(bins['binfr']), len(values))
            np.testing.assert_allclose(bins['binfr'], counts / n)
            order = np.unique(bins['binids'], return_inverse=True)[1]
            np.testing.assert_array_equal(order.ravel(), inverse.ravel())
            if gen is quantised_scores:
                self.assertIn(1.0, values)

    def test_unknown_strategy(self):
        score, y = continuous_scores(200, 0.0, 0)
        with self.assertRaises(ValueError):
            self.binning(score, y, 'doan')
        with self.assertRaises(ValueError):
            diagnose(self.cf, score, y, strategy='per_value')


if __name__ == '__main__':
    unittest.main()
