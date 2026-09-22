import unittest
import numpy as np
from calfram.calibration_framework import CalibrationFramework


class TestPooledSweepAndIdeal(unittest.TestCase):
    """strategy='pooled_sweep' (tie-safe binning by pooling) and ideal_calibration."""

    def setUp(self):
        self.cf = CalibrationFramework()
        rng = np.random.default_rng(0)
        # scores on a 0.01 grid, so that many items share one value
        self.p = np.round(rng.beta(2, 2, 3000), 2)
        self.y_cal = (rng.random(3000) < self.p).astype(int)          # calibrated by construction
        self.y_sharp = (rng.random(3000) < 0.5 + 0.5 * (self.p - 0.5)).astype(int)  # flatter truth: over-confident scores
        self.y_prob = np.column_stack([1 - self.p, self.p])
        self.y_pred = (self.p >= 0.5).astype(int)

    def diagnose(self, y, **kwargs):
        scores = self.cf.select_probability(y, self.y_prob, self.y_pred)
        return self.cf.calibrationdiagnosis({'1': scores['1']}, **kwargs)

    def test_pooled_sweep_never_splits_a_tied_block(self):
        binids, bins, binfr = self.cf.pooled_sweep_bins(self.p, self.y_sharp)
        for v in np.unique(self.p):
            self.assertEqual(len(np.unique(binids[self.p == v])), 1)
        self.assertAlmostEqual(float(binfr.sum()), 1.0)
        self.assertEqual(len(bins), binids.max() + 1)
        self.assertTrue(np.all(np.diff(bins) > 0))

    def test_pooled_sweep_is_the_isotonic_regression(self):
        from sklearn.isotonic import IsotonicRegression
        binids, _, _ = self.cf.pooled_sweep_bins(self.p, self.y_sharp)
        freq = np.bincount(binids, weights=self.y_sharp) / np.bincount(binids)
        self.assertTrue(np.all(np.diff(freq) >= 0))
        iso = IsotonicRegression(increasing=True).fit(self.p, self.y_sharp).predict(self.p)
        np.testing.assert_allclose(freq[binids], iso, atol=1e-9)

    def test_pooled_sweep_as_a_strategy(self):
        measures, binning = self.diagnose(self.y_sharp, strategy='pooled_sweep')
        self.assertTrue(0.0 <= measures['1']['ec_g'] <= 1.0)
        self.assertTrue(np.isfinite(measures['1']['ece_fp']))
        self.assertEqual(len(binning['1']['bins']), len(binning['1']['binfr']))
        for v in np.unique(self.p):
            self.assertEqual(len(np.unique(binning['1']['binids'][self.p == v])), 1)
        # already tie-safe: the keyword changes nothing
        same, _ = self.diagnose(self.y_sharp, strategy='pooled_sweep', tie_safe=True)
        self.assertEqual(same['1']['ec_g'], measures['1']['ec_g'])
        # the order of the rows does not matter
        order = np.argsort(self.p, kind='stable')[::-1]
        scores = self.cf.select_probability(self.y_sharp[order], self.y_prob[order], self.y_pred[order])
        shuffled, _ = self.cf.calibrationdiagnosis({'1': scores['1']}, strategy='pooled_sweep')
        self.assertAlmostEqual(shuffled['1']['ec_g'], measures['1']['ec_g'], places=12)

    def test_mass_balance_on_pooled_bins(self):
        r = self.diagnose(self.y_sharp, strategy='pooled_sweep', balance='mass')[0]['1']
        self.assertAlmostEqual(r['ec_dir'], r['ec_overconf_mass'] - r['ec_underconf_mass'])
        self.assertGreater(r['ec_dir'], 0.0)  # over-confident scores: predictions above outcomes
        self.assertLessEqual(abs(r['ec_dir']), 1.0 - r['ec_g'] + 1e-9)
        # a mirrored problem flips the sign
        scores_m = self.cf.select_probability(1 - self.y_sharp, self.y_prob[:, ::-1], 1 - self.y_pred)
        r_m = self.cf.calibrationdiagnosis({'1': scores_m['1']}, strategy='pooled_sweep', balance='mass')[0]['1']
        self.assertAlmostEqual(r_m['ec_dir'], -r['ec_dir'])

    def test_select_probability_with_an_absent_class(self):
        y = self.y_cal.copy()
        y[:] = 1  # only one label present
        scores = self.cf.select_probability(y, self.y_prob, self.y_pred, n_classes=2)
        self.assertIn('0', scores)
        self.assertIn('1', scores)
        self.assertEqual(scores['0']['y_one_hot_nclass'].shape, (3000, 2))
        with self.assertRaises(ValueError):
            self.cf.select_probability(np.array([0, 2]), np.ones((2, 2)) / 2, np.array([0, 0]), n_classes=2)

    def test_ideal_is_below_one_and_a_calibrated_model_is_not_rejected(self):
        r = self.cf.ideal_calibration(self.y_cal, self.y_prob, self.y_pred, strategy='pooled_sweep', n_sim=200, seed=1)['1']
        self.assertLess(r['ec_g']['ideal'], 1.0)
        self.assertGreater(r['ec_g']['ideal'], 0.9)
        self.assertGreater(r['ec_g']['p_value'], 0.05)
        self.assertGreater(r['ece_fp']['p_value'], 0.05)
        self.assertGreater(r['ece_fp']['ideal'], 0.0)
        self.assertEqual(r['ec_g']['direction'], 'less')
        self.assertEqual(r['ec_dir']['direction'], 'two-sided')

    def test_over_confident_scores_are_rejected(self):
        r = self.cf.ideal_calibration(self.y_sharp, self.y_prob, self.y_pred, strategy='pooled_sweep', n_sim=200, seed=1)['1']
        self.assertLess(r['ec_g']['p_value'], 0.01)
        self.assertLess(r['ece_fp']['p_value'], 0.01)
        self.assertLess(r['ec_g']['observed'], r['ec_g']['ci'][0])

    def test_ideal_with_mass_balance_and_the_tie_safe_sweep(self):
        r = self.cf.ideal_calibration(self.y_sharp, self.y_prob, self.y_pred, adaptive=True, tie_safe=True,
                                      balance='mass', n_sim=50, seed=2)['1']
        self.assertIn('ec_overconf_mass', r)
        self.assertEqual(r['ec_overconf_mass']['direction'], 'greater')
        self.assertLess(r['ec_overconf_mass']['p_value'], 0.05)
        self.assertGreater(r['ec_dir']['observed'], r['ec_dir']['ideal'])

    def test_ideal_is_reproducible_and_restores_the_global_generator(self):
        np.random.seed(7)
        before = np.random.get_state()
        a = self.cf.ideal_calibration(self.y_cal, self.y_prob, self.y_pred, strategy=15, n_sim=20, seed=3)
        after = np.random.get_state()
        b = self.cf.ideal_calibration(self.y_cal, self.y_prob, self.y_pred, strategy=15, n_sim=20, seed=3)
        self.assertEqual(a['1']['ec_g']['ideal'], b['1']['ec_g']['ideal'])
        self.assertEqual(a['1']['ec_g']['p_value'], b['1']['ec_g']['p_value'])
        self.assertTrue(np.array_equal(before[1], after[1]))

    def test_ideal_on_a_multiclass_problem(self):
        rng = np.random.default_rng(2)
        P = rng.dirichlet(np.ones(3), 600).round(2)
        P = P / P.sum(axis=1, keepdims=True)
        y = np.array([rng.choice(3, p=row) for row in P])
        r = self.cf.ideal_calibration(y, P, P.argmax(axis=1), strategy=10, n_sim=30, seed=0, measures=('ec_g', 'ece_fp'))
        self.assertEqual(set(r), {'0', '1', '2'})
        for c in r:
            self.assertTrue(np.isfinite(r[c]['ec_g']['ideal']))


if __name__ == '__main__':
    unittest.main()
