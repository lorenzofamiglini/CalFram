import unittest
import numpy as np
from calfram.calibration_framework import CalibrationFramework

class TestCalibrationFramework(unittest.TestCase):
    def setUp(self):
        self.cf = CalibrationFramework()
        
        # Create some sample data for testing
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 1000)
        self.y_prob = np.random.random((1000, 2))
        self.y_prob = self.y_prob / self.y_prob.sum(axis=1, keepdims=True)
        self.y_pred = np.argmax(self.y_prob, axis=1)

    def test_select_probability(self):
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        
        self.assertIn('0', classes_scores)
        self.assertIn('1', classes_scores)
        self.assertEqual(len(classes_scores['0']['proba']), 1000)
        self.assertEqual(len(classes_scores['0']['y']), 1000)
        self.assertEqual(classes_scores['0']['y_one_hot_nclass'].shape, (1000, 2))
        self.assertEqual(classes_scores['0']['y_prob_one_hotnclass'].shape, (1000, 2))
        self.assertEqual(classes_scores['0']['y_pred_one_hotnclass'].shape, (1000, 2))

    def test_calibrationdiagnosis(self):
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        measures, binning_dict = self.cf.calibrationdiagnosis(classes_scores)
        
        self.assertIn('0', measures)
        self.assertIn('1', measures)
        self.assertIn('ece_acc', measures['0'])
        self.assertIn('ece_fp', measures['0'])
        self.assertIn('ec_g', measures['0'])
        self.assertIn('brier_loss', measures['0'])
        
        self.assertIn('0', binning_dict)
        self.assertIn('1', binning_dict)
        self.assertIn('bins', binning_dict['0'])
        self.assertIn('binids', binning_dict['0'])
        self.assertIn('binfr', binning_dict['0'])

    def test_classwise_calibration(self):
        classes_scores = self.cf.select_probability(self.y_true, self.y_prob, self.y_pred)
        measures, _ = self.cf.calibrationdiagnosis(classes_scores)
        class_wise_metrics = self.cf.classwise_calibration(measures)
        
        self.assertIn('ec_g', class_wise_metrics)
        self.assertIn('ec_dir', class_wise_metrics)
        self.assertIn('ece_freq', class_wise_metrics)
        self.assertIn('ece_acc', class_wise_metrics)
        self.assertIn('ec_underconf', class_wise_metrics)
        self.assertIn('ec_overconf', class_wise_metrics)
        self.assertIn('brierloss', class_wise_metrics)

    def test_end_points(self):
        x = np.array([0.1, 0.5, 0.9])
        y = np.array([0.2, 0.6, 0.8])
        result = self.cf.end_points(x, y)
        
        self.assertEqual(result.shape, (4, 2))
        np.testing.assert_array_almost_equal(result[0], [0, 0])
        np.testing.assert_array_almost_equal(result[-1], [0.9, 0.8])

    def test_add_tilde(self):
        pts = np.array([[0, 0], [0.5, 0.6], [1, 1]])
        result = self.cf.add_tilde(pts)
        
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.5])

    def test_h_triangle(self):
        new_pts = np.array([[0, 0], [0.5, 0.6], [1, 1]])
        tilde = np.array([[0, 0], [0.5, 0.5], [1, 1]])
        result = self.cf.h_triangle(new_pts, tilde)
        
        self.assertEqual(result.shape, (2,))
        self.assertGreater(result[0], 0)

    def test_underbelow_line(self):
        pts = np.array([[0, 0], [0.4, 0.3], [0.6, 0.7], [1, 1]])
        result = self.cf.underbelow_line(pts)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1], 'right')
        self.assertEqual(result[2], 'left')

    def test_split_probabilities(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        prob_ranges, bin_edges = self.cf.split_probabilities(probs, 3)
        
        self.assertEqual(len(prob_ranges), 3)
        self.assertEqual(len(bin_edges), 3)
        self.assertAlmostEqual(bin_edges[-1], 0.9)

    def test_compute_equal_mass_bin_heights(self):
        data = [(0.1, 0), (0.3, 1), (0.5, 1), (0.7, 0), (0.9, 1)]
        result = self.cf.compute_equal_mass_bin_heights(data, 2)
        
        self.assertEqual(len(result), 2)
        self.assertGreaterEqual(result[0], 0)
        self.assertLessEqual(result[0], 1)

    def test_is_monotonic(self):
        self.assertTrue(self.cf.is_monotonic([0.1, 0.3, 0.5, 0.7]))
        self.assertFalse(self.cf.is_monotonic([0.1, 0.5, 0.3, 0.7]))

    def test_monotonic_sweep_calibration(self):
        data = [(0.1, 0), (0.3, 1), (0.5, 1), (0.7, 0), (0.9, 1)]
        result = self.cf.monotonic_sweep_calibration(data, 5)
        
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 5)

if __name__ == '__main__':
    unittest.main()

class TestTieSafeBinningAndIdeal(unittest.TestCase):
    """The tie-safe binning (strategy='pooled_sweep') and ideal_calibration."""

    def setUp(self):
        self.cf = CalibrationFramework()
        rng = np.random.default_rng(0)
        # scores on a 0.01 grid, so that many items share one value
        self.p = np.round(rng.beta(2, 2, 3000), 2)
        self.y_cal = (rng.random(3000) < self.p).astype(int)          # calibrated by construction
        self.y_sharp = (rng.random(3000) < 0.5 + 0.5 * (self.p - 0.5)).astype(int)  # flatter truth: over-confident scores
        self.y_prob = np.column_stack([1 - self.p, self.p])
        self.y_pred = (self.p >= 0.5).astype(int)

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
        scores = self.cf.select_probability(self.y_sharp, self.y_prob, self.y_pred)
        measures, binning = self.cf.calibrationdiagnosis({'1': scores['1']}, strategy='pooled_sweep')
        self.assertTrue(0.0 <= measures['1']['ec_g'] <= 1.0)
        self.assertTrue(np.isfinite(measures['1']['ece_fp']))
        self.assertEqual(len(binning['1']['bins']), len(binning['1']['binfr']))
        # a tied block is never split: every item of one value shares a bin
        for v in np.unique(self.p):
            self.assertEqual(len(np.unique(binning['1']['binids'][self.p == v])), 1)

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

    def test_ideal_is_reproducible_and_restores_the_global_generator(self):
        np.random.seed(7)
        before = np.random.get_state()
        a = self.cf.ideal_calibration(self.y_cal, self.y_prob, self.y_pred, strategy=15, n_sim=20, seed=3)
        after = np.random.get_state()
        b = self.cf.ideal_calibration(self.y_cal, self.y_prob, self.y_pred, strategy=15, n_sim=20, seed=3)
        self.assertEqual(a['1']['ec_g']['ideal'], b['1']['ec_g']['ideal'])
        self.assertEqual(a['1']['ec_g']['p_value'], b['1']['ec_g']['p_value'])
        self.assertTrue(np.array_equal(before[1], after[1]))

    def test_signed_index_matches_its_sides_and_bounds_ec_g(self):
        scores = self.cf.select_probability(self.y_sharp, self.y_prob, self.y_pred)
        m, _ = self.cf.calibrationdiagnosis({'1': scores['1']}, strategy='pooled_sweep')
        r = m['1']
        self.assertAlmostEqual(r['ec_signed'], r['ec_signed_over'] + r['ec_signed_under'])
        self.assertGreater(r['ec_signed'], 0.0)  # over-confident scores: predictions above outcomes
        self.assertLessEqual(abs(r['ec_signed']), 1.0 - r['ec_g'] + 1e-9)
        # a mirrored problem flips the sign
        scores_m = self.cf.select_probability(1 - self.y_sharp, self.y_prob[:, ::-1], 1 - self.y_pred)
        r_m = self.cf.calibrationdiagnosis({'1': scores_m['1']}, strategy='pooled_sweep')[0]['1']
        self.assertAlmostEqual(r_m['ec_signed'], -r['ec_signed'])

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
