"""balance='mass': ECI balance weighted by the share of all data in each bin (see PATCH_NOTES.md)."""
import unittest
import warnings

import numpy as np

from calfram.calibration_framework import CalibrationFramework
from tests.test_tie_safe import continuous_scores, quantised_scores


def diagnose(cf, score, y, **kwargs):
    """Measures and bins of the positive class of a binary task; an exception inside calibrationdiagnosis fails."""
    proba = np.column_stack([1 - score, score])
    classes_scores = cf.select_probability(y, proba, (score >= 0.5).astype(int))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measures, binning_dict = cf.calibrationdiagnosis(classes_scores, **kwargs)
    errors = [str(w.message) for w in caught if str(w.message).startswith("Error")]
    assert not errors, errors
    return measures['1'], binning_dict['1']


def blocks(spec):
    """Scores and labels made of blocks (score, n, positives): with one bin per value each block is a bin."""
    score = np.concatenate([np.full(n, p) for p, n, _ in spec])
    y = np.concatenate([np.arange(n) < k for _, n, k in spec]).astype(int)
    return score, y


class TestMassBalance(unittest.TestCase):

    def setUp(self):
        self.cf = CalibrationFramework()

    def test_default_unchanged(self):
        score, y = quantised_scores(3000, 1.0, 0)
        default, _ = diagnose(self.cf, score, y, adaptive=True, tie_safe=True)
        sides, _ = diagnose(self.cf, score, y, adaptive=True, tie_safe=True, balance='sides')
        mass, _ = diagnose(self.cf, score, y, adaptive=True, tie_safe=True, balance='mass')
        self.assertNotIn('ec_dir_sides', default)
        self.assertEqual(set(default), set(sides))
        self.assertEqual(default['ec_dir'], sides['ec_dir'])
        # the mass option changes ec_dir only, and keeps the old value under ec_dir_sides
        self.assertEqual(set(mass) - set(default), {'ec_dir_sides', 'ec_underconf_mass', 'ec_overconf_mass'})
        self.assertEqual(mass['ec_dir_sides'], default['ec_dir'])
        for key in ('ec_g', 'ece_fp', 'ec_underconf', 'ec_overconf'):
            self.assertEqual(mass[key], default[key])
        np.testing.assert_array_equal(mass['relative-freq'], default['relative-freq'])

    def test_invalid_balance(self):
        score, y = quantised_scores(500, 0.0, 0)
        with self.assertRaises(ValueError):
            diagnose(self.cf, score, y, balance='weighted')

    def test_tiny_over_forecast_bin(self):
        # 3000 rows at 0.3 with an observed frequency of 0.5 (under-forecast, d = 0.2 / 0.7), plus 3 rows at 0.9
        # that are all negative (over-forecast, d = 1). The 3 rows flip the old balance, not the mass balance.
        base = [(0.3, 3000, 1500)]
        score, y = blocks(base)
        m0, _ = diagnose(self.cf, score, y, strategy='doane', balance='mass')
        score, y = blocks(base + [(0.9, 3, 0)])
        m1, bins = diagnose(self.cf, score, y, strategy='doane', balance='mass')
        self.assertEqual(len(bins['binfr']), 2)
        d_under = 0.2 / 0.7
        self.assertAlmostEqual(m0['ec_dir'], -d_under, places=6)
        self.assertAlmostEqual(m0['ec_dir_sides'], -d_under, places=6)
        self.assertAlmostEqual(m1['ec_dir_sides'], 1 - d_under, places=6)   # +0.714: reads as over-forecast
        self.assertAlmostEqual(m1['ec_dir'], (3 * 1 - 3000 * d_under) / 3003, places=6)  # -0.284
        self.assertLess(m1['ec_dir'], 0)
        self.assertLess(abs(m1['ec_dir'] - m0['ec_dir']), 0.002)

    def test_bounded_by_ec_g(self):
        rng = np.random.default_rng(0)
        for seed in range(30):
            shift = rng.normal(0, 1.5)
            score, y = (quantised_scores if seed % 2 else continuous_scores)(int(rng.integers(200, 3000)), shift, seed)
            for kwargs in (dict(strategy=10), dict(strategy='doane'), dict(adaptive=True),
                           dict(adaptive=True, tie_safe=True), dict(strategy=15, tie_safe=True)):
                m, _ = diagnose(self.cf, score, y, balance='mass', **kwargs)
                self.assertLessEqual(abs(m['ec_dir']), 1 - m['ec_g'] + 1e-12, (seed, kwargs))
                # and it is the signed version of the same sum
                w = m['relative-freq'] / np.sum(m['relative-freq'])
                d = 1 - m['ec_l_all']
                s = (m['where'] == 'right').astype(float) - (m['where'] == 'left').astype(float)
                self.assertAlmostEqual(m['ec_dir'], np.sum(w * s * d), places=12)

    def test_side_shares(self):
        # ec_overconf_mass - ec_underconf_mass = ec_dir, and the two add up to 1 - ec_g
        rng = np.random.default_rng(1)
        for seed in range(20):
            gen = quantised_scores if seed % 2 else continuous_scores
            score, y = gen(int(rng.integers(200, 3000)), rng.normal(0, 1.5), seed)
            for kwargs in (dict(strategy=10), dict(adaptive=True, tie_safe=True), dict(strategy=15, tie_safe=True)):
                m, _ = diagnose(self.cf, score, y, balance='mass', **kwargs)
                self.assertAlmostEqual(m['ec_overconf_mass'] - m['ec_underconf_mass'], m['ec_dir'], places=12)
                self.assertAlmostEqual(m['ec_overconf_mass'] + m['ec_underconf_mass'], 1 - m['ec_g'], places=9)
                self.assertGreaterEqual(min(m['ec_overconf_mass'], m['ec_underconf_mass']), 0)
        # an empty side contributes 0, and the within-side measures are unchanged
        score, y = blocks([(0.2, 100, 10), (0.5, 100, 30), (0.8, 100, 60)])
        m, _ = diagnose(self.cf, score, y, strategy=10, balance='mass')
        self.assertEqual(m['ec_underconf_mass'], 0.0)
        self.assertAlmostEqual(m['ec_overconf_mass'], m['ec_dir'], places=12)
        self.assertTrue(np.isnan(m['ec_underconf']))
        self.assertAlmostEqual(m['ec_overconf'], 1 - np.mean([0.1 / 0.8, 0.2 / 0.5, 0.2 / 0.8]), places=12)

    def test_identity_when_one_side(self):
        over = [(0.2, 100, 10), (0.5, 100, 30), (0.8, 100, 60)]      # observed frequency below every score
        under = [(0.2, 100, 30), (0.5, 100, 70), (0.8, 100, 95)]     # above every score
        for spec, sign in ((over, 1), (under, -1)):
            score, y = blocks(spec)
            m, _ = diagnose(self.cf, score, y, strategy='doane', balance='mass')
            self.assertTrue(np.all(m['where'] == ('right' if sign > 0 else 'left')))
            self.assertAlmostEqual(m['ec_dir'], sign * (1 - m['ec_g']), places=10)
        # a bin on the diagonal adds weight but no distance: the identity still holds
        score, y = blocks(over + [(0.25, 100, 25)])  # 0.25 is exact in binary: x == y, not off by one ulp
        m, _ = diagnose(self.cf, score, y, strategy='doane', balance='mass')
        self.assertIn('lie', m['where'])
        self.assertAlmostEqual(m['ec_dir'], 1 - m['ec_g'], places=10)

    def test_sign(self):
        for seed in range(3):
            for gen in (continuous_scores, quantised_scores):
                for kwargs in (dict(strategy=10), dict(adaptive=True), dict(strategy='doane')):
                    over, _ = diagnose(self.cf, *gen(3000, 1.0, seed), balance='mass', **kwargs)
                    under, _ = diagnose(self.cf, *gen(3000, -1.0, seed), balance='mass', **kwargs)
                    self.assertGreater(over['ec_dir'], 0.02, (gen.__name__, kwargs, seed))
                    self.assertLess(under['ec_dir'], -0.02, (gen.__name__, kwargs, seed))

    def test_perfect_calibration(self):
        # Labels drawn from the scores. With bins that depend on the scores only (int strategy), each bin's
        # contribution is w_b * (x_b - y_b) / max(x_b, 1 - x_b), with E[y_b] = x_b: the mass balance is unbiased.
        # The tolerance is 4 standard errors of the mean over the draws (about 0.007 here), so a false failure
        # has probability below 1e-4 and a bias of the size of the old balance's (0.01-0.8, below) fails.
        rng = np.random.default_rng(12345)
        draws = 300
        results = {'strategy10': [], 'per_value_sides': [], 'per_value_mass': []}
        for _ in range(draws):
            score = rng.beta(2, 5, 500)
            y = (rng.random(500) < score).astype(int)
            m, _ = diagnose(self.cf, score, y, strategy=10, balance='mass')
            results['strategy10'].append(m['ec_dir'])
            m, _ = diagnose(self.cf, score, y, strategy='doane', balance='mass')
            results['per_value_sides'].append(m['ec_dir_sides'])
            results['per_value_mass'].append(m['ec_dir'])
        for key in ('strategy10', 'per_value_mass'):
            values = np.array(results[key])
            se = values.std(ddof=1) / np.sqrt(draws)
            self.assertLess(abs(values.mean()), 4 * se, key)
            self.assertLess(abs(values.mean()), 0.01, key)
        # contrast: with one bin per value (the default str strategy on continuous scores) every bin is a single
        # row, and the old balance of a perfectly calibrated model is about -0.56
        self.assertLess(np.mean(results['per_value_sides']), -0.3)

    def test_tie_safe(self):
        for seed in range(3):
            for kwargs in (dict(adaptive=True, tie_safe=True), dict(strategy=15, tie_safe=True)):
                over, bins = diagnose(self.cf, *quantised_scores(20000, 1.0, seed), balance='mass', **kwargs)
                under, _ = diagnose(self.cf, *quantised_scores(20000, -1.0, seed), balance='mass', **kwargs)
                calibrated, _ = diagnose(self.cf, *quantised_scores(20000, 0.0, seed), balance='mass', **kwargs)
                self.assertGreater(over['ec_dir'], 0.02)
                self.assertLess(under['ec_dir'], -0.02)
                self.assertLess(abs(calibrated['ec_dir']), 0.01)
                for m in (over, under, calibrated):
                    self.assertLessEqual(abs(m['ec_dir']), 1 - m['ec_g'] + 1e-12)
                # same bins as without the option
                _, bins_default = diagnose(self.cf, *quantised_scores(20000, 1.0, seed), **kwargs)
                np.testing.assert_array_equal(bins['binids'], bins_default['binids'])

    def test_classwise(self):
        rng = np.random.default_rng(0)
        n, k = 3000, 5
        logits = rng.normal(0, 2, (n, k))
        true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y = np.array([rng.choice(k, p=p) for p in true])
        sharp = true ** 2 / (true ** 2).sum(1, keepdims=True)   # over-confident
        classes_scores = self.cf.select_probability(y, sharp, sharp.argmax(1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            measures, _ = self.cf.calibrationdiagnosis(classes_scores, strategy=10, balance='mass')
        summary = self.cf.classwise_calibration(measures)
        self.assertAlmostEqual(summary['ec_dir'], round(np.nanmean([m['ec_dir'] for m in measures.values()]), 3))
        for m in measures.values():
            self.assertLessEqual(abs(m['ec_dir']), 1 - m['ec_g'] + 1e-12)


if __name__ == '__main__':
    unittest.main()
