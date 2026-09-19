"""Follow-up fixes (see "Follow-up fixes" in PATCH_NOTES.md)."""
import unittest
import warnings

import numpy as np

from calfram.calibration_framework import CalibrationFramework
from tests.test_mass_balance import blocks, diagnose


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


if __name__ == '__main__':
    unittest.main()
