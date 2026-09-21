import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings

# str strategies: the rules of np.histogram_bin_edges, or 'unique' (one bin per unique score)
HISTOGRAM_RULES = ('auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt')
STR_STRATEGIES = HISTOGRAM_RULES + ('unique',)


def check_strategy(strategy: Union[int, str]) -> None:
    if isinstance(strategy, str) and strategy not in STR_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}: use an int (number of quantile bins), 'unique' (one bin per "
                         f"unique score) or one of the np.histogram_bin_edges rules {HISTOGRAM_RULES}.")


class CalibrationFramework:
    def __init__(self) -> None:
        self.ohe: OneHotEncoder = OneHotEncoder(sparse_output=False)

    def monotonic_sweep_calibration(self, data: List[Tuple[float, int]], n: int) -> int:
        b: int = 2
        while b <= n:
            bin_heights: List[float] = self.compute_equal_mass_bin_heights(data, b)
            if not self.is_monotonic(bin_heights):
                b -= 1
                break
            b += 1
        return b

    def monotonic_sweep_calibration_multiclass(self, Y_probs: NDArray[np.float64], Y: NDArray[np.int64], n_classes: int) -> int:
        if Y_probs.ndim == 1:
            binary_data: List[Tuple[float, int]] = [(prob, int(true_label == 1)) for prob, true_label in zip(Y_probs, Y)]
        else:
            binary_data = [(prob[1], int(true_label == 1)) for prob, true_label in zip(Y_probs, Y)]

        b: int = self.monotonic_sweep_calibration(binary_data, len(Y_probs))
        return min(b, len(np.unique(Y_probs)))

    def monotonic_sweep_calibration_tie_safe(self, cum_counts: NDArray[np.int64], cum_positives: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Monotonic sweep that never splits a block of tied scores.

        cum_counts and cum_positives are the cumulative number of samples and of positives over the distinct
        scores in increasing order, both starting with 0. The bins checked for each b are the ones of
        compute_tie_safe_cuts, so they only depend on the counts per distinct score and not on the order of
        the rows. Returns the cuts of the last monotonic binning (no cut stands for a single bin), as checked
        by is_monotonic_tie_safe. With all-distinct scores the bins, and therefore the selected b, are those
        of monotonic_sweep_calibration.
        """
        n: int = int(cum_counts[-1])
        n_unique: int = len(cum_counts) - 1
        ties: bool = n_unique < n
        best_cuts: NDArray[np.int64] = np.array([], dtype=np.int64)
        b: int = 2
        while b <= n:
            cuts: NDArray[np.int64] = self.compute_tie_safe_cuts(cum_counts, b)
            idx: NDArray[np.int64] = np.concatenate([[0], cuts, [n_unique]])
            if not self.is_monotonic_tie_safe(np.diff(cum_counts[idx]), np.diff(cum_positives[idx]), robust=ties):
                break
            best_cuts = cuts
            if len(cuts) + 1 >= n_unique:
                break  # One bin per distinct score already: nothing left to refine
            b += 1
        return best_cuts

    def binning_schema(self, prob: NDArray[np.float64], Y: NDArray[np.int64], method: Union[int, str] = 15, ndim: Union[int, None] = 1, adaptive: bool = False, tie_safe: bool = False) -> Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]:
        if ndim is not None:
            prob = prob[:, ndim]
        else:
            prob = np.max(prob, axis=-1)

        n_classes: int = len(np.unique(Y))

        if adaptive and not tie_safe:
            b: int = self.monotonic_sweep_calibration_multiclass(prob, Y, n_classes)
            b = min(b, len(np.unique(prob)))
        elif adaptive:
            b = None  # tie_safe: the sweep below returns the bins themselves, not only their number
        else:
            if not isinstance(method, (int, str)):
                raise ValueError("Please provide an int or str object for selecting the number of bins or select adaptive = True for monotonic sweep.")
            check_strategy(method)
            b = method

        if prob.ndim == 1:
            binary_data: List[Tuple[float, int]] = [(probs, int(true_label == 1)) for probs, true_label in zip(prob, Y)]
        else:
            binary_data = [(probs[1], int(true_label == 1)) for probs, true_label in zip(prob, Y)]
        
        unique_probs = sorted(set([data[0] for data in binary_data]))
        n_unique = len(unique_probs)
        per_value: bool = False  # True when there is one bin per unique probability value
        
        if n_unique < 2:
            warnings.warn(f"Only {n_unique} unique probability values found. Creating artificial bins.")
            if n_unique == 1:
                bin_edges = np.array([0.0, unique_probs[0], 1.0])
            else:
                bin_edges = np.array([0.0, 1.0])
        elif isinstance(b, int) and n_unique < b:
            warnings.warn(f"Only {n_unique} unique probability values found, using these instead of {b} bins.")
            bin_edges = np.array(unique_probs)
            # Only with tie_safe: the default keeps the bins of the previous versions, where the two largest
            # values share a bin when 1.0 is one of them
            per_value = tie_safe
        elif tie_safe and (adaptive or isinstance(b, int)):
            # Bins made of whole unique values: a block of tied probabilities is never split and no value
            # lies on an edge, so the result depends neither on the order of the rows nor on np.digitize
            # being left- or right-closed
            unique_values, inverse, counts = np.unique(prob, return_inverse=True, return_counts=True)
            positives = np.bincount(inverse.ravel(), weights=(np.ravel(Y) == 1), minlength=n_unique)
            cum_counts = np.concatenate([[0], np.cumsum(counts)])
            if adaptive:
                cuts = self.monotonic_sweep_calibration_tie_safe(cum_counts, np.concatenate([[0.0], np.cumsum(positives)]))
            else:
                cuts = self.compute_tie_safe_cuts(cum_counts, b)
            bin_edges = self.tie_safe_bin_edges(unique_values, cuts)
        else:
            # Use quantile-based binning for better distribution
            if isinstance(b, int):
                bin_edges = np.quantile(prob, np.linspace(0, 1, min(b + 1, n_unique)))
                bin_edges = np.unique(bin_edges)  # Remove duplicates
            elif b == 'unique':
                bin_edges = np.array(unique_probs)
                per_value = True
            else:
                # A rule of np.histogram_bin_edges: equal-width bins between the smallest and the largest score, with
                # np.histogram's assignment (left-closed bins, the last one closed); no padding to [0, 1]
                bin_edges = np.histogram_bin_edges(prob, bins=b)
                return self.nonempty_bins(bin_edges, np.digitize(prob, bin_edges[1:-1]), len(prob))

        if bin_edges[0] > 0:
            bin_edges = np.concatenate([[0.0], bin_edges])
        if bin_edges[-1] < 1:
            bin_edges = np.concatenate([bin_edges, [1.0]])
        elif per_value:
            # np.digitize only gets the inner edges: without a closing edge the largest value (1.0) would
            # not be one of them and would share the bin of the second largest value
            bin_edges = np.concatenate([bin_edges, [bin_edges[-1]]])
        
        binids: NDArray[np.int64] = np.digitize(prob, bin_edges[1:-1])
        return self.nonempty_bins(bin_edges, binids, len(prob))

    @staticmethod
    def nonempty_bins(bin_edges: NDArray[np.float64], binids: NDArray[np.int64], n: int) -> Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]:
        bin_counts = np.bincount(binids, minlength=len(bin_edges) - 1)
        relative_freq_bin: NDArray[np.float64] = bin_counts / n
        
        # Remove empty bins
        non_empty = bin_counts > 0
        if not np.any(non_empty):
            warnings.warn("All bins are empty!")
            return {
                'bins': bin_edges,
                'binids': binids,
                'binfr': relative_freq_bin
            }

        return {
            'bins': bin_edges[:-1][non_empty],  # Keep only non-empty bin edges
            'binids': binids,
            'binfr': relative_freq_bin[non_empty]
        }

    def calibrationcurve(self, y_true: NDArray[np.int64], y_prob: NDArray[np.float64], strategy: Union[int, str] = 10, undersampling: bool = False, adaptive: bool = False, tie_safe: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]]:
        np.random.seed(123)

        labels: NDArray[np.int64] = np.unique(y_true)
        if len(labels) > 2:
            raise ValueError(f"Only binary classification is supported. Provided labels {labels}. For Multiclass use 1 vs All Approach.")

        bins_dict: Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]] = self.binning_schema(y_prob, y_true, method=strategy, ndim=1, adaptive=adaptive, tie_safe=tie_safe)
        
        if y_prob.ndim == 1:
            prob_values = y_prob
        else:
            prob_values = y_prob[:, 1]
        
        bin_sums: NDArray[np.float64] = np.bincount(bins_dict['binids'], weights=prob_values, minlength=len(bins_dict['bins']))
        bin_true: NDArray[np.float64] = np.bincount(bins_dict['binids'], weights=y_true.squeeze(), minlength=len(bins_dict['bins']))
        bin_total: NDArray[np.int64] = np.bincount(bins_dict['binids'], minlength=len(bins_dict['bins']))
        
        nonzero: NDArray[np.bool_] = bin_total != 0

        prob_true: NDArray[np.float64] = bin_true[nonzero] / bin_total[nonzero]
        prob_pred: NDArray[np.float64] = bin_sums[nonzero] / bin_total[nonzero]

        return prob_true, prob_pred, bins_dict

    def select_probability(self, y_actual: NDArray[np.int64], y_prob: NDArray[np.float64], y_pred: NDArray[np.int64]) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        """
        Fixed version that handles class probability alignment correctly
        """

        unique_labels = np.unique(y_actual)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        y_actual_mapped = np.array([label_map[label] for label in y_actual])
        y_pred_mapped = np.array([label_map[label] for label in y_pred])
        
        self.ohe.fit(np.arange(len(unique_labels)).reshape(-1, 1))
        y_one_hot: NDArray[np.float64] = self.ohe.transform(y_actual_mapped.reshape(-1, 1))
        y_pred_hot: NDArray[np.float64] = self.ohe.transform(y_pred_mapped.reshape(-1, 1))
        
        # Verify probability matrix dimensions
        if y_prob.shape[1] != len(unique_labels):
            raise ValueError(f"Probability matrix has {y_prob.shape[1]} columns but found {len(unique_labels)} unique labels")
        
        y_prob_one_hot: NDArray[np.float64] = y_prob.copy()

        final_dict: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        
        for i in range(len(unique_labels)):
            y_class: NDArray[np.int64] = y_actual_mapped.copy()
            indices_class: NDArray[np.int64] = np.argwhere(y_class == i).flatten()
            indices_other: NDArray[np.int64] = np.argwhere(y_class != i).flatten()

            y_proba_class: NDArray[np.float64] = y_prob[:, i].reshape(-1, 1)
            y_proba_rest: NDArray[np.float64] = 1 - y_proba_class
            new_y_proba: NDArray[np.float64] = np.concatenate([y_proba_rest, y_proba_class], axis=1)
            
            y_class[indices_class] = 1
            y_class[indices_other] = 0
            
            dict_clss: Dict[str, NDArray[np.float64]] = {
                'proba': new_y_proba,
                'y': y_class,
                'y_one_hot_nclass': y_one_hot,
                'y_prob_one_hotnclass': y_prob_one_hot,
                'y_pred_one_hotnclass': y_pred_hot
            }

            final_dict[str(i)] = dict_clss

        return final_dict

    def calibrationdiagnosis(self, classes_scores: Dict[str, Dict[str, NDArray[np.float64]]], strategy: Union[int, str] = 'doane', undersampling: bool = False, adaptive: bool =False, tie_safe: bool = False, balance: str = 'sides') -> Tuple[Dict[str, Dict[str, Union[float, NDArray[np.float64]]]], Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]]]:
        """
        ECI measures of each class, computed on the bins of calibrationcurve.

        ec_dir (ECI_balance) is signed, positive when the model over-forecasts (mean prediction above the observed
        frequency, points below the diagonal) and negative when it under-forecasts. `balance` selects how it is
        computed from the normalised bin distances d_b (the same distances as ec_g):

        - 'sides' (default, unchanged): mean of d_b over the over-forecast bins minus the mean over the
          under-forecast bins, each side weighted only within itself. The share of the data on each side does not
          enter, so a single small bin alone on one side counts as much as the rest of the data on the other.
        - 'mass': sum over all bins of w_b * s_b * d_b, with w_b the bin's share of the data (binfr), s_b = +1 for
          over-forecast bins, -1 for under-forecast bins and 0 on the diagonal. |ec_dir| <= 1 - ec_g, with
          equality when all bins are on one side. The old value is then also returned as 'ec_dir_sides', together
          with 'ec_underconf_mass' and 'ec_overconf_mass': the sums of w_b * d_b over the under- and over-forecast
          bins (each side's share of 1 - ec_g, 0 for an empty side), so that ec_dir = ec_overconf_mass -
          ec_underconf_mass and ec_underconf_mass + ec_overconf_mass = 1 - ec_g up to the bins on the diagonal.
          Unlike ec_underconf / ec_overconf (1 - the mean distance within one side, unchanged) they are
          miscalibration shares: 0 is best.

        See PATCH_NOTES.md.
        """
        if balance not in ('sides', 'mass'):
            raise ValueError(f"balance must be 'sides' or 'mass', got {balance!r}.")
        check_strategy(strategy)  # here, so that an unknown name raises instead of becoming a warning per class
        measures: Dict[str, Dict[str, Union[float, NDArray[np.float64]]]] = {}
        binning_dict: Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]] = {}
        
        for i in classes_scores.keys():
            try:
                y, x, bins_dict = self.calibrationcurve(classes_scores[i]['y'], classes_scores[i]['proba'], strategy=strategy, undersampling=undersampling, adaptive=adaptive, tie_safe=tie_safe)
                new_pts: NDArray[np.float64] = self.end_points(x, y)

                # Normalised distance of each bin to the diagonal: |y - x| / max(x, 1 - x), i.e. the distance of
                # (x, y) to the diagonal divided by that of the farthest point in the same column, (x, 1) or (x, 0).
                # This is what the triangle heights of h_triangle_safe computed (to 3e-10 on random points), without
                # Heron's formula, which lost up to ~4e-6 when two consecutive bins had (almost) the same x.
                pts_distance_norm: NDArray[np.float64] = self.normalised_distance(x, y)
                
                where_are: List[str] = self.underbelow_line(new_pts[1:])  

                # binfr (non-empty bins of binning_schema) and the curve points (non-empty bins of calibrationcurve)
                # must describe the same bins, in the same order; the weights below index binfr with masks on the points
                if len(where_are) > 0 and len(bins_dict['binfr']) != len(where_are):
                    raise ValueError(f"{len(bins_dict['binfr'])} bin weights for {len(where_are)} calibration points.")

                mask_left: NDArray[np.bool_] = np.array([w == 'left' for w in where_are])
                mask_right: NDArray[np.bool_] = np.array([w == 'right' for w in where_are])
            
                # A curve whose points all lie on the diagonal is measured (ec_g = 1), not reported as NaN; only the
                # measures of the empty sides are NaN
                if len(where_are) == 0:
                    dict_msr: Dict[str, Union[float, NDArray[np.float64]]] = {
                        'ece_acc': np.nan, 'ece_fp': np.nan, 'ec_g': np.nan, 'ec_under': np.nan, 'under_fr': np.nan,
                        'ec_over': np.nan, 'over_fr': np.nan, 'ec_underconf': np.nan, 'ec_overconf': np.nan,
                        'ec_dir': np.nan, 'over_pts': np.nan, 'under_pts': np.nan, 'ec_l_all': np.nan, 'where': np.nan,
                        'relative-freq': np.nan, 'x': np.nan, 'y': np.nan
                    }
                    if balance == 'mass':
                        dict_msr.update(ec_dir_sides=np.nan, ec_underconf_mass=np.nan, ec_overconf_mass=np.nan)
                else:
                    up_dist: NDArray[np.float64] = pts_distance_norm[mask_left]
                    below_dist: NDArray[np.float64] = pts_distance_norm[mask_right]
                    up_pts: NDArray[np.float64] = new_pts[1:][mask_left]
                    below_pts: NDArray[np.float64] = new_pts[1:][mask_right]
                    up_weight: NDArray[np.float64] = bins_dict['binfr'][mask_left]
                    below_weight: NDArray[np.float64] = bins_dict['binfr'][mask_right]

                    # Fix: Safe weighted average calculations
                    if len(bins_dict['binfr']) > 0 and np.sum(bins_dict['binfr']) > 0:
                        fcc_g: float = 1 - np.average(pts_distance_norm, weights=bins_dict['binfr'])
                    else:
                        fcc_g = np.nan
                        
                    if len(up_weight) != 0 and np.sum(up_weight) > 0:
                        up_weight1: NDArray[np.float64] = up_weight / np.sum(up_weight)
                        fcc_underconf: float = 1 - np.average(up_dist, weights=up_weight1)
                    else:
                        fcc_underconf = np.nan
                        up_weight = np.array([])
                        
                    if len(below_weight) != 0 and np.sum(below_weight) > 0:
                        below_weight1: NDArray[np.float64] = below_weight / np.sum(below_weight)
                        fcc_overconf: float = 1 - np.average(below_dist, weights=below_weight1)
                    else:
                        fcc_overconf = np.nan
                        below_weight = np.array([])
                        
                    if len(up_weight) > 0 and len(below_weight) > 0 and np.sum(up_weight) > 0 and np.sum(below_weight) > 0:    
                        fcc_dir: float = np.average(below_dist, weights=below_weight) - np.average(up_dist, weights=up_weight)
                    elif len(up_weight) == 0 and len(below_weight) > 0 and np.sum(below_weight) > 0:
                        fcc_dir = np.average(below_dist, weights=below_weight)
                    elif len(up_weight) > 0 and len(below_weight) == 0 and np.sum(up_weight) > 0:
                        fcc_dir = -np.average(up_dist, weights=up_weight)
                    else:
                        fcc_dir = np.nan

                    if balance == 'mass':
                        # Signed and weighted by the share of all data: +d_b over-forecast, -d_b under-forecast,
                        # 0 on the diagonal, so 0 when every bin lies on the diagonal. NaN only without weights.
                        fcc_dir_sides: float = fcc_dir
                        weights: NDArray[np.float64] = np.asarray(bins_dict['binfr'], dtype=float)
                        side_sign: NDArray[np.float64] = mask_right.astype(float) - mask_left.astype(float)
                        if np.sum(weights) <= 0:
                            fcc_dir = fcc_under_mass = fcc_over_mass = np.nan
                        else:
                            share: NDArray[np.float64] = weights * pts_distance_norm / np.sum(weights)
                            # each side's share of 1 - ec_g (0 for an empty side); over - under = the mass balance
                            fcc_under_mass: float = float(np.sum(share[mask_left]))
                            fcc_over_mass: float = float(np.sum(share[mask_right]))
                            fcc_dir = float(np.sum(side_sign * share))

                    ece: float = self.compute_eces(classes_scores[i]['y_one_hot_nclass'], classes_scores[i]['y_prob_one_hotnclass'],
                                    classes_scores[i]['y_pred_one_hotnclass'], bins_dict['binids'],
                                    bins_dict['bins'], 'fp', int(i))
                    ece_acc: float = self.compute_eces(classes_scores[i]['y_one_hot_nclass'], classes_scores[i]['y_prob_one_hotnclass'],
                                    classes_scores[i]['y_pred_one_hotnclass'], bins_dict['binids'], bins_dict['bins'], 'acc', int(i))
                    brierloss: float = brier_score_loss(classes_scores[i]['y'], classes_scores[i]['proba'][:,1])
                
                    dict_msr = {
                        'ece_acc': ece_acc, 'ece_fp': ece, 'ec_g': fcc_g, 'ec_under': 1-up_dist, 'under_fr': up_weight, 'ec_over': 1-below_dist, 
                        'over_fr': below_weight, 'ec_underconf': fcc_underconf, 'ec_overconf': fcc_overconf, 
                        'ec_dir': fcc_dir, 'brier_loss': brierloss, 'over_pts': below_pts, 'under_pts': up_pts, 
                        'ec_l_all': 1-pts_distance_norm, 'where': np.array(where_are),
                        'relative-freq': bins_dict['binfr'], 'x': x, 'y': y
                    }
                    if balance == 'mass':
                        dict_msr['ec_dir_sides'] = fcc_dir_sides
                        dict_msr['ec_underconf_mass'] = fcc_under_mass
                        dict_msr['ec_overconf_mass'] = fcc_over_mass
            except Exception as e:
                warnings.warn(f"Error processing class {i}: {str(e)}")
                dict_msr = {
                    'ece_acc': np.nan, 'ece_fp': np.nan, 'ec_g': np.nan, 'ec_under': np.nan, 'under_fr': np.nan,
                    'ec_over': np.nan, 'over_fr': np.nan, 'ec_underconf': np.nan, 'ec_overconf': np.nan,
                    'ec_dir': np.nan, 'over_pts': np.nan, 'under_pts': np.nan, 'ec_l_all': np.nan, 'where': np.nan,
                    'relative-freq': np.nan, 'x': np.nan, 'y': np.nan, 'brier_loss': np.nan
                }
                if balance == 'mass':
                    dict_msr.update(ec_dir_sides=np.nan, ec_underconf_mass=np.nan, ec_overconf_mass=np.nan)

            measures[str(i)] = dict_msr
            binning_dict[str(i)] = bins_dict

        return measures, binning_dict

    @staticmethod
    def normalised_distance(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distance of the points (x, y) to the diagonal divided by the largest possible one at the same x:
        |y - x| / max(x, 1 - x), in [0, 1]. 1 - this is the local ECI of each bin."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.abs(y - x) / np.maximum(x, 1 - x)

    def h_triangle_safe(self, new_pts: NDArray[np.float64], tilde: NDArray[np.float64]) -> NDArray[np.float64]:
        """Safe version of h_triangle that handles degenerate cases. No longer used by calibrationdiagnosis, which
        computes the same normalised distances exactly with normalised_distance; kept for backward compatibility."""
        height_list: List[float] = []
        for idx in range(1, len(new_pts)):
            a, b, c = tilde[idx-1], tilde[idx], new_pts[idx]
            ab: float = np.linalg.norm(a - b)
            ac: float = np.linalg.norm(a - c)
            bc: float = np.linalg.norm(b - c)
            
            # Ensure minimum distance to avoid numerical issues
            ab = max(ab, 1e-10)
            ac = max(ac, 1e-10)
            bc = max(bc, 1e-10)
            
            # Check triangle inequality
            if ab + ac <= bc or ab + bc <= ac or ac + bc <= ab:
                # Degenerate triangle - points are collinear
                # Return distance from point to line
                if ab > 1e-10:
                    cross = np.cross(c - a, b - a)
                    h = abs(cross) / ab if not isinstance(cross, np.ndarray) else np.linalg.norm(cross) / ab
                else:
                    h = 0.0
            else:
                s: float = (ab + ac + bc) / 2
                area_sq = s * (s - ab) * (s - ac) * (s - bc)
                if area_sq < 0:
                    area_sq = 0  # Handle numerical errors
                area: float = np.sqrt(area_sq)
                h: float = 2 * area / ab if ab > 0 else 0.0
            
            height_list.append(h)
        return np.array(height_list)

    def compute_eces(self, y: NDArray[np.float64], prob: NDArray[np.float64], y_pred: NDArray[np.float64], 
                     binids: NDArray[np.int64], bins: NDArray[np.float64], groupby: str = 'fp', ndim: Optional[int] = None) -> float:
        try:
            if ndim is not None:
                prob = prob[:, ndim].copy()
                y = y[:, ndim].copy()
                y_pred = y_pred[:, ndim].copy()
            else:
                prob = np.max(prob, axis=1)
                y = np.argmax(y, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
                
            bin_total: NDArray[np.int64] = np.bincount(binids, minlength=len(bins))
            nonzero: NDArray[np.bool_] = bin_total != 0
            
            if not np.any(nonzero):
                return np.nan
                
            if groupby == 'fp':
                bin_true: NDArray[np.float64] = np.bincount(binids, weights=y, minlength=len(bins))
                prob_true: NDArray[np.float64] = bin_true[nonzero] / bin_total[nonzero]
                confscore_bins: NDArray[np.float64] = np.bincount(binids, weights=prob.squeeze())[nonzero] / bin_total[nonzero]       
            else:       
                prob_true = np.bincount(binids, weights=(y == y_pred), minlength=len(bins))
                confscore_bins = np.bincount(binids, weights=prob, minlength=len(bins))
                confscore_bins = confscore_bins[nonzero] / bin_total[nonzero]
                prob_true = prob_true[nonzero] / bin_total[nonzero]
                if ndim is not None:
                    mask: NDArray[np.bool_] = confscore_bins < 0.5
                    confscore_bins[mask] = 1 - confscore_bins[mask]
            
            loce_w: NDArray[np.float64] = bin_total[nonzero] / len(y)
            lece: NDArray[np.float64] = np.abs(prob_true - confscore_bins)
            ece: float = np.sum(loce_w * lece)
            return ece
        except Exception as e:
            warnings.warn(f"Error computing ECE: {str(e)}")
            return np.nan

    def classwise_calibration(self, measures: Dict[str, Dict[str, Union[float, NDArray[np.float64]]]]) -> Dict[str, float]:
        classes_global: float = np.nanmean([measures[key]['ec_g'] for key in measures.keys()]).round(3)
        classes_direction: float = np.nanmean([measures[key]['ec_dir'] for key in measures.keys()]).round(3)
        classes_underconf: float = np.nanmean([measures[key]['ec_underconf'] for key in measures.keys()]).round(3)
        classes_overconf: float = np.nanmean([measures[key]['ec_overconf'] for key in measures.keys()]).round(3)
        classes_ece: float = np.nanmean([measures[key]['ece_fp'] for key in measures.keys()]).round(3)
        classes_ece_acc: float = np.nanmean([measures[key]['ece_acc'] for key in measures.keys()]).round(3)
        classes_brier: float = np.nanmean([measures[key].get('brier_loss', np.nan) for key in measures.keys()]).round(3) / 2
        return {
            'ec_g': classes_global,
            'ec_dir': classes_direction,
            'ece_freq': classes_ece,
            'ece_acc': classes_ece_acc,
            'ec_underconf': classes_underconf,
            'ec_overconf': classes_overconf,
            'brierloss': classes_brier
        }

    def reliabilityplot(self, classes_scores: Dict[str, Dict[str, NDArray[np.float64]]], strategy: Union[int, str] = 'doane', split: bool = True, undersampling: bool = False, adaptive: bool = False, tie_safe: bool = False) -> None:
        marker_list: List[str] = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's']
        plt.figure(figsize=(10, 10))
        for idx, (i, class_score) in enumerate(classes_scores.items()):
            try:
                prob_true, prob_pred, _ = self.calibrationcurve(class_score['y'], class_score['proba'], strategy=strategy, undersampling=undersampling, adaptive=adaptive, tie_safe=tie_safe)

                plt.rcParams["font.weight"] = "bold"
                plt.rcParams["axes.labelweight"] = "bold"
                plt.rcParams['legend.title_fontsize'] = 'xx-small'
                plt.rc('grid', linestyle=":", color='black')
                plt.plot(prob_pred, prob_true, label=f'Class {i}', linestyle='--', markersize=3)
                plt.scatter(prob_pred, prob_true, marker=marker_list[idx % len(marker_list)])
            except Exception as e:
                warnings.warn(f"Error plotting class {i}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.legend(loc='lower right', fancybox=True, shadow=True, ncol=3, fontsize=7)
        plt.gca().set_aspect('equal', adjustable='box')
        if split: 
            plt.show()

    @staticmethod
    def end_points(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        points: NDArray[np.float64] = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        return np.concatenate([np.array([[0, 0]]), points], axis=0)

    @staticmethod
    def add_tilde(pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([pts[0]] + [np.array([pt[0], pt[0]]) for pt in pts[1:]])

    @staticmethod
    def h_triangle(new_pts: NDArray[np.float64], tilde: NDArray[np.float64]) -> NDArray[np.float64]:
        """Original h_triangle method - use h_triangle_safe instead"""
        height_list: List[float] = []
        for idx in range(1, len(new_pts)):
            a, b, c = tilde[idx-1], tilde[idx], new_pts[idx]
            ab: float = np.linalg.norm(a - b)
            ac: float = np.linalg.norm(a - c)
            bc: float = np.linalg.norm(b - c)
            ab = max(ab, 1e-10)
            s: float = (ab + ac + bc) / 2
            area: float = np.sqrt(s * (s - ab) * (s - ac) * (s - bc))
            h: float = 2 * area / ab
            height_list.append(h)
        return np.array(height_list)

    @staticmethod
    def underbelow_line(pts: NDArray[np.float64], atol: float = 1e-12) -> List[str]:
        """'left' (above the diagonal, y > x: under-forecast), 'right' (below, y < x: over-forecast) or 'lie' (on it)
        for each point (x, y). Points within atol of the diagonal lie on it: the mean score of a bin is a floating-point
        sum, so a bin that is exactly calibrated (100 rows at 0.4 with 40 positives) has x = 0.4000000000000001."""
        return [
                'lie' if np.isclose(pt[1], pt[0], rtol=0, atol=atol) else
                'left' if pt[1] > pt[0] else
                'right' for pt in pts
                ]

    @staticmethod
    def check_idx(pts: NDArray[np.float64]) -> List[int]:
        line: List[str] = CalibrationFramework.underbelow_line(pts)
        return [idx for idx in range(1, len(line) - 1) if line[idx] != line[idx+1]]

    @staticmethod
    def find_inters(dir_pts: NDArray[np.float64], dir_m: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[(q := -(m * x) + y) / (1 - m), q / (1 - m)] 
                         for (x, y), m in zip(dir_pts, dir_m)])

    @staticmethod
    def finite_diff(pts: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([1] + [(pts[i+1][1] - pts[i][1]) / (pts[i+1][0] - pts[i][0]) 
                               for i in range(1, len(pts)-1)] + [1])

    @staticmethod
    def find_points(x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[NDArray[np.float64], Union[NDArray[np.float64], float]]:
        pts: NDArray[np.float64] = CalibrationFramework.end_points(x, y)
        m: NDArray[np.float64] = CalibrationFramework.finite_diff(pts)
        dir_point_idx: List[int] = CalibrationFramework.check_idx(pts)
        if len(dir_point_idx) != 0:
            if np.all(pts[dir_point_idx[-1],:] == [1,1]):
                dir_point_idx = dir_point_idx[:-2]
            dir_m: NDArray[np.float64] = m[dir_point_idx]
            dir_pts: NDArray[np.float64] = pts[dir_point_idx,:]
            new_points: NDArray[np.float64] = CalibrationFramework.find_inters(dir_pts, dir_m)
            dir_point_idx = np.array([i+1 for i in dir_point_idx])
            new_pts: NDArray[np.float64] = np.insert(pts, dir_point_idx, new_points, axis=0)
            return new_pts, new_points
        else:
            return pts, np.nan

    @staticmethod
    def split_probabilities(probs: NDArray[np.float64], r: int) -> Tuple[List[NDArray[np.float64]], List[float]]:
        sorted_probs: NDArray[np.float64] = np.sort(probs)
        prob_ranges: List[NDArray[np.float64]] = np.array_split(sorted_probs, r)
        bin_edges: List[float] = [prob_range[-1] for prob_range in prob_ranges[:-1]] + [prob_ranges[-1][-1]]
        return prob_ranges, bin_edges

    @staticmethod
    def compute_bin_heights(data: List[Tuple[float, int]], b: int) -> List[float]:
        bin_edges: List[float] = [i / b for i in range(b + 1)]
        bin_counts: List[int] = [0] * b
        bin_heights: List[float] = [0] * b

        for score, true_label in data:
            for i in range(b):
                if bin_edges[i] <= score < bin_edges[i + 1]:
                    bin_counts[i] += 1
                    bin_heights[i] += true_label
                    break

        for i in range(b):
            if bin_counts[i] > 0:
                bin_heights[i] /= bin_counts[i]

        return bin_heights

    @staticmethod
    def compute_equal_mass_bin_heights(data: List[Tuple[float, int]], b: int) -> List[float]:
        if len(data) < b:
            warnings.warn(f"Not enough data points ({len(data)}) for {b} bins")
            b = len(data)
            
        data = sorted(data, key=lambda x: x[0])
        bin_size: int = max(1, len(data) // b)
        bin_heights: List[float] = []
        
        for i in range(b):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < b - 1 else len(data)
            if start_idx < len(data) and end_idx > start_idx:
                bin_data = data[start_idx:end_idx]
                if bin_data:
                    bin_heights.append(sum(true_label for _, true_label in bin_data) / len(bin_data))
                else:
                    bin_heights.append(0.0)
            else:
                bin_heights.append(0.0)
                
        return bin_heights

    @staticmethod
    def compute_tie_safe_cuts(cum_counts: NDArray[np.int64], b: int) -> NDArray[np.int64]:
        """
        Equal-mass bins made of whole distinct scores.

        Same targets as compute_equal_mass_bin_heights (a cut after every n // b samples, the last bin takes
        the remainder), but a bin can only end where a block of tied scores ends:
        1. each target k * (n // b), k = 1..b-1, is moved to the end of a tied block nearest to it (the
           targets that meet at the same place give one cut);
        2. a distinct score that holds n // b samples or more alone (a heavy block) gets a cut on both sides;
        3. a bin with fewer than half of n // b samples joins its smaller neighbour, and while there are
           more than b bins the smallest one does too (so a heavy block is a bin of its own unless a sliver
           next to it has no other neighbour).
        So a block is never split, no bin is a sliver, there are at most b bins (fewer when blocks take the
        place of several bins), and with all-distinct scores the bins are exactly those of
        compute_equal_mass_bin_heights.

        cum_counts is the cumulative number of samples over the distinct scores in increasing order,
        starting with 0. A returned cut k separates the distinct scores k - 1 and k.
        """
        n: int = int(cum_counts[-1])
        n_unique: int = len(cum_counts) - 1
        b = max(1, min(b, n))
        bin_size: int = max(1, n // b)

        targets: NDArray[np.int64] = bin_size * np.arange(1, b)
        if n_unique == n:
            return targets.astype(np.int64)  # All-distinct scores: plain equal-mass cuts

        # 1. The block end nearest to each target (the upper one when both are as near)
        upper: NDArray[np.int64] = np.searchsorted(cum_counts, targets, side='left')
        lower: NDArray[np.int64] = np.maximum(upper - 1, 0)
        nearest: NDArray[np.int64] = np.where(targets - cum_counts[lower] < cum_counts[upper] - targets, lower, upper)
        # 2. Heavy blocks start and end a bin
        heavy: NDArray[np.int64] = np.flatnonzero(np.diff(cum_counts) >= bin_size)
        cuts_arr = np.unique(np.concatenate([nearest, heavy, heavy + 1]))
        cuts: List[int] = cuts_arr[(cuts_arr > 0) & (cuts_arr < n_unique)].tolist()

        # 3. Slivers, and bins beyond b, join their smaller neighbour
        while cuts:
            sizes: NDArray[np.int64] = np.diff(cum_counts[[0] + cuts + [n_unique]])
            i: int = int(np.argmin(sizes))  # The first one when several are as small
            if 2 * sizes[i] >= bin_size and len(sizes) <= b:
                break
            if i == 0 or (i < len(sizes) - 1 and sizes[i + 1] < sizes[i - 1]):
                del cuts[i]  # Joins the next bin
            else:
                del cuts[i - 1]  # Joins the previous bin

        return np.array(cuts, dtype=np.int64)

    @staticmethod
    def tie_safe_bin_edges(unique_probs: NDArray[np.float64], cuts: NDArray[np.int64]) -> NDArray[np.float64]:
        """Bin edges half-way between the distinct scores that each cut separates, so that no score lies on an edge."""
        lower: NDArray[np.float64] = unique_probs[cuts - 1]
        upper: NDArray[np.float64] = unique_probs[cuts]
        inner: NDArray[np.float64] = (lower + upper) / 2
        # Two adjacent floats have no value in between: fall back on the upper one, which np.digitize (left-closed) sends to the upper bin
        inner = np.where((inner > lower) & (inner < upper), inner, upper)
        return np.concatenate([[min(0.0, unique_probs[0])], inner, [max(1.0, unique_probs[-1])]])

    @staticmethod
    def is_monotonic(bin_heights: List[float]) -> bool:
        return all(bin_heights[i] <= bin_heights[i + 1] for i in range(len(bin_heights) - 1))

    @staticmethod
    def is_monotonic_tie_safe(counts: NDArray[np.int64], positives: NDArray[np.float64], robust: bool = False) -> bool:
        """
        Same check as is_monotonic on the bin heights positives / counts. With robust=True, a decrease between
        two bins whose sizes differ by a factor of 2 or more only counts if it survives changing one label in
        the smaller bin: next to a large tied block (for example the block at 0.00 of a one-vs-rest column) a
        small bin that holds one positive less than expected would otherwise stop the sweep. Bins of similar
        size are checked exactly as by is_monotonic.
        """
        c0, c1 = counts[:-1], counts[1:]
        p0, p1 = positives[:-1], positives[1:]
        decrease: NDArray[np.bool_] = p0 * c1 > p1 * c0
        if robust and np.any(decrease):
            unequal: NDArray[np.bool_] = np.maximum(c0, c1) >= 2 * np.minimum(c0, c1)
            # One positive more in the smaller bin on the right, or one less in the smaller bin on the left
            survives: NDArray[np.bool_] = np.where(c0 >= c1, p0 * c1 > (p1 + 1) * c0, (p0 - 1) * c1 > p1 * c0)
            decrease &= ~unequal | survives
        return not np.any(decrease)
