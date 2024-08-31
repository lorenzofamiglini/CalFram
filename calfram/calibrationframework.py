import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from numpy.typing import NDArray
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

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

    def binning_schema(self, prob: NDArray[np.float64], Y: NDArray[np.int64], method: Union[int, str] = 15, ndim: Union[int, None] = 1, adaptive: bool = False) -> Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]:
        if ndim is not None:
            prob = prob[:, ndim]
        else:
            prob = np.max(prob, axis=-1)

        n_classes: int = len(np.unique(Y))

        if adaptive:
            b: int = self.monotonic_sweep_calibration_multiclass(prob, Y, n_classes)
            b = min(b, len(np.unique(prob)))
        else:
            if not isinstance(method, (int, str)):
                raise ValueError("Please provide an int or str object for selecting the number of bins or select adaptive = True for monotonic sweep.")
            b = method

        if prob.ndim == 1:
            binary_data: List[Tuple[float, int]] = [(probs, int(true_label == 1)) for probs, true_label in zip(prob, Y)]
        else:
            binary_data = [(probs[1], int(true_label == 1)) for probs, true_label in zip(prob, Y)]
        
        bin_edges: NDArray[np.float64] = np.array(sorted(set([data[0] for data in binary_data])))
        binids: NDArray[np.int64] = np.digitize(prob, bin_edges) - 1
        relative_freq_bin: NDArray[np.float64] = np.unique(binids, return_counts=True)[1] / len(prob)

        return {
            'bins': bin_edges,
            'binids': binids,
            'binfr': relative_freq_bin
        }

    def calibrationcurve(self, y_true: NDArray[np.int64], y_prob: NDArray[np.float64], strategy: Union[int, str] = 10, undersampling: bool = False, adaptive: bool = False) -> Tuple[NDArray[np.float64], NDArray[np.float64], Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]]:
        np.random.seed(123)

        labels: NDArray[np.int64] = np.unique(y_true)
        if len(labels) > 2:
            raise ValueError(f"Only binary classification is supported. Provided labels {labels}. For Multiclass use 1 vs All Approach.")

        bins_dict: Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]] = self.binning_schema(y_prob, y_true, method=strategy, ndim=1, adaptive=adaptive)
        bin_sums: NDArray[np.float64] = np.bincount(bins_dict['binids'], weights=y_prob[:,1], minlength=len(bins_dict['bins']))
        
        bin_true: NDArray[np.float64] = np.bincount(bins_dict['binids'], weights=y_true.squeeze(), minlength=len(bins_dict['bins']))
        bin_total: NDArray[np.int64] = np.bincount(bins_dict['binids'], minlength=len(bins_dict['bins']))
        
        nonzero: NDArray[np.bool_] = bin_total != 0

        prob_true: NDArray[np.float64] = bin_true[nonzero] / bin_total[nonzero]
        prob_pred: NDArray[np.float64] = bin_sums[nonzero] / bin_total[nonzero]

        return prob_true, prob_pred, bins_dict

    def select_probability(self, y_actual: NDArray[np.int64], y_prob: NDArray[np.float64], y_pred: NDArray[np.int64]) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        y_one_hot: NDArray[np.float64] = self.ohe.fit_transform(y_actual.reshape(-1, 1))
        y_pred_hot: NDArray[np.float64] = self.ohe.transform(y_pred.reshape(-1, 1))
        y_prob_one_hot: NDArray[np.float64] = y_prob.copy()

        labels: NDArray[np.int64] = np.unique(y_actual)
        final_dict: Dict[str, Dict[str, NDArray[np.float64]]] = {}
        for i in range(len(labels)):
            y_class: NDArray[np.int64] = y_actual.copy()
            indices_class: NDArray[np.int64] = np.argwhere(y_class == i)
            indices_other: NDArray[np.int64] = np.argwhere(y_class != i)

            y_proba_class: NDArray[np.float64] = np.expand_dims(y_prob[:,i], 1)
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

    def calibrationdiagnosis(self, classes_scores: Dict[str, Dict[str, NDArray[np.float64]]], strategy: Union[int, str] = 'doane', undersampling: bool = False, adaptive: bool =False) -> Tuple[Dict[str, Dict[str, Union[float, NDArray[np.float64]]]], Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]]]:
        measures: Dict[str, Dict[str, Union[float, NDArray[np.float64]]]] = {}
        binning_dict: Dict[str, Dict[str, Union[NDArray[np.float64], NDArray[np.int64], float]]] = {}
        for i in classes_scores.keys():
            y, x, bins_dict = self.calibrationcurve(classes_scores[i]['y'], classes_scores[i]['proba'], strategy=strategy, undersampling=undersampling, adaptive=adaptive)
            new_pts: NDArray[np.float64] = self.end_points(x, y)

            tilde: NDArray[np.float64] = self.add_tilde(new_pts)
            
            pts_distance: NDArray[np.float64] = self.h_triangle(new_pts, tilde)
            max_pts: NDArray[np.float64] = new_pts.copy()

            for pt in range(1, len(new_pts)):
                if new_pts[pt][0] <= 0.5:
                    max_pts[pt] = [new_pts[pt][0], 1]
                else:
                    max_pts[pt] = [new_pts[pt][0], 0]

            max_height_distance: NDArray[np.float64] = self.h_triangle(max_pts, tilde)
            pts_distance_norm: NDArray[np.float64] = pts_distance / max_height_distance
            where_are: List[str] = self.underbelow_line(new_pts[1:])  # Exclude the first point

            mask_left: NDArray[np.bool_] = np.array([w == 'left' for w in where_are])
            mask_right: NDArray[np.bool_] = np.array([w == 'right' for w in where_are])
        
            if len(where_are) == 0 or np.all(np.array(where_are) == 'lie'): 
                dict_msr: Dict[str, Union[float, NDArray[np.float64]]] = {
                    'ece_acc': np.nan, 'ece_fp': np.nan, 'ec_g': np.nan, 'ec_under': np.nan, 'under_fr': np.nan,
                    'ec_over': np.nan, 'over_fr': np.nan, 'ec_underconf': np.nan, 'ec_overconf': np.nan,
                    'ec_dir': np.nan, 'over_pts': np.nan, 'under_pts': np.nan, 'ec_l_all': np.nan, 'where': np.nan,
                    'relative-freq': np.nan, 'x': np.nan, 'y': np.nan
                }
            else:
                up_dist: NDArray[np.float64] = pts_distance_norm[mask_left]
                below_dist: NDArray[np.float64] = pts_distance_norm[mask_right]
                up_pts: NDArray[np.float64] = new_pts[1:][mask_left]
                below_pts: NDArray[np.float64] = new_pts[1:][mask_right]
                up_weight: NDArray[np.float64] = bins_dict['binfr'][mask_left]
                below_weight: NDArray[np.float64] = bins_dict['binfr'][mask_right]

                fcc_g: float = 1 - np.average(pts_distance_norm, weights=bins_dict['binfr'])
                if len(up_weight) != 0:
                    up_weight1: NDArray[np.float64] = up_weight / np.sum(up_weight)
                    fcc_underconf: float = 1 - np.average(up_dist, weights=up_weight1)
                else:
                    fcc_underconf = np.nan
                    up_weight = np.array([])
                if len(below_weight) != 0:
                    below_weight1: NDArray[np.float64] = below_weight / np.sum(below_weight)
                    fcc_overconf: float = 1 - np.average(below_dist, weights=below_weight1)
                else:
                    fcc_overconf = np.nan
                    below_weight = np.array([])
                if len(up_weight) > 0 and len(below_weight) > 0:    
                    fcc_dir: float = np.average(below_dist, weights=below_weight) - np.average(up_dist, weights=up_weight)
                elif len(up_weight) == 0 and len(below_weight) > 0:
                    fcc_dir = np.average(below_dist, weights=below_weight)
                elif len(up_weight) > 0 and len(below_weight) == 0:
                    fcc_dir = -np.average(up_dist, weights=up_weight)
                else:
                    fcc_dir = np.nan

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

            measures[str(i)] = dict_msr
            binning_dict[str(i)] = bins_dict

        return measures, binning_dict

    def compute_eces(self, y: NDArray[np.float64], prob: NDArray[np.float64], y_pred: NDArray[np.float64], 
                     binids: NDArray[np.int64], bins: NDArray[np.float64], groupby: str = 'fp', ndim: Optional[int] = None) -> float:
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

    def classwise_calibration(self, measures: Dict[str, Dict[str, Union[float, NDArray[np.float64]]]]) -> Dict[str, float]:
        classes_global: float = np.mean([measures[key]['ec_g'] for key in measures.keys()]).round(3)
        classes_direction: float = np.mean([measures[key]['ec_dir'] for key in measures.keys()]).round(3)
        classes_underconf: float = np.nanmean([measures[key]['ec_underconf'] for key in measures.keys()]).round(3)
        classes_overconf: float = np.nanmean([measures[key]['ec_overconf'] for key in measures.keys()]).round(3)
        classes_ece: float = np.mean([measures[key]['ece_fp'] for key in measures.keys()]).round(3)
        classes_ece_acc: float = np.mean([measures[key]['ece_acc'] for key in measures.keys()]).round(3)
        classes_brier: float = np.mean([measures[key]['brier_loss'] for key in measures.keys()]).round(3) / 2
        return {
            'ec_g': classes_global,
            'ec_dir': classes_direction,
            'ece_freq': classes_ece,
            'ece_acc': classes_ece_acc,
            'ec_underconf': classes_underconf,
            'ec_overconf': classes_overconf,
            'brierloss': classes_brier
        }

    def reliabilityplot(self, classes_scores: Dict[str, Dict[str, NDArray[np.float64]]], strategy: Union[int, str] = 'doane', split: bool = True, undersampling: bool = False) -> None:

        marker_list: List[str] = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's']
        plt.figure(figsize=(10, 10))
        for idx, (i, class_score) in enumerate(classes_scores.items()):
            prob_true, prob_pred, _ = self.calibrationcurve(class_score['y'], class_score['proba'], strategy=strategy, undersampling=undersampling)

            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['legend.title_fontsize'] = 'xx-small'
            plt.rc('grid', linestyle=":", color='black')
            plt.plot(prob_pred, prob_true, label=f'Class {i}', linestyle='--', markersize=3)
            plt.scatter(prob_pred, prob_true, marker=marker_list[idx % len(marker_list)])
        
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
    def underbelow_line(pts: NDArray[np.float64]) -> List[str]:
        return ['lie' if idx == 0 else 
                'left' if (1 - 0) * (pt[1] - 0) - (pt[0] - 0) * (1 - 0) > 0 else 
                'right' if (1 - 0) * (pt[1] - 0) - (pt[0] - 0) * (1 - 0) < 0 else 
                'lie' for idx, pt in enumerate(pts)]

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
        data = sorted(data, key=lambda x: x[0])
        bin_size: int = len(data) // b
        bin_heights: List[float] = [
            sum(true_label for _, true_label in data[i*bin_size:(i+1)*bin_size]) / bin_size
            for i in range(b)
        ]
        return bin_heights

    @staticmethod
    def is_monotonic(bin_heights: List[float]) -> bool:
        return all(bin_heights[i] <= bin_heights[i + 1] for i in range(len(bin_heights) - 1))
