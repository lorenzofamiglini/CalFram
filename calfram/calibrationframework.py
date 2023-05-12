import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .ece import compute_eces
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.metrics import brier_score_loss

def end_points(x,y):
    #check coordinates (0,0) and (1,1)
    points = np.concatenate([x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)], axis = 1)
    # if not np.all(points[0,:] == [0,0]):
    points = np.concatenate([np.array([[0,0]]), points], axis = 0)
    # else:
        # pass
    return points

def add_tilde(pts):
    new_array = []
    for length in range(len(pts)):
        if length == 0:
            new_array.append(pts[length])
        else:
            #new_array.append(pts[length])
            tilde = np.array([pts[length][0],pts[length][0]])
            new_array.append(tilde)
    return np.array(new_array)

def h_triangle(new_pts, tilde):
    height_list = []

    for idx, points in enumerate(zip(new_pts, tilde)):
        if idx == 0:
            pass
        else:
            a = tilde[idx-1]
            b = tilde[idx]
            c = new_pts[idx]
            ab = np.sqrt(((a[0]-b[0])**2)+((a[1]-b[1])**2))
            ac = np.sqrt(((a[0]-c[0])**2)+((a[1]-c[1])**2))
            bc = np.sqrt(((b[0]-c[0])**2)+((b[1]-c[1])**2))
            if ab == 0:
                ab = 0.0000000000001
            sorted_side = np.sort([ab,ac,bc])
            c = sorted_side[0]
            b = sorted_side[1]
            a = sorted_side[2]
            area = 1/4*np.sqrt(((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))))
            h = 2*area/ab
            height_list.append(h)
    return np.array(height_list)

def underbelow_line(pts):
    """
    Se il punto e' sopra o sotto la bisettrice
    """
    sign = []
    for index, elem in enumerate(pts):
        if (index - 1 >= 0):
            v1 = (1 - 0)*(pts[index][1] - 0) - (pts[index][0] - 0)*(1 - 0)
            if v1 > 0:
                sign.append('left')
            elif v1 < 0:
                sign.append('right')
            else:
                sign.append('lie')
    return np.array(sign)

def check_idx(pts):
    """
    Controllo se il punti sono collegati da una retta che interseca la bisettrice
    Inoltre prendo gli indici dei punti da calcolare come punto di interserzione sulla bisettrice
    """
    line = underbelow_line(pts)
    idx_line = []
    for idx in range(len(line)):
        if (idx+1 < len(line) - 1 and idx - 1 >= 0):
            if line[idx] != line[idx+1]:
                idx_line.append(idx)
    return np.array(idx_line)

def find_inters(dir_pts, dir_m): 
    pts_star = []
    for index, elem in enumerate(dir_pts):
        #find q 
        q = -(dir_m[index] * dir_pts[index][0]) + dir_pts[index][1]
        x_star = q/(1-dir_m[index])
        y_star = x_star 
        pts_star.append([x_star, y_star])
    return np.array(pts_star)

def finite_diff(pts):
    """
    Importante la presenza dei due estremi su questi indici 
    pts[0] : (0,0)
    pts[-1] : (1,1)
    """
    deriv = []
    for index, elem in enumerate(pts):
        if (index+1 < len(pts) and index - 1 >= 0):
            num = pts[index+1][1] - pts[index][1]
            den = pts[index+1][0] - pts[index][0]
            dydx = num / den
            deriv.append(dydx)
        else:
            deriv.append(1)
    return np.array(deriv)

def find_points(x,y):
    pts = end_points(x,y)
    m = finite_diff(pts)
    dir_point_idx = check_idx(pts)
    if len(dir_point_idx) != 0:
        if np.all(pts[dir_point_idx[-1],:] == [1,1]):
            dir_point_idx = dir_point_idx[:-2] #tra penultimo punto e l'ultimo conosco gia il punto essendo 1,1 che chiude la funzione 
        dir_m = m[dir_point_idx]
        dir_pts = pts[dir_point_idx,:]
        new_points = find_inters(dir_pts, dir_m)
        dir_point_idx = np.array([i+1 for i in dir_point_idx])
        new_pts = np.insert(pts, dir_point_idx, new_points, axis = 0)
        return new_pts, new_points
    else:
        return pts, np.nan

def split_probabilities(probs, r):
    """
    Args:
        probs: array of probabilities
        r: number of bins
    Returns:
        prob_ranges: list of arrays of probabilities
        bin_edges: list of bin edges
    """

    sorted_probs = np.sort(probs)

    prob_ranges = np.array_split(sorted_probs, r)

    bin_edges = [prob_range[-1] for prob_range in prob_ranges[:-1]] + [prob_ranges[-1][-1]]

    return prob_ranges, bin_edges
    
def compute_bin_heights(data, b):
    bin_edges = [i / b for i in range(b + 1)]
    bin_counts = [0] * b
    bin_heights = [0] * b

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

def compute_equal_mass_bin_heights(data, b):
    data = sorted(data, key=lambda x: x[0])  # Sort data based on scores
    bin_size = len(data) // b
    bin_counts = [0] * b
    bin_heights = [0] * b

    for i in range(b):
        start = i * bin_size
        end = (i + 1) * bin_size if i < b - 1 else len(data)
        bin_data = data[start:end]
        bin_counts[i] = len(bin_data)
        bin_heights[i] = sum(true_label for _, true_label in bin_data) / bin_counts[i]

    return bin_heights
    
def is_monotonic(bin_heights):

    return all(bin_heights[i] <= bin_heights[i + 1] for i in range(len(bin_heights) - 1))


def monotonic_sweep_calibration(data, n):
    b = 2
    while b <= n:
        bin_heights = compute_equal_mass_bin_heights(data, b)
        if not is_monotonic(bin_heights):
            b -= 1
            break
        b += 1
    print('Number of bins: ', b)
    return b

def monotonic_sweep_calibration_multiclass(Y_probs, Y, n_classes):
    if Y_probs.ndim == 1:
        binary_data = [(prob, int(true_label == 1)) for prob, true_label in zip(Y_probs, Y)]
    else:
        binary_data = [(prob[1], int(true_label == 1)) for prob, true_label in zip(Y_probs, Y)]

    b = monotonic_sweep_calibration(binary_data, len(Y_probs))
    return int(b) #int(np.mean(b_per_class))

def binning_schema(prob, Y, method=15, ndim=1, adaptive=False):
    if ndim is not None:
        prob = prob[:, ndim]
    else:
        prob = np.max(prob, axis=-1)

    n_classes = len(np.unique(Y))

    if adaptive:
        b = monotonic_sweep_calibration_multiclass(prob, Y, n_classes)
        b = min(b, len(np.unique(prob)))  # Limit the number of bins to the number of unique data points
    else:
        if not isinstance(method, (int)):
            raise ValueError("Please provide an int object for selecting the number of bins (we implement equal mass approach) or select adaptive = True for monotonic sweep for identifying the right number of bins.")
        b = method

    if prob.ndim == 1:
        binary_data = [(probs, int(true_label == 1)) for probs, true_label in zip(prob, Y)]
    else:
        binary_data = [(probs[1], int(true_label == 1)) for probs, true_label in zip(prob, Y)]
        
    bin_edges = []

    data = sorted(binary_data, key=lambda x: x[0])
    bin_size = len(data) // b

    for i in range(b):
        start = i * bin_size
        end = (i + 1) * bin_size if i < b - 1 else len(data)
        bin_edges.append(data[start][0])

    bin_edges.append(data[-1][0])
    binids = np.digitize(prob, bin_edges) - 1
    relative_freq_bin = np.unique(binids, return_counts=True)[1] / len(prob)

    return {'bins': bin_edges,
            'binids': binids,
            'binfr': relative_freq_bin}

def calibrationcurve(y_true,y_prob,strategy =10, undersampling = False, adaptive = False):
    """
    Calibration Curve for Binary Classification or Multiclass 1 vs All
    Strategy: int, number of bins
    Adaptive: Bool, it True strategy is ignored and calculates automatically the right number of bins. 
    """
    np.random.seed(123)

    labels = np.unique(y_true)
    if undersampling:
        pass

    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}. For Multiclass use 1 vs All Approach."
        )

    bins_dict = binning_schema(y_prob, y_true, method = strategy, ndim = 1, adaptive=adaptive)
    bin_sums = np.bincount(bins_dict['binids'], weights=y_prob[:,1], minlength=len(bins_dict['bins']))
    
    bin_true = np.bincount(bins_dict['binids'], weights=y_true.squeeze(), minlength=len(bins_dict['bins']))
    bin_total = np.bincount(bins_dict['binids'], minlength=len(bins_dict['bins']))
    
    nonzero = bin_total != 0

    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred, bins_dict

def select_probability(y_actual, y_prob, y_pred):
    """
    Choose the probability for the class of interest, for 1 vs All approach
    """
    ohe = OneHotEncoder()
    y_one_hot = ohe.fit_transform(y_actual.reshape(-1, 1) )    
    y_pred_hot = ohe.transform(y_pred.reshape(-1, 1) )
    y_prob_one_hot = y_prob.copy()

    labels = np.unique(y_actual)
    final_dict = {}
    for i in range(len(labels)):
        y_class = y_actual.copy()
        indices_class = np.argwhere(y_class == i)
        indices_other = np.argwhere(y_class != i)

        y_proba_class = np.expand_dims(y_prob[:,i],1)
        y_proba_rest = 1 - y_proba_class
        new_y_proba = np.concatenate([y_proba_rest,y_proba_class], axis=1)
        y_class[indices_class] = 1
        y_class[indices_other] = 0
        
        dict_clss = {'proba': new_y_proba, 'y': y_class,
                     'y_one_hot_nclass': np.array(y_one_hot.todense()), 
                     'y_prob_one_hotnclass': y_prob_one_hot,
                     'y_pred_one_hotnclass': np.array(y_pred_hot.todense())}

        final_dict[str(i)] = dict_clss     

    return final_dict

def reliabilityplot(classes_scores,strategy = 'doane',split = True, undersampling=False):
    marker_list =  ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's']
    conta = 0
    plt.figure(figsize=(10,10))
    for i in classes_scores.keys():
        mark = marker_list[conta]
        conta += 1
        prob_true, prob_pred, bins_dict = calibrationcurve(classes_scores[i]['y'],classes_scores[i]['proba'],strategy=strategy, undersampling=undersampling)

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams['legend.title_fontsize'] = 'xx-small'
        plt.rc('grid', linestyle=":", color='black')
        plt.plot(prob_pred, prob_true, label = f'Class {i}', linestyle='--', markersize=3)
        plt.scatter(prob_pred, prob_true,marker=mark)
        plt.plot([0, 1], [0, 1], linestyle='--', color='black')
        plt.legend(loc='lower right',fancybox=True, shadow=True,ncol=3,fontsize=7)
        plt.gca().set_aspect('equal', adjustable='box')
        if split: 
            plt.show()

def calibrationdiagnosis(classes_scores, strategy = 'doane', undersampling=False):
    measures = {}
    binning_dict = {}
    for i in classes_scores.keys():
        y, x, bins_dict = calibrationcurve(classes_scores[i]['y'],classes_scores[i]['proba'], strategy = strategy,
                                            undersampling=undersampling)
        new_pts = end_points(x, y)

        tilde = add_tilde(new_pts)
        
        pts_distance = h_triangle(new_pts, tilde)
        max_pts = new_pts.copy()

        for pt in range(len(new_pts)):
            if pt != 0:
                if new_pts[pt][0] <= 0.5:
                    max_pts[pt] = [new_pts[pt][0], 1]
                else:
                    max_pts[pt] = [new_pts[pt][0], 0]

        max_height_distance = h_triangle(max_pts, tilde)
        pts_distance_norm = pts_distance / max_height_distance
        where_are = underbelow_line(new_pts)

        mask_left = where_are == 'left'
        mask_right = where_are == 'right'
    
        if np.all((len(where_are) == 1) & ((np.unique(where_are)) == 'lie')): 
            dict_msr = {'ece_acc': np.nan, 'ece_fp': np.nan, 'ec_g': np.nan, 'ec_under': np.nan, 'under_fr': np.nan,
                        'ec_over': np.nan, 'over_fr': np.nan, 'ec_underconf': np.nan, 'ec_overconf': np.nan,
                        'ec_dir': np.nan, 'over_pts': np.nan, 'under_pts': np.nan, 'ec_l_all': np.nan, 'where': np.nan,
                        'relative-freq': np.nan, 'x': np.nan, 'y': np.nan}
        else:
            up_dist = pts_distance_norm[mask_left]
            below_dist = pts_distance_norm[mask_right]
            up_pts = new_pts[1:][mask_left]
            below_pts = new_pts[1:][mask_right]
            up_weight = bins_dict['binfr'][mask_left]
            below_weight = bins_dict['binfr'][mask_right]

            #metrics
            fcc_g = 1 - np.average(pts_distance_norm, weights=bins_dict['binfr'])
            if len(up_weight) != 0:
                up_weight1 = up_weight / np.sum(up_weight)
                fcc_underconf = 1-np.average(up_dist, weights=up_weight1)
            else:
                fcc_underconf = np.nan
                up_weight = None
            if len(below_weight) != 0:
                below_weight1 = below_weight / np.sum(below_weight)
                fcc_overconf = 1-np.average(below_dist, weights=below_weight1)
            else:
                fcc_overconf = np.nan
                below_weight = None
            if up_weight is not None and below_weight is not None:    
                fcc_dir = np.average(below_dist, weights=below_weight) - np.average(up_dist, weights=up_weight)
            elif up_weight is None and below_weight is not None:
                fcc_dir = np.average(below_dist, weights=below_weight)
            elif up_weight is not None and below_weight is None:
                fcc_dir = -np.average(up_dist, weights=up_weight)
            else:
                fcc_dir = np.nan

                
            ece = compute_eces(classes_scores[i]['y_one_hot_nclass'], classes_scores[i]['y_prob_one_hotnclass'],
                            classes_scores[i]['y_pred_one_hotnclass'], bins_dict['binids'],
                            bins_dict['bins'],'fp',int(i))
            ece_acc = compute_eces(classes_scores[i]['y_one_hot_nclass'], classes_scores[i]['y_prob_one_hotnclass'],
                            classes_scores[i]['y_pred_one_hotnclass'], bins_dict['binids'],  bins_dict['bins'],'acc',int(i))
            brierloss = brier_score_loss(classes_scores[i]['y'], classes_scores[i]['proba'][:,1])
        
            dict_msr = {'ece_acc': ece_acc, 'ece_fp':ece, 'ec_g': fcc_g, 'ec_under': 1-up_dist,'under_fr': up_weight, 'ec_over': 1-below_dist, 
                    'over_fr': below_weight,'ec_underconf': fcc_underconf,'ec_overconf': fcc_overconf, 
                    'ec_dir': fcc_dir, 'brier_loss': brierloss, 'over_pts': below_pts, 'under_pts': up_pts, 
                    'ec_l_all': 1-pts_distance_norm, 'where': where_are,
                    'relative-freq': bins_dict['binfr'], 'x': x, 'y' :y
                    }


        measures['{}'.format(i)] = dict_msr
        binning_dict['{}'.format(i)] = bins_dict

            # ece, mce = mcece(prob, n_bins, y_test, predictions)

    return measures, binning_dict


def classwise_calibration(measures):
    classes_global = np.mean([measures[key]['ec_g'] for key in measures.keys()]).round(3)
    classes_direction = np.mean([measures[key]['ec_dir'] for key in measures.keys()]).round(3)
    classes_underconf = np.nanmean([measures[key]['ec_underconf'] for key in measures.keys()]).round(3)
    classes_overconf = np.nanmean([measures[key]['ec_overconf'] for key in measures.keys()]).round(3)
    classes_ece = np.mean([measures[key]['ece_fp'] for key in measures.keys()]).round(3)
    classes_ece_acc = np.mean([measures[key]['ece_acc'] for key in measures.keys()]).round(3)
    classes_brier = np.mean([measures[key]['brier_loss'] for key in measures.keys()]).round(3)/2
    return {'ec_g': classes_global, 'ec_dir': classes_direction, 'ece_freq': classes_ece, 'ece_acc': classes_ece_acc,
            'ec_underconf': classes_underconf, 'ec_overconf': classes_overconf, 'brierloss': classes_brier} #'1-ece': classes_ece
