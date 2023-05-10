import numpy as np 
import ipdb 

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
    
def sampling_schema(prob, method = 'auto', ndim = 1, adaptive = False):
    """
    ndim: which column should be selected for the proba vector: 
          0 prob for class 0, 1 prob for class 1, etc..; None max proba
    """
    if ndim is not None: #and ndim != 'classwise':
        prob = prob[:,ndim]
    else:
        prob = np.max(prob, axis = -1)
    
    b_range = np.histogram_bin_edges(prob, bins=method, weights=None)
    binids = np.digitize(prob, b_range) - 1
    ranges = np.nan
    if adaptive:
        ranges = b_range.shape[0]
        _, b_range = split_probabilities(prob, ranges)
        binids = np.digitize(prob, b_range) 

    relative_freq_bin = np.unique(binids, return_counts=True)[1] / len(prob)
    return {'bins': b_range,
            'binids': binids,
            'binfr': relative_freq_bin,
            'ranges': ranges}

def compute_eces(y, prob,y_pred, binids, bins, groupby='fp', ndim = None):
    '''
    Args:
        y: ground truth
        prob: probabilities
        y_pred: predicted labels
        binids: bin ids
        bins: bin edges
        groupby: 'fp' or 'acc'
        ndim: which column should be selected for the proba vector:
                0 prob for class 0, 1 prob for class 1, None max proba
    '''

    if ndim is not None:
        prob = prob[:,ndim].copy()
        y = y[:,ndim].copy()
        y_pred = y_pred[:,ndim].copy()
    else:
        prob = np.max(prob, axis = 1)
        y = np.argmax(y, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)
        
    bin_total = np.bincount(binids, minlength=len(bins))
    nonzero = bin_total != 0

    if groupby == 'fp':
        bin_true = np.bincount(binids, weights=y, minlength=len(bins))
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        confscore_bins = np.bincount(binids, weights=prob.squeeze())[nonzero]/bin_total[nonzero]       
    else:       
        prob_true = np.bincount(binids, weights=(y == y_pred), minlength=len(bins))
        confscore_bins = np.bincount(binids, weights=prob, minlength=len(bins))
        confscore_bins = confscore_bins[nonzero] / bin_total[nonzero]
        prob_true = prob_true[nonzero] / bin_total[nonzero]
        if ndim is not None:
            mask = confscore_bins < 0.5
            confscore_bins[mask] = 1 - confscore_bins[mask]

    loce_w = bin_total[nonzero]/len(y)
    

    lece = np.abs(prob_true - confscore_bins)
    ece = np.sum(loce_w*lece)
    return ece

def class_wise_ece(y_one_hot, probabilities,pred_one_hot, method='auto', groupby = 'acc', adaptive = False):
    '''Compute the class-wise ECE for a given set of predictions.
    Args:
        y_one_hot: ground truth labels in one-hot encoding1000,
        method: binning method
        groupby: 'acc' or 'fp'
    Returns:    
        cw_ece: class-wise ECE
    '''
    k_ece = []
    n_classes = y_one_hot.shape[1]
    for i in range(n_classes):
        bins_dct = sampling_schema(probabilities,  method = method, ndim = i, adaptive = adaptive)
        class_ece = compute_eces(y_one_hot, probabilities,pred_one_hot, bins_dct['binids'], 
                                 bins_dct['bins'], groupby=groupby, ndim = i)
        k_ece.append(class_ece)
    k_ece = np.array(k_ece)
    cw_ece = k_ece.mean()

    return cw_ece