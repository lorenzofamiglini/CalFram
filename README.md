---
<h1 align="center">CalFram: A Comprehensive Framework for Calibration Assessment</h1>
<p align="center">
  <img src="reliabilitydiag.png" alt="Reliability Diagram" width="40%"/>
</p>

## Introduction

Calibration is a multidimensional concept essential for assessing machine learning models. It helps understand a model's global calibration performance, identify miscalibrated regions of the probability space, and determine the level of overconfidence or underconfidence of a model. Therefore, multi-dimensionality is critical to gain a thorough understanding of a machine learning model's performance and limitations.

To address the above concerns, we developed CalFram - a comprehensive framework for assessing calibration for binary and multiclass classification models. This framework relies on the Estimated Calibration Index (ECI). 

The higher the ECI, the better the calibration.

Our framework offers various calibration metrics for a holistic evaluation of your model's calibration. It works directly with numpy arrays, making it model-agnostic. These metrics include:

- **Global Measures (ECI<sub>g</sub>)**: For an overall assessment of the model's calibration. Bounds [0,1], 0 is totally non-calibrated, 1 is perfectly calibrated.
- **Local Measures (ECI<sub>l</sub>)**: To provide detailed insight into the model's performance in specific regions of the input space. The bounds are the same as the global measure.
- **Balance Measures (ECI<sub>b</sub>)**: To quantify how much the model is overconfident or underconfident. Bounds [-1, 1], -1 is totally underconfident, 1 is totally overconfident, and 0 is the trade-off. 
- **Overconfident and Underconfident Area Metrics (ECI<sub>over</sub>, ECI<sub>under</sub>)**: To highlight parts of the input space where the model is especially overconfident or underconfident. The bounds are the same as the global measure.
- **ECE Accuracy based formulation**
- **ECE Frequency based formulation**
- **Brier Score Loss** (Note: for both binary and multiclass, the brier score loss is bounded in [0,1]).

Together, these measures provide a complete understanding of your model's calibration and help to make targeted modifications to improve the model. Our framework is open-sourced and available on GitHub for the community. It works directly with any model's outputs, making it agnostic to any Machine Learning and Deep Learning libraries. 

In addition, we report the Brier Score Loss, ECE accuracy-based, and ECE frequency-based metrics.

## Installation

```bash
pip install calibrationframework
```

## Example
```python
from calfram.calibrationframework import select_probability, calibrationdiagnosis

# Your model predictions and actual values
y_pred = ... # shape: (n, 1)
y_true = ... # shape: (n, 1)
y_prob = ... # shape: (n, c), where c is the number of classes 

classes_scores = select_probability(y_test, y_prob, y_pred)
classes_scores = {
    'class_0': {
        'proba': np.array([]),  # The class probabilities for the given class as a 2D numpy array
        'y': np.array([]),  # The true labels for the given class as a 1D numpy array
        'y_one_hot_nclass': np.array([]),  # The true labels in one-hot encoding format as a 2D numpy array
        'y_prob_one_hotnclass': np.array([]),  # The predicted probabilities in one-hot encoding format as a 2D numpy array
        'y_pred_one_hotnclass': np.array([]),  # The predicted labels in one-hot encoding format as a 2D numpy array
    },
    'class_1': {
        'proba': np.array([]),  # The class probabilities for the given class as a 2D numpy array
        'y': np.array([]),  # The true labels for the given class as a 1D numpy array
        'y_one_hot_nclass': np.array([]),  # The true labels in one-hot encoding format as a 2D numpy array
        'y_prob_one_hotnclass': np.array([]),  # The predicted probabilities in one-hot encoding format as a 2D numpy array
        'y_pred_one_hotnclass': np.array([]),  # The predicted labels in one-hot encoding format as a 2D numpy array
    },
    # ...
    # The same keys and subkeys would be repeated for each class
}
```

Once the object classes_scores is created: 
```python
# Compute all the metrics based on 15 bins with equal-width
results, _ = calibrationdiagnosis(classes_scores, strategy = 15, adaptive = False)
# Or, compute all the metrics based on automatic monothonic sweep method for identifying the right number of bins 
results, _ = calibrationdiagnosis(classes_scores, adaptive = True)

results = {
    'class_0': { 
        'ece_acc': np.array([]),  # Expected Calibration Error for accuracy for class '0'
        'ece_fp': np.array([]),  # Expected Calibration Error for freq positives for class '0'
        'ec_g': np.array([]),  # A measure of global Estimated Calibration Index for class '0'
        'ec_under': np.array([]),  # Estimated Calibration Index for under-confident predictions for class '0'
        'under_fr': np.array([]),  # Relative frequency of under-confident predictions for class '0'
        'ec_over': np.array([]),  #  Estimated Calibration Index for over-confident predictions for class '0'
        'over_fr': np.array([]),  # Relative frequency of over-confident predictions for class '0'
        'ec_underconf': np.array([]),  # A measure of under-confidence across all predictions for class '0'
        'ec_overconf': np.array([]),  # A measure of over-confidence across all predictions for class '0'
        'ec_dir': np.array([]),  # A measure of the general direction of miscalibration for class '0'
        'brier_loss': np.array([]),  # Brier score loss for class '0'
        'over_pts': np.array([]),  # Points that represent over-confident predictions for class '0'
        'under_pts': np.array([]),  # Points that represent under-confident predictions for class '0'
        'ec_l_all': np.array([]),  # All local Estimated Calibration measures for class '0'
        'where': np.array([]),  # An array indicating where each bin falls for class '0'
        'relative-freq': np.array([]),  # The relative frequencies of the samples falling into each bin for class '0'
        'x': np.array([]),  # The mean predicted confidence of each bin for class '0'
        'y': np.array([]),  # The estimated probability or actual accuracy of each bin for class '0'
    },
    'class_1': {
        'ece_acc': np.array([]),  # Expected Calibration Error for accuracy for class '1'
        # ... Same as above, but for class '1'
    },
    # ... The same keys would be repeated for each class
}

```

## Contributing
We welcome contributions to this project. Please feel free to open issues or submit pull requests.

## License
This project is open source and licensed under the MIT license. See the LICENSE file for more information.
