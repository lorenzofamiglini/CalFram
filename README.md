---
<h1 align="center">CalFram: A Comprehensive Framework for Calibration Assessment</h1>
<p align="center">
  <img src="reliabilitydiag.png" alt="Reliability Diagram" width="60%"/>
</p>

## Introduction

Calibration is a multidimensional concept essential for assessing machine learning models. It helps understand a model's global calibration performance, identify miscalibrated regions of the probability space, and determine the level of overconfidence or underconfidence of a model. Therefore, multi-dimensionality is critical to gain a thorough understanding of a machine learning model's performance and limitations.

To address the above concerns, we developed CalFram - a comprehensive framework for assessing calibration for binary and multiclass classification models. This framework relies on the Estimated Calibration Index (ECI). 

The higher the ECI, the better the calibration.

Our framework offers various calibration metrics for a holistic evaluation of your model's calibration. It works directly with numpy arrays, making it model-agnostic. These metrics include:

- **Global Measures (ECI$_g$)**: For an overall assessment of the model's calibration. Bounds [0,1], 0 is totally non-calibrated, 1 is perfectly calibrated.
- **Local Measures (ECI$_l$)**: To provide detailed insight into the model's performance in specific regions of the input space. The bounds are the same as the global measure.
- **Balance Measures (ECI$_b$)**: To quantify how much the model is overconfident or underconfident. Bounds [-1, 1], -1 is totally underconfident, 1 is totally overconfident, and 0 is the trade-off. 
- **Overconfident and Underconfident Area Metrics (ECI_${over}$, ECI_${under}$)**: To highlight parts of the input space where the model is especially overconfident or underconfident. The bounds are the same as the global measure.
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

# Compute all the metrics based on 15 bins with equal-width
results, _ = calibrationdiagnosis(classes_scores, strategy = 15, adaptive = False)
# Or, compute all the metrics based on automatic monothonic sweep method for identifying the right number of bins 
results, _ = calibrationdiagnosis(classes_scores, adaptive = True)
```

## Contributing
We welcome contributions to this project. Please feel free to open issues or submit pull requests.

## License
This project is open source and licensed under the MIT license. See the LICENSE file for more information.
