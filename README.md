# CalFram: A Comprehensive Framework for Calibration Assessment

## Introduction

Calibration is a multidimensional concept that is essential for assessing machine learning models. It helps in understanding a model's global calibration performance, identifying miscalibrated regions of the probability space, and determining the level of overconfidence or underconfidence of a model. Therefore, multi-dimensionality is critical to gain a thorough understanding of a machine learning model's performance and limitations.

To address the above concerns, we have developed a comprehensive framework for assessing calibration for binary and multiclass classification models.

Our framework offers various calibration metrics for a holistic evaluation of your model's calibration. It includes:

- Global Measures: For an overall assessment of the model's calibration.
- Local Measures: To provide detailed insight into the model's performance in specific regions of the input space.
- Balance Measures: To quantify how much the model is overconfident or underconfident.
- Overconfident and Underconfident Area Metrics: To highlight parts of the input space where the model is especially overconfident or underconfident.

Together, these measures provide a complete understanding of your model's calibration and help to make targeted modifications to improve the model. Our framework is open-sourced and available on GitHub for the community. It works directly with any model's outputs, making it agnostic to any Machine Learning and Deep Learning libraries.

Moreover, Brier Score Loss and ECE accuracy based and ECE frequency based are reported. 

## Example
```python
from calibrationframework import select_probability, calibrationdiagnosis

# Your model predictions and actual values
y_pred = ... (n x 1)
y_true = ...  (n x 1)
y_prob = ... (n x c), where c is the number of classes 

classes_scores = select_probability(y_test, y_prob, y_pred)

#compute all the metrics based on 15 bins with equal-width
results, _ = calibrationdiagnosis(classes_scores, strategy = 15, adaptive = False)
#or, compute all the metrics based on automatic monothonic sweep method for identifying the right number of bins 
results, _ = calibrationdiagnosis(classes_scores, adaptive = True)
```

## Contributing
We welcome contributions to this project. Please feel free to open issues or submit pull requests.

## License
This project is open source under the MIT license. See the LICENSE file for more information.

