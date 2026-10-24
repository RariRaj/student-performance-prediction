# Prediction Interface Testing

## Test procedure

1. Start the application using:
   `python -m streamlit run app.py`
2. Enter each sample below.
3. Click "Predict result".
4. Record the displayed prediction and estimated Fail probability.
5. Save a screenshot and download the prediction record.

The application loads the saved preprocessing and classification
pipeline. It does not retrain the model when making predictions.

## Sample inputs and results

| Field | Sample 1 | Sample 2 |
|---|---|---|
| Study hours per week | 15 | 30 |
| Attendance (%) | 65 | 95 |
| Previous score | 60 | 90 |
| Sleep hours per day | 7 | 8 |
| Tutoring sessions | 0 | 3 |
| Access to resources | Low | High |
| Predicted result | Fail | Pass |
| Estimated Fail probability | 100.00% | 0.1% |
| Actual result | Unknown | Unknown |

These examples are illustrative inputs, not students with known
outcomes. They demonstrate that the interface accepts inputs and
produces predictions; they do not establish prediction accuracy.

Displayed probabilities are model estimates, not guarantees.

## Input validation

Complete the observed-result column after performing each check.

| Check | Expected behaviour | Observed result |
|---|---|---|
| Attendance above 100 | Input rejected by numeric control | True |
| Negative tutoring count | Input rejected by numeric control | True |
| Study hours 100 and daily sleep hours 12 | Error shown because combined weekly hours exceed 168 | True |
| Resource category selection | Only Low, Medium, or High available | True |

## Evidence files

- interface_sample_1.csv
- interface_sample_2.csv
- figures/interface_sample_1.png
- figures/interface_sample_2.png

Only list evidence files that have actually been saved.