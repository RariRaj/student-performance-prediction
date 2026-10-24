# Student Performance Prediction

A binary classification demonstration developed for the Verveox
AI/ML Trainee assessment.

The project compares Logistic Regression and K-Nearest Neighbors
(KNN) to predict a derived Pass/Fail result from study-related and
academic indicators. A Streamlit application loads the selected
saved pipeline and predicts results for new inputs.

This project is a learning exercise. Predictions must not be the
sole basis for decisions about students.

## Dataset

Source: Student Performance Factors by lainguyn123 on Kaggle.

https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Original file: `StudentPerformanceFactors.csv`

The downloaded dataset contains 6,607 rows and 20 columns.
Its representativeness of real school populations has not been
independently verified.

### Download instructions

1. Open the dataset link and download the dataset.
2. Extract `StudentPerformanceFactors.csv`.
3. Place it at:

   `data/raw/StudentPerformanceFactors.csv`

The application can run using the supplied saved model without
the dataset. The dataset is required to rerun the analysis notebook.

## Input features

| Feature | Description |
|---|---|
| Hours_Studied | Study hours per week |
| Attendance | Attendance percentage |
| Previous_Scores | Previous academic score |
| Sleep_Hours | Sleep hours per day |
| Tutoring_Sessions | Number of tutoring sessions |
| Access_to_Resources | Low, Medium, or High |

Assignment marks were unavailable and were omitted rather than
invented. Tutoring sessions and resource access were included as
additional study-related indicators.

## Target definition

The original dataset contains an Exam_Score column rather than
the binary target used in this project.

Labels were derived using an assumed threshold:

- Pass (0): Exam_Score >= 70
- Fail (1): Exam_Score < 70

Fail is the positive class for binary precision, recall, and F1.

The threshold of 70 is a demonstration assumption, not a verified
school policy. Exam_Score is excluded from model inputs because
it directly determines the target.

## Data quality and preprocessing

- No exact duplicate rows were found.
- Missing values were present in Teacher_Quality,
  Parental_Education_Level, and Distance_from_Home.
  These columns were not selected as model inputs.
- The six selected input features contained no missing values,
  so no imputation was required.
- One row had Exam_Score = 101 and was excluded under the
  documented assumption that scores must lie between 0 and 100.
- The resulting modelling dataset contains 6,606 rows.
- Valid but unusual input values were retained rather than
  automatically removed as outliers.
- Numeric features were scaled using StandardScaler.
- Access_to_Resources was encoded using OneHotEncoder.

Preprocessing and classification were combined in Scikit-learn
pipelines. Preprocessing was fitted separately within each
cross-validation training fold, then on the full training set
when the selected configuration was refitted.

## Exploratory data analysis

The notebook includes class counts, numerical distributions,
study hours by result, resource-access comparisons, and a
correlation heatmap.

Training-data patterns included:

- Students labelled Pass generally had higher study hours and
  attendance than students labelled Fail.
- Higher resource-access categories had a larger proportion of
  Pass labels, although the groups overlapped.

These are observed associations, not evidence of causation.

## Training and model selection

- Train/test split: 80% / 20%.
- Random seed: 42.
- Stratification preserved the approximate class proportions.
- Training rows: 5,284.
- Test rows: 1,322.
- Five-fold stratified cross-validation was performed on training data.
- A majority-class DummyClassifier provided a baseline.
- GridSearchCV tuned both pipelines using training data only.
- Macro F1 was used for selecting configurations and comparing models.

### Best configurations

| Model | Best parameters | Training CV macro F1 |
|---|---|---:|
| Logistic Regression | C=10.0 | 0.8685 |
| KNN | n_neighbors=21, weights="distance" | 0.8413 |

Logistic Regression was selected based on its higher training
cross-validation macro F1.

### Development history

Baseline test results were inspected during an earlier development
iteration, before the training-only tuning experiment. The final
notebook presents training, tuning, and testing in that order.

Test rows were excluded from fitting and hyperparameter searches,
but the reused test set should not be described as never previously
inspected. An independent external dataset would provide a stronger
final evaluation.

## Test results

| Model | Accuracy | Precision (Fail) | Recall (Fail) | F1 (Fail) | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9130 | 0.9332 | 0.9529 | 0.9429 | 0.8800 |
| KNN | 0.9002 | 0.9069 | 0.9669 | 0.9359 | 0.8549 |

### Confusion matrix counts

Fail is the positive class.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 257 | 68 | 47 | 950 |
| KNN | 226 | 99 | 33 | 964 |

- False positive: actual Pass, predicted Fail; a false support alert.
- False negative: actual Fail, predicted Pass; a potentially missed
  opportunity to offer support.

KNN missed 14 fewer Fail cases but raised 31 more false alarms.
Logistic Regression provided higher macro F1 and fewer total errors.

The test set contains approximately 75.4% Fail labels. A model
predicting Fail for every student would achieve approximately
75.4% accuracy while identifying no Pass students. This is why
accuracy was considered alongside class-specific metrics,
macro F1, and confusion matrices.

## Prerequisites

- Python 3.11
- Windows CMD for the commands below
- VS Code with the Python and Jupyter extensions to run the notebook

## Installation

Open CMD in the project folder:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use the recorded package versions when loading the saved pipeline,
particularly the Scikit-learn version.

## Run the notebook

1. Download and place the dataset as described above.
2. Open the project folder in VS Code.
3. Open `notebooks/analysis.ipynb`.
4. Select the Python interpreter from `.venv` as the notebook kernel.
5. Restart the kernel and select Run All.

The notebook performs data checks, EDA, cross-validation, tuning,
test evaluation, and model saving.

## Run the prediction interface

From the project folder with `.venv` activated:

```cmd
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

Enter all six features and click **Predict result**.
The app displays Pass or Fail and an estimated Fail probability.

The application loads:

- `models/student_performance_pipeline.joblib`
- `models/model_metadata.json`

It does not retrain the model for each prediction.

If these files are missing, run the notebook's model-saving cells.

## Interface validation and testing

The interface includes numeric bounds, integer tutoring counts,
a resource-category dropdown, and a check that combined weekly
study and sleep hours do not exceed 168.

Two illustrative examples were tested:

| Sample | Hours | Attendance | Previous score | Sleep | Tutoring | Resources | Prediction | Displayed Fail probability |
|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 15 | 65 | 60 | 7 | 0 | Low | Fail | 100.0% |
| 2 | 30 | 95 | 90 | 8 | 3 | High | Pass | 0.1% |

The actual outcomes are unknown. These tests demonstrate that the
interface produces outputs, not that these predictions are correct.

Probabilities are rounded model estimates, not guarantees.
Their calibration has not been evaluated.

Testing details and screenshots are recorded in
`reports/interface_testing.md` and `reports/figures/`.

## Main project files

- `notebooks/analysis.ipynb`: analysis and model development
- `app.py`: Streamlit prediction interface
- `models/student_performance_pipeline.joblib`: fitted pipeline
- `models/model_metadata.json`: feature and label information
- `reports/`: metrics, interpretation, and interface testing evidence
- `reports/project_report.md`: project methodology, results, and limitations
- `reports/figures/`: EDA charts, confusion matrices, and screenshots
- `requirements.txt`: environment dependencies
- `README.md`: setup and project overview

## Limitations and future improvements

- Pass/Fail labels depend on an assumed score threshold.
- Class imbalance can make accuracy misleading.
- The evaluation uses one dataset and has no external validation.
- Predicted probabilities have not been calibrated.
- Plausible inputs may still differ substantially from training data.

Future work could evaluate the model on independently collected
data with verified outcomes, assess probability calibration using
training-side validation, and investigate performance across
different student populations.

## AI assistance and personal contribution

ChatGPT assisted with project planning, code drafts, explanations
of machine-learning concepts, troubleshooting, and documentation.

I used this assistance to assemble and run the project locally,
review outputs, and test the prediction interface. The implementation
contains AI-assisted code and is not claimed to have been written
entirely without assistance.

I am responsible for reviewing the submitted code and understanding
its behaviour and limitations.

## External resources

- Kaggle dataset:
  https://www.kaggle.com/datasets/lainguyn123/student-performance-factors
- Scikit-learn documentation:
  https://scikit-learn.org/stable/
- Streamlit documentation:
  https://docs.streamlit.io/