# Student Academic Performance Prediction

AI/ML internship assessment project comparing Logistic Regression
and K-Nearest Neighbors for student performance classification.

## Dataset
Student Performance Factors by lainguyn123 on Kaggle.

Source:
https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

Download and extract StudentPerformanceFactors.csv into data/raw/.

## Planned workflow
1. Inspect and preprocess the dataset.
2. Perform exploratory data analysis.
3. Define and document performance categories from Exam_Score.
4. Build preprocessing pipelines and compare Logistic Regression and KNN.
5. Evaluate accuracy, precision, recall, F1-score, and confusion matrices.
6. Build a Streamlit prediction interface.
7. Document results and limitations.

Exam_Score will not be used as an input when predicting categories
derived from that score.

## Training and evaluation strategy

The project uses six input features:
Hours_Studied, Attendance, Previous_Scores, Sleep_Hours,
Tutoring_Sessions, and Access_to_Resources.

Pass/Fail labels are derived from Exam_Score using an assumed
demonstration threshold of 70:
- Fail = 1 for scores below 70.
- Pass = 0 for scores of 70 or above.

This threshold is not an official school policy. Exam_Score is excluded
from the model inputs to prevent target leakage.

The data is split into 80% training and 20% testing using a stratified
split with random_state=42.

Five-fold stratified cross-validation on the training set is planned
for model comparison and any tuning. Preprocessing will be fitted
inside the pipelines. Macro F1 will be the primary selection metric,
with Fail-class recall and precision also reviewed.

The held-out test set will be used for final evaluation, not tuning.