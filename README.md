# Machine Learning Assignment 2: Classification Model Comparison

> **Learning scaffold:** Review the code, rerun the experiments in BITS Virtual Lab, understand each step, and write your own interpretation before submission. Do not submit without verifying and personalizing the work.

## a. Problem statement
Build and compare five supervised classification models to predict whether a breast mass is benign or malignant using numerical measurements derived from digitized fine-needle aspirate (FNA) images. The task is a binary classification problem.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Official page:** https://uci-ics-mlr-prod.aws.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic
- **Instances:** 569
- **Input features:** 30 numerical measurements
- **Target:** `diagnosis_malignant` (0 = Benign, 1 = Malignant)
- **Missing values:** None in the source dataset
- **License:** CC BY 4.0
- **Data citation:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

The 30 input features summarize ten characteristics of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension), each represented by mean, standard error and worst values.

## c. GitHub Repository Link
**Add your repository URL here after creating your own GitHub repository.**

## d. Models used

A stratified 80:20 train-test split was used with `random_state=42`. The training subset was used to fit all five models. The test subset was held out for evaluation. StandardScaler is included in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. The Decision Tree is trained without scaling.

### Model comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9954 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9298 | 0.9010 | 0.9250 | 0.8810 | 0.9024 | 0.8483 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

### Observations on model performance

Use the following as a starting point, then explain the metric trade-offs in your own words after rerunning the code:

- **Logistic Regression:** Accuracy = 0.974, AUC = 0.995, Precision = 0.976, Recall = 0.952, F1 = 0.964, MCC = 0.943. Discuss its balance between identifying malignant cases and avoiding false alarms.
- **Decision Tree:** Accuracy = 0.930, AUC = 0.901, Precision = 0.925, Recall = 0.881, F1 = 0.902, MCC = 0.848. Discuss its balance between identifying malignant cases and avoiding false alarms.
- **KNN:** Accuracy = 0.956, AUC = 0.982, Precision = 0.974, Recall = 0.905, F1 = 0.938, MCC = 0.906. Discuss its balance between identifying malignant cases and avoiding false alarms.
- **Naive Bayes:** Accuracy = 0.921, AUC = 0.989, Precision = 0.923, Recall = 0.857, F1 = 0.889, MCC = 0.829. Discuss its balance between identifying malignant cases and avoiding false alarms.
- **Support Vector Machine:** Accuracy = 0.982, AUC = 0.995, Precision = 0.976, Recall = 0.976, F1 = 0.976, MCC = 0.962. Discuss its balance between identifying malignant cases and avoiding false alarms.

- **Overall leading model by test F1 (AUC used as tie-breaker):** Support Vector Machine. This is a metric-based selection on this particular split, not a claim that it is universally best. Consider malignant-class recall and MCC when justifying your final choice.

## Streamlit application
The app supports:
1. CSV upload.
2. Model selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1 and MCC display.
4. Confusion matrix and classification report.
5. Loading saved models without retraining on uploaded test data.

Run locally:
```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

## Repository contents
- `app.py`: Streamlit user interface and evaluation logic.
- `train_models.py`: reproducible model training and metric generation.
- `dataset.csv`: full public dataset used for the experiment.
- `test_data.csv`: held-out test rows including target.
- `model/`: five serialized trained model pipelines.
- `results/model_comparison.csv`: evaluation table.

