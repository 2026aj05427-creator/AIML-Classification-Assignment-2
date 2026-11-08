
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ----------------------------------
# 1. Application configuration
# ----------------------------------

st.set_page_config(
    page_title="Classification Model Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Classification Model Comparison Lab")

st.markdown("""
### Machine Learning Assignment 2

Compare five supervised classification algorithms
using the Breast Cancer Wisconsin Diagnostic dataset.

Upload your labelled test CSV to evaluate the saved models.
""")

TARGET = "diagnosis_malignant"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

# ----------------------------------
# 2. Load saved models
# ----------------------------------

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        path = os.path.join("model", filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded_models[name] = joblib.load(path)

    return loaded_models


try:
    models = load_models()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# ----------------------------------
# 3. Upload test dataset
# ----------------------------------

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload your labelled test CSV",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload test_data.csv using the sidebar.")

    st.markdown("""
    **Instructions**

    1. Upload the test CSV.
    2. Select a classification model.
    3. View evaluation metrics.
    4. Compare all five models.
    """)

    st.stop()


try:
    data = pd.read_csv(uploaded_file)

except Exception as e:
    st.error(f"Unable to read CSV: {e}")
    st.stop()


# ----------------------------------
# 4. Validate uploaded dataset
# ----------------------------------

if TARGET not in data.columns:

    st.error(
        f"Target column '{TARGET}' is missing from uploaded CSV."
    )

    st.stop()


X_test = data.drop(columns=[TARGET])
y_test = data[TARGET].astype(int)

if X_test.empty:
    st.error("No input features found.")
    st.stop()

if X_test.isnull().any().any() or y_test.isnull().any():
    st.error("The uploaded dataset contains missing values.")
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):
    st.error("Target must contain 0 (Benign) and 1 (Malignant).")
    st.stop()

if y_test.nunique() != 2:
    st.error("Both target classes must be present to calculate AUC.")
    st.stop()

# Validate input feature names and order against saved models
reference_features = list(models["Logistic Regression"].feature_names_in_)

missing = sorted(set(reference_features) - set(X_test.columns))
extra = sorted(set(X_test.columns) - set(reference_features))

if missing or extra:
    st.error(
        f"Feature mismatch. Missing columns: {missing}. "
        f"Unexpected columns: {extra}."
    )
    st.stop()

X_test = X_test[reference_features]

st.success("Test dataset uploaded and validated successfully!")

st.write("### Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Test Records", len(data))
col2.metric("Input Features", X_test.shape[1])
col3.metric("Target Classes", y_test.nunique())

with st.expander("Preview uploaded dataset"):
    st.dataframe(data.head(10), use_container_width=True)


# ----------------------------------
# 5. Evaluate all five models
# ----------------------------------

all_predictions = {}
all_probabilities = {}
comparison = []

for name, model in models.items():

    predictions = model.predict(X_test)

    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(X_test)[:, positive_index]

    all_predictions[name] = predictions
    all_probabilities[name] = probabilities

    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, predictions)
    })

comparison_df = pd.DataFrame(comparison)

# ----------------------------------
# 6. Display model comparison
# ----------------------------------

st.header("📊 Performance Comparison")

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}"
    }),
    use_container_width=True
)

best_model = comparison_df.sort_values(
    ["F1", "AUC"],
    ascending=False
).iloc[0]

st.success(
    f"Best model by F1 score: {best_model['Model']} "
    f"({best_model['F1']:.4f})"
)


# ----------------------------------
# 7. Individual model selection
# ----------------------------------

st.header("🔍 Individual Model Evaluation")

selected_model = st.selectbox(
    "Select a classification model",
    list(MODEL_FILES.keys())
)

selected_predictions = all_predictions[selected_model]

selected_probabilities = all_probabilities[selected_model]

selected_row = comparison_df[
    comparison_df["Model"] == selected_model
].iloc[0]

metric_cols = st.columns(3)

metric_names = [
    ("Accuracy", "Accuracy"),
    ("AUC", "AUC"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1", "F1"),
    ("MCC", "MCC")
]

for i, (label, key) in enumerate(metric_names):

    metric_cols[i % 3].metric(
        label,
        f"{selected_row[key]:.4f}"
    )


# ----------------------------------
# 8. Confusion matrix
# ----------------------------------

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 4))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title(f"Confusion Matrix: {selected_model}")

st.pyplot(fig)
plt.close(fig)


# ----------------------------------
# 9. Classification report
# ----------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ----------------------------------
# 10. Footer
# ----------------------------------

st.divider()

st.caption("""
M.Tech AIML | Machine Learning Assignment 2

All models are loaded from previously saved files.
Uploaded test data is used only for evaluation.
No retraining takes place in this application.

Educational demonstration only. Not a medical diagnostic tool.
""")
