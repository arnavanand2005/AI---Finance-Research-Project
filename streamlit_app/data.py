"""
Static research results extracted from the project notebooks
(01_Exploratory_Data_Analysis ... 06_SHAP_Explainability).

Every number here was copied directly out of notebook output cells —
nothing is simulated. This module has no dependency on the raw dataset
or trained model artifacts, so the showcase app can run standalone.
"""

TEAM = ["Arnav Anand", "Devansh Gupta", "Ansh Mathur"]

HEADLINE_STATS = [
    {"label": "Applications analyzed", "value": "307,511"},
    {"label": "Engineered features", "value": "422"},
    {"label": "Test ROC-AUC", "value": "0.7884"},
    {"label": "Default recall @ threshold", "value": "46.8%"},
]

# ---------------------------------------------------------------------------
# Dataset & EDA (Notebook 01)
# ---------------------------------------------------------------------------

DATASET_OVERVIEW = {
    "n_applications": 307511,
    "n_columns": 122,
    "target_negative_pct": 91.93,
    "target_positive_pct": 8.07,
    "n_missing_features": 67,
    "n_categorical_features": 16,
    "max_cardinality_feature": "ORGANIZATION_TYPE",
    "max_cardinality": 58,
}

HISTORICAL_SOURCES = [
    {"name": "bureau.csv", "desc": "Previous credit history from other institutions"},
    {"name": "bureau_balance.csv", "desc": "Monthly bureau credit status history"},
    {"name": "previous_application.csv", "desc": "Applicant's previous loan applications"},
    {"name": "installments_payments.csv", "desc": "Historical installment payment behavior"},
    {"name": "POS_CASH_balance.csv", "desc": "Previous point-of-sale / cash loan behavior"},
    {"name": "credit_card_balance.csv", "desc": "Previous credit-card account behavior"},
]

TARGET_CORRELATIONS = [
    {"feature": "EXT_SOURCE_3", "correlation": -0.179},
    {"feature": "EXT_SOURCE_2", "correlation": -0.160},
    {"feature": "EXT_SOURCE_1", "correlation": -0.155},
]

EDA_FINDINGS = [
    "Applicants experiencing payment difficulty are younger on average than those without difficulty.",
    "Applicants experiencing payment difficulty have shorter employment durations on average.",
    "Lower education levels are associated with higher payment-difficulty rates.",
    "Basic financial variables (income, credit amount, annuity) show relatively small differences between groups.",
    "The basic credit-to-income ratio shows limited separation between the two target groups.",
    "External credit-score features (EXT_SOURCE_1/2/3) are the strongest numerical correlates with default.",
]

# ---------------------------------------------------------------------------
# Preprocessing & Feature Engineering (Notebooks 02-03)
# ---------------------------------------------------------------------------

ENGINEERED_APPLICATION_FEATURES = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
]

PREPROCESSING_STEPS = [
    "Stratified train / validation / test split to preserve the imbalanced target distribution",
    "Median imputation for missing numerical values",
    "Most-frequent-category imputation for missing categorical values",
    "One-hot encoding of categorical variables (unknown categories ignored at inference)",
    "Pipeline fit exclusively on training data, then applied to validation/test to prevent leakage",
]

FEATURE_GROUP_COUNTS = [
    {"group": "Application", "n_features": 254},
    {"group": "Bureau", "n_features": 38},
    {"group": "Bureau Balance", "n_features": 16},
    {"group": "Previous Applications", "n_features": 40},
    {"group": "Installments", "n_features": 16},
    {"group": "POS Cash", "n_features": 24},
    {"group": "Credit Card", "n_features": 34},
]

# ---------------------------------------------------------------------------
# Model Training & Ablation Study (Notebook 04)
# ---------------------------------------------------------------------------

ABLATION_STUDY = [
    {"experiment": "Application Only", "roc_auc": 0.763046, "pr_auc": 0.255328},
    {"experiment": "+ Bureau", "roc_auc": 0.768690, "pr_auc": 0.262631},
    {"experiment": "+ Bureau Balance", "roc_auc": 0.767901, "pr_auc": 0.263126},
    {"experiment": "+ Previous Applications", "roc_auc": 0.775064, "pr_auc": 0.272123},
    {"experiment": "+ Installments", "roc_auc": 0.776533, "pr_auc": 0.274880},
    {"experiment": "+ POS Cash", "roc_auc": 0.779164, "pr_auc": 0.278727},
    {"experiment": "+ Credit Card (All Sources)", "roc_auc": 0.781644, "pr_auc": 0.280101},
]

CONFIG_COMPARISON = [
    {"experiment": "All Historical Sources (selected)", "roc_auc": 0.781644, "pr_auc": 0.280101},
    {"experiment": "Regularized LightGBM", "roc_auc": 0.781424, "pr_auc": 0.279951},
    {"experiment": "63-Leaf LightGBM", "roc_auc": 0.781274, "pr_auc": 0.280070},
    {"experiment": "Min Child Samples = 20", "roc_auc": 0.780706, "pr_auc": 0.279721},
    {"experiment": "100% Feature LightGBM", "roc_auc": 0.780175, "pr_auc": 0.279303},
    {"experiment": "Weighted LightGBM (class-weighted)", "roc_auc": 0.718384, "pr_auc": 0.185339},
]

FINAL_MODEL_CONFIG = {
    "algorithm": "Unweighted LightGBM",
    "n_features": 422,
    "n_train": 184506,
    "n_val": 61502,
    "n_test": 61503,
    "best_iteration": 731,
    "val_roc_auc": 0.7816,
    "val_pr_auc": 0.2801,
}

# ---------------------------------------------------------------------------
# Final Test Evaluation (Notebook 05)
# ---------------------------------------------------------------------------

FINAL_TEST_METRICS = {
    "roc_auc": 0.7884,
    "pr_auc": 0.2800,
    "threshold": 0.15,
    "accuracy": 0.8543,
    "precision": 0.2687,
    "recall": 0.4679,
    "f1": 0.3414,
}

CONFUSION_MATRIX = {
    "tn": 50216,
    "fp": 6322,
    "fn": 2642,
    "tp": 2323,
}

CLASSIFICATION_REPORT = [
    {"class": "Non-Default", "precision": 0.95, "recall": 0.89, "f1": 0.92, "support": 56538},
    {"class": "Default", "precision": 0.27, "recall": 0.47, "f1": 0.34, "support": 4965},
]

# ---------------------------------------------------------------------------
# SHAP Explainability (Notebook 06)
# ---------------------------------------------------------------------------

SHAP_TOP_FEATURES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "CREDIT_GOODS_RATIO",
    "LATE_PAYMENT_RATE",
    "AMT_ANNUITY",
    "POS_INSTALMENT_FUTURE_MAX",
    "AMT_PAYMENT_SUM",
    "POS_MONTH_COUNT",
    "DAYS_EMPLOYED",
]

# Relative importance is illustrative ordering (descending, evenly spaced)
# for the bar chart — SHAP mean(|value|) magnitudes are not individually
# reported as numbers in the notebook, only the ranked order.
SHAP_IMPORTANCE_RANK = [
    {"feature": f, "rank": i + 1} for i, f in enumerate(SHAP_TOP_FEATURES)
]

CASE_STUDIES = [
    {
        "label": "High-Risk Applicant",
        "predicted_probability": 0.8683,
        "risk_level": "High Risk",
        "top_contributors": [
            "EXT_SOURCE_2",
            "PREV_REFUSAL_RATE",
            "LATE_PAYMENT_RATE",
            "AMT_ANNUITY",
            "POS_DPD_DEF_RATE",
            "EXT_SOURCE_3",
        ],
        "narrative": (
            "Major risk-increasing contributors included a low EXT_SOURCE_2 score, "
            "a high previous-application refusal rate, a high late-payment rate, "
            "a large annuity, and delinquent POS behavior."
        ),
    },
    {
        "label": "Low-Risk Applicant",
        "predicted_probability": 0.0029,
        "risk_level": "Low Risk",
        "top_contributors": ["EXT_SOURCE_2", "EXT_SOURCE_3"],
        "narrative": (
            "Strong EXT_SOURCE_2 and EXT_SOURCE_3 scores contributed negatively to the "
            "model output, substantially reducing predicted default risk."
        ),
    },
]

SHAP_FINDINGS = [
    "Lower external-source scores (EXT_SOURCE_1/2/3) generally push predictions toward higher default risk; "
    "higher scores push predictions toward lower risk.",
    "The SHAP dependence analysis for EXT_SOURCE_2 shows a clear negative relationship: "
    "as the score increases, its contribution to predicted default risk decreases.",
    "Both application-level information and historical credit-behavior features contribute "
    "substantially to the model's predictions — no single data source dominates.",
]

PIPELINE_STAGES = [
    {
        "stage": "01",
        "title": "Exploratory Data Analysis",
        "desc": "Understand dataset structure, class imbalance, missing values, and applicant characteristics associated with payment difficulty.",
    },
    {
        "stage": "02",
        "title": "Data Preprocessing",
        "desc": "Stratified splitting, missing-value imputation, categorical encoding, and leakage-safe pipeline fitting.",
    },
    {
        "stage": "03",
        "title": "Feature Engineering",
        "desc": "Aggregate six historical credit/repayment datasets into applicant-level risk features.",
    },
    {
        "stage": "04",
        "title": "Model Training & Selection",
        "desc": "Train and ablate LightGBM configurations across feature groups and class-weighting strategies.",
    },
    {
        "stage": "05",
        "title": "Model Evaluation",
        "desc": "Evaluate the selected model on a held-out test set with threshold-based classification metrics.",
    },
    {
        "stage": "06",
        "title": "SHAP Explainability",
        "desc": "Interpret global and individual predictions to make the model's decisions transparent.",
    },
]
