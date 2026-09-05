from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import numpy as np
import pandas as pd


app = FastAPI(
    title="AI Finance Credit Risk API",
    description="Explainable AI API for credit risk assessment",
    version="1.0.0"
)


# --------------------------------------------------
# Load model and preprocessing artifacts
# --------------------------------------------------

MODEL_PATH = "../data/final_lightgbm_model.pkl"
PREPROCESSOR_PATH = "../data/final_preprocessor.pkl"
FEATURE_NAMES_PATH = "../data/final_feature_names.pkl"
EXPLAINER_PATH = "../data/shap_explainer.pkl"


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)
explainer = joblib.load(EXPLAINER_PATH)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class PredictionRequest(BaseModel):

    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float

    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None

    DAYS_BIRTH: int
    DAYS_EMPLOYED: int

    CODE_GENDER: str = "M"
    NAME_FAMILY_STATUS: str = "Married"
    NAME_EDUCATION_TYPE: str = "Secondary / secondary special"


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Finance Credit Risk API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    # Convert request into dictionary
    input_data = request.model_dump()

    # Create dataframe
    df = pd.DataFrame([input_data])

    # --------------------------------------------------
    # Add derived application features
    # --------------------------------------------------

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"]
        / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    )

    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Match the exact raw features expected by the
    # final 298 -> 422 preprocessing pipeline
    # --------------------------------------------------

    expected_features = list(preprocessor.feature_names_in_)

    for feature in expected_features:

        if feature not in df.columns:
            df[feature] = np.nan

    # Keep only expected features and exact order
    df = df[expected_features]

    # --------------------------------------------------
    # Transform using final preprocessing pipeline
    # --------------------------------------------------

    X_processed = preprocessor.transform(df)

    # Safety check
    if X_processed.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Preprocessed feature count mismatch: "
            f"got {X_processed.shape[1]}, "
            f"expected {model.n_features_in__}"
        )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    probability = float(
        model.predict_proba(X_processed)[0, 1]
    )

    # Validation-selected threshold
    threshold = 0.15

    prediction = int(probability >= threshold)

    # --------------------------------------------------
    # Risk category
    # --------------------------------------------------

    if probability >= 0.50:

        risk_level = "High Risk"

    elif probability >= threshold:

        risk_level = "Moderate Risk"

    else:

        risk_level = "Low Risk"

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return {
        "default_probability": round(probability, 4),
        "default_percentage": round(probability * 100, 2),
        "prediction": prediction,
        "risk_level": risk_level,
        "threshold": threshold
    }