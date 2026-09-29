"""Interactive Iris species predictor."""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from train_iris_model import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, tune_model


DATA_PATH = Path(__file__).parent / "data" / "Iris.csv"
FEATURE_LABELS = {
    "SepalLengthCm": "Sepal length (cm)",
    "SepalWidthCm": "Sepal width (cm)",
    "PetalLengthCm": "Petal length (cm)",
    "PetalWidthCm": "Petal width (cm)",
}


@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_model():
    data = get_data()
    return tune_model(data[FEATURE_COLUMNS], data[TARGET_COLUMN]).best_estimator_


st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

st.title("Iris Species Predictor")
st.write(
    "Explore the Iris dataset with a logistic regression model. "
    "Adjust the sliders to update the prediction instantly."
)

data = get_data()
model = get_model()
st.caption(
    f"Cross-validated logistic regression trained on all {len(data)} rows. "
    f"Selected regularization C={model.named_steps['logisticregression'].C:g}."
)

st.subheader("Flower measurements")
left, right = st.columns(2)
inputs = {}
for index, column in enumerate(FEATURE_COLUMNS):
    values = data[column]
    with (left if index % 2 == 0 else right):
        inputs[column] = st.slider(
            FEATURE_LABELS[column],
            min_value=float(values.min()),
            max_value=float(values.max()),
            value=round(float(values.median()), 1),
            step=0.1,
            format="%.1f cm",
            key=column,
        )

sample = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
probability_by_species = dict(zip(model.classes_, probabilities))
confidence = probability_by_species[prediction]

st.markdown(
    """
    <style>
    .prediction-overlay {
        position: fixed;
        right: 1.5rem;
        bottom: 1.5rem;
        z-index: 999;
        width: min(19rem, calc(100vw - 3rem));
        padding: 1rem 1.25rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 1rem;
        background: var(--secondary-background-color);
        color: var(--text-color);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2);
    }
    .prediction-overlay-label {
        margin: 0 0 0.2rem;
        color: #548235;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .prediction-overlay-species {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 750;
    }
    .prediction-overlay-confidence {
        margin: 0.3rem 0 0;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    @media (max-width: 640px) {
        .prediction-overlay {
            right: 1rem;
            bottom: 1rem;
            width: calc(100vw - 4rem);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="prediction-overlay" role="status" aria-live="polite">
        <p class="prediction-overlay-label">Predicted species</p>
        <p class="prediction-overlay-species">{escape(str(prediction))}</p>
        <p class="prediction-overlay-confidence">{confidence:.1%} model confidence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Estimated probability by species")
probability_data = pd.DataFrame(
    {
        "Species": model.classes_,
        "Probability": probabilities,
    }
).set_index("Species")
st.bar_chart(probability_data)
st.caption(
    "Probabilities are model estimates, not guarantees. "
    "Measurements outside the dataset's observed ranges are not supported."
)

with st.expander("About this model"):
    st.write(
        "The app compares regularization strengths using stratified five-fold "
        "cross-validation, then uses the best logistic regression model. It uses "
        "all four measurements and standardizes them before fitting."
    )
    st.write("The CSV's `Id` column is excluded from model training.")
