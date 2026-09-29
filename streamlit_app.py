"""Interactive Iris species predictor."""

from pathlib import Path

import pandas as pd
import streamlit as st

from train_iris_model import FEATURE_COLUMNS, TARGET_COLUMN, build_model, load_dataset


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
    model = build_model()
    model.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    return model


st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

st.title("Iris Species Predictor")
st.write(
    "Explore the Iris dataset with a logistic regression model. "
    "Enter flower measurements to see the predicted species."
)

data = get_data()
model = get_model()
st.caption(f"Model trained on all {len(data)} rows in the included Iris dataset.")

with st.form("measurements"):
    st.subheader("Flower measurements")
    left, right = st.columns(2)
    inputs = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        values = data[column]
        with (left if index % 2 == 0 else right):
            inputs[column] = st.number_input(
                FEATURE_LABELS[column],
                min_value=float(values.min()),
                max_value=float(values.max()),
                value=float(values.median()),
                step=0.1,
                format="%.1f",
            )
    submitted = st.form_submit_button("Predict species", type="primary")

if submitted:
    sample = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    probability_by_species = dict(zip(model.classes_, probabilities))
    confidence = probability_by_species[prediction]

    st.subheader("Prediction")
    st.success(f"**{prediction}** — {confidence:.1%} model confidence")
    probability_data = pd.DataFrame(
        {
            "Species": model.classes_,
            "Probability": probabilities,
        }
    ).set_index("Species")
    st.write("Estimated probability by species")
    st.bar_chart(probability_data)
    st.caption(
        "Probabilities are model estimates, not guarantees. "
        "Measurements outside the dataset's observed ranges are not supported."
    )

with st.expander("About this model"):
    st.write(
        "The app uses all four sepal and petal measurements, standardizes them, "
        "and predicts one of the three species with scikit-learn logistic regression."
    )
    st.write("The CSV's `Id` column is excluded from model training.")
