"""Interactive Iris species predictor."""

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from train_iris_model import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, tune_model


DATA_PATH = Path(__file__).parent / "data" / "Iris.csv"
FEATURE_LABELS = {
    "SepalLengthCm": "Sepal length",
    "SepalWidthCm": "Sepal width",
    "PetalLengthCm": "Petal length",
    "PetalWidthCm": "Petal width",
}
SPECIES_COLORS = {
    "Iris-setosa": "#16a085",
    "Iris-versicolor": "#d58b16",
    "Iris-virginica": "#8262c6",
}


@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)


@st.cache_resource
def get_model():
    data = get_data()
    return tune_model(data[FEATURE_COLUMNS], data[TARGET_COLUMN]).best_estimator_


def set_measurements(values):
    for column, value in values.items():
        st.session_state[column] = value


st.set_page_config(
    page_title="Iris — Species Studio",
    page_icon="🌸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 7rem;
    }
    .eyebrow {
        margin: 0 0 0.5rem;
        color: #16a085;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero-copy {
        max-width: 42rem;
        margin: 0.4rem 0 1.5rem;
        opacity: 0.76;
        font-size: 1.08rem;
        line-height: 1.6;
    }
    .section-kicker {
        margin: 0 0 0.25rem;
        color: #16a085;
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .result-card {
        min-height: 10rem;
        margin: 0.5rem 0 1rem;
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-left: 5px solid var(--species-color);
        border-radius: 1rem;
        background: var(--secondary-background-color);
    }
    .result-label {
        margin: 0 0 0.45rem;
        opacity: 0.68;
        font-size: 0.8rem;
        font-weight: 650;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .result-species {
        margin: 0;
        color: var(--species-color);
        font-size: clamp(1.5rem, 4vw, 2.15rem);
        font-weight: 780;
        line-height: 1.15;
    }
    .result-confidence {
        margin: 0.55rem 0 0;
        opacity: 0.76;
        font-size: 0.95rem;
    }
    .probability-row {
        margin: 0.8rem 0;
    }
    .probability-label {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.35rem;
        font-size: 0.88rem;
    }
    .probability-track {
        height: 0.55rem;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(128, 128, 128, 0.18);
    }
    .probability-fill {
        height: 100%;
        border-radius: inherit;
        transition: width 180ms ease;
    }
    .prediction-overlay {
        position: fixed;
        right: 1.5rem;
        bottom: 1.25rem;
        z-index: 999;
        width: min(20rem, calc(100vw - 3rem));
        padding: 0.9rem 1.15rem;
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-left: 5px solid var(--species-color);
        border-radius: 0.9rem;
        background: var(--secondary-background-color);
        color: var(--text-color);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.17);
    }
    .overlay-label {
        margin: 0 0 0.2rem;
        opacity: 0.68;
        font-size: 0.7rem;
        font-weight: 750;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .overlay-species {
        margin: 0;
        color: var(--species-color);
        font-size: 1.2rem;
        font-weight: 780;
    }
    .overlay-confidence {
        margin: 0.2rem 0 0;
        opacity: 0.72;
        font-size: 0.82rem;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
        }
        .prediction-overlay {
            right: 0.8rem;
            bottom: 0.7rem;
            width: calc(100vw - 2.6rem);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

data = get_data()
model = get_model()
species = sorted(data[TARGET_COLUMN].unique())
presets = {
    name: data.loc[data[TARGET_COLUMN] == name, FEATURE_COLUMNS]
    .median()
    .round(1)
    .to_dict()
    for name in species
}
defaults = data[FEATURE_COLUMNS].median().round(1).to_dict()

for column, value in defaults.items():
    st.session_state.setdefault(column, float(value))

st.markdown('<p class="eyebrow">Interactive machine learning demo</p>', unsafe_allow_html=True)
st.title("Iris Species Studio")
st.markdown(
    "Explore how flower measurements shape a prediction. Adjust the sliders or "
    "load a species example to see the model respond live.",
    help="All measurements are in centimeters and limited to the range in the dataset.",
)
st.caption(
    f"Scikit-learn logistic regression · {len(data)} training samples · "
    f"cross-validated regularization C="
    f"{model.named_steps['logisticregression'].C:g}"
)

controls, results = st.columns([1.08, 0.92], gap="large")

with controls:
    st.markdown('<p class="section-kicker">01 / Measurements</p>', unsafe_allow_html=True)
    st.subheader("Describe your flower")
    st.write("Choose an example or fine-tune each measurement.")
    preset_columns = st.columns(4)
    for column, name in zip(preset_columns[:3], species):
        short_name = name.replace("Iris-", "")
        column.button(
            short_name,
            key=f"preset_{short_name}",
            use_container_width=True,
            on_click=set_measurements,
            args=(presets[name],),
            help=f"Load typical measurements for {name}.",
        )
    preset_columns[3].button(
        "Reset",
        key="preset_reset",
        use_container_width=True,
        on_click=set_measurements,
        args=(defaults,),
        help="Restore the dataset's median measurements.",
    )

    st.markdown("")
    slider_columns = st.columns(2)
    for index, feature in enumerate(FEATURE_COLUMNS):
        values = data[feature]
        with slider_columns[index % 2]:
            st.slider(
                f"{FEATURE_LABELS[feature]} (cm)",
                min_value=float(values.min()),
                max_value=float(values.max()),
                step=0.1,
                format="%.1f",
                key=feature,
                help=(
                    f"Observed range: {values.min():.1f}–{values.max():.1f} cm. "
                    f"Dataset median: {values.median():.1f} cm."
                ),
            )

with results:
    st.markdown('<p class="section-kicker">02 / Live result</p>', unsafe_allow_html=True)
    st.subheader("Model prediction")
    sample = pd.DataFrame(
        [{feature: st.session_state[feature] for feature in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )
    prediction = str(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    color = SPECIES_COLORS.get(prediction, "#16a085")
    probability_by_species = dict(zip(model.classes_, probabilities))
    confidence = float(probability_by_species[prediction])

    st.markdown(
        f"""
        <div class="result-card" style="--species-color: {color}" role="status" aria-live="polite">
            <p class="result-label">Most likely species</p>
            <p class="result-species">{escape(prediction.replace("Iris-", ""))}</p>
            <p class="result-confidence">{confidence:.1%} estimated confidence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Probability by species**")
    for label, probability in zip(model.classes_, probabilities):
        label = str(label)
        percentage = float(probability) * 100
        label_color = SPECIES_COLORS.get(label, "#16a085")
        st.markdown(
            f"""
            <div class="probability-row">
                <div class="probability-label">
                    <span>{escape(label.replace("Iris-", ""))}</span>
                    <strong>{percentage:.1f}%</strong>
                </div>
                <div class="probability-track" role="progressbar"
                     aria-label="{escape(label)} probability"
                     aria-valuemin="0" aria-valuemax="100"
                     aria-valuenow="{percentage:.1f}">
                    <div class="probability-fill"
                         style="width: {percentage:.1f}%; background: {label_color}"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Probabilities are model estimates, not guarantees.")

st.markdown(
    f"""
    <div class="prediction-overlay" style="--species-color: {color}"
         role="status" aria-live="polite">
        <p class="overlay-label">Live prediction</p>
        <p class="overlay-species">{escape(prediction.replace("Iris-", ""))}</p>
        <p class="overlay-confidence">{confidence:.1%} estimated confidence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How this model works"):
    st.write(
        "A standard scaler normalizes the four measurements before a multiclass "
        "logistic regression predicts the species. The regularization strength "
        "was selected from a small candidate grid using stratified five-fold "
        "cross-validation. The CSV's row ID is not used as a feature."
    )
    st.write(
        "The sliders are restricted to the observed measurement ranges in this "
        "150-row dataset. Predictions outside the range of the training data "
        "should not be considered reliable."
    )
