"""Train and evaluate a logistic regression model on the Iris CSV dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
TARGET_COLUMN = "Species"


def load_dataset(csv_path):
    data = pd.read_csv(csv_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    return data


def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate logistic regression on the Iris dataset."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent / "data" / "Iris.csv",
        help="Path to the Iris CSV file (default: data/Iris.csv).",
    )
    args = parser.parse_args()

    data = load_dataset(args.csv)
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(target.unique())

    print(f"Dataset: {args.csv}")
    print(f"Rows: {len(data)} | Features: {', '.join(FEATURE_COLUMNS)}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=labels, zero_division=0))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(f"Labels: {labels}")
    print(confusion_matrix(y_test, predictions, labels=labels))


if __name__ == "__main__":
    main()
