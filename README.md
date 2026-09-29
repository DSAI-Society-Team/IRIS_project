# IRIS_project

Explore and evaluate a multiclass logistic regression model with the included
`data/Iris.csv` dataset.

## Interactive app

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app provides measurement inputs and displays the predicted species and
estimated probabilities for each class. It trains the model on the included
dataset when the app starts.

## Command-line evaluation

```bash
python -m pip install -r requirements.txt
python train_iris_model.py
```

The script uses the four sepal and petal measurements as features, scales them,
and predicts `Species`. It intentionally ignores the CSV's `Id` column. The
output includes holdout accuracy, a per-species classification report, and a
confusion matrix.

To use another CSV with the same columns:

```bash
python train_iris_model.py --csv path/to/Iris.csv
```
