# IRIS_project

Explore and evaluate a multiclass logistic regression model with the included
`data/Iris.csv` dataset.

## Interactive app

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app provides live measurement sliders and a floating prediction overlay
that updates as inputs change. It also displays estimated probabilities for
each species. At startup, it selects logistic-regression regularization using
stratified five-fold cross-validation on the included dataset.

## Command-line evaluation

```bash
python -m pip install -r requirements.txt
python train_iris_model.py
```

The script compares the default logistic regression against a model whose
regularization strength is selected by stratified five-fold cross-validation
using only the training split. It then reports both holdout accuracies, a
per-species report, and a confusion matrix for the tuned model. The four
measurements are scaled; the CSV's `Id` column is ignored.

On the included dataset and fixed stratified 80/20 split, cross-validation
selected `C=3` and improved holdout accuracy from 93.3% to 96.7%. This is a
single small-dataset evaluation, not a guarantee of future performance.

To use another CSV with the same columns:

```bash
python train_iris_model.py --csv path/to/Iris.csv
```
