# Week 1 – Activity 2: Iris Dataset Exploration and Analysis

## Purpose

This activity explores the Iris dataset from the UCI Machine Learning Repository. The analysis identifies the number of records, features, and classes and checks the dataset for duplicate records.

Dataset: [Iris – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris)

## Environment and dependencies

- Python 3
- `ucimlrepo`
- `pandas`

From the root of the MSE803 repository, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required package if necessary:

```powershell
python -m pip install ucimlrepo
```

## Steps followed

1. Imported `fetch_ucirepo` from the `ucimlrepo` package.
2. Retrieved the Iris dataset using UCI dataset ID `53`.
3. Separated the feature columns and target column.
4. Combined the features and target to check complete records for duplicates.
5. Counted the records, features, unique classes, and duplicate records.

## Run the analysis

From the root of the MSE803 repository, run:

```powershell
python .\week1\activity2\iris_dataset.py
```

## Findings

| Measure | Result |
| --- | --- |
| Records | 150 |
| Features | 4 |
| Classes | 3 |
| Duplicate records | 3 |

The four features are sepal length, sepal width, petal length, and petal width. The three target classes are:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

The duplicate check includes all four feature values and the class label. Three complete records are duplicated in the dataset.

