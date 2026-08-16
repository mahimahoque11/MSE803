"""Week 3 Activity 2: compare regression models for a missing value.

The program predicts David's missing Net worth from Salary using:
1. ordinary linear regression, and
2. degree-2 polynomial regression.

Models are compared with leave-one-out cross-validation (LOOCV), which is
appropriate here because only seven complete Salary/Net-worth pairs exist.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np


ACTIVITY_DIR = Path(__file__).parent
SOURCE = ACTIVITY_DIR.parent / "activity1" / "cleaned_dataset.csv"
OUTPUT = ACTIVITY_DIR / "predicted_dataset.csv"
PLOT = ACTIVITY_DIR / "regression_comparison.png"


def to_float(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, x] for linear or [1, x, x^2] for quadratic."""
    return np.column_stack([x**power for power in range(degree + 1)])


def fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Fit least-squares regression and return its coefficients."""
    coefficients, *_ = np.linalg.lstsq(design_matrix(x, degree), y, rcond=None)
    return coefficients


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    degree = len(coefficients) - 1
    return design_matrix(x, degree) @ coefficients


def loocv(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, float]:
    """Evaluate unseen-point predictions by holding out each row once."""
    predictions = []
    for index in range(len(x)):
        keep = np.arange(len(x)) != index
        coefficients = fit(x[keep], y[keep], degree)
        predictions.append(float(predict(x[index : index + 1], coefficients)[0]))

    predicted = np.array(predictions)
    errors = y - predicted
    squared_error = float(np.sum(errors**2))
    total_variation = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "R2": 1 - squared_error / total_variation,
    }


def save_predictions(rows: list[dict[str, str]], predicted_value: float) -> None:
    output_rows = [row.copy() for row in rows]
    for row in output_rows:
        if row["Name"] == "David" and not row["Net worth"].strip():
            row["Net worth"] = f"{predicted_value:.2f}"
            row["Prediction note"] = "Predicted by selected regression model from Salary"
        elif any(not row[column].strip() for column in ("Age", "Net worth", "Salary")):
            row["Prediction note"] = "Not predicted: insufficient numeric predictors"
        else:
            row["Prediction note"] = "Observed value"

    fieldnames = list(rows[0]) + ["Prediction note"]
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def create_plot(
    x: np.ndarray,
    y: np.ndarray,
    models: dict[str, np.ndarray],
    david_salary: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is missing; install it with: python -m pip install matplotlib")
        return False

    x_curve = np.linspace(x.min(), x.max(), 200)
    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=70, color="#1677b8", label="Observed complete records")
    plt.plot(x_curve, predict(x_curve, models["Linear"]), color="#e45756", label="Linear model")
    plt.plot(
        x_curve,
        predict(x_curve, models["Polynomial (degree 2)"]),
        color="#54a24b",
        label="Polynomial model (degree 2)",
    )
    for name, value in predictions.items():
        marker = "X" if name == "Linear" else "D"
        plt.scatter(david_salary, value, s=110, marker=marker, label=f"David: {name} prediction")

    plt.title("Predicting Missing Net Worth from Salary")
    plt.xlabel("Salary")
    plt.ylabel("Net worth")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main() -> None:
    rows = load_rows()
    complete = [
        row
        for row in rows
        if to_float(row["Salary"]) is not None and to_float(row["Net worth"]) is not None
    ]
    salary = np.array([to_float(row["Salary"]) for row in complete], dtype=float)
    net_worth = np.array([to_float(row["Net worth"]) for row in complete], dtype=float)
    david = next(row for row in rows if row["Name"] == "David")
    david_salary = float(david["Salary"])

    specifications = {"Linear": 1, "Polynomial (degree 2)": 2}
    models = {name: fit(salary, net_worth, degree) for name, degree in specifications.items()}
    scores = {name: loocv(salary, net_worth, degree) for name, degree in specifications.items()}
    predictions = {
        name: float(predict(np.array([david_salary]), coefficients)[0])
        for name, coefficients in models.items()
    }

    # Lower cross-validated RMSE indicates better prediction on unseen records.
    selected = min(scores, key=lambda name: scores[name]["RMSE"])
    selected_prediction = predictions[selected]
    save_predictions(rows, selected_prediction)
    plot_created = create_plot(salary, net_worth, models, david_salary, predictions)

    print("ACTIVITY 2: MISSING-VALUE PREDICTION")
    print("=" * 55)
    print(f"Training observations: {len(complete)}")
    print("Target: Net worth | Predictor: Salary")
    print("Evaluation: leave-one-out cross-validation\n")
    for name in specifications:
        print(name)
        print(f"  LOOCV MAE:  {scores[name]['MAE']:,.2f}")
        print(f"  LOOCV RMSE: {scores[name]['RMSE']:,.2f}")
        print(f"  LOOCV R2:   {scores[name]['R2']:.3f}")
        print(f"  David prediction at Salary {david_salary:,.0f}: {predictions[name]:,.2f}\n")

    print(f"Selected model: {selected} (lowest LOOCV RMSE)")
    print(f"Predicted missing Net worth for David: {selected_prediction:,.2f}")
    print("Heidi's numeric values remain missing because she has no numeric predictor values.")
    print("Non-numeric missing fields remain missing because regression is inappropriate for them.")
    print(f"\nPredicted dataset saved to: {OUTPUT}")
    if plot_created:
        print(f"Comparison graph saved to: {PLOT}")


if __name__ == "__main__":
    main()
