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
from datetime import date
from pathlib import Path

import numpy as np


ACTIVITY_DIR = Path(__file__).parent
SOURCE = ACTIVITY_DIR.parent / "activity1" / "cleaned_dataset.csv"
OUTPUT = ACTIVITY_DIR / "prediction_results.csv"
COMPLETED = ACTIVITY_DIR / "completed_dataset.csv"
PLOT = ACTIVITY_DIR / "regression_comparison.png"
DATE_PLOT = ACTIVITY_DIR / "joining_date_regression.png"
COUNTRY_PLOT = ACTIVITY_DIR / "country_regression.png"
HEIDI_PLOT = ACTIVITY_DIR / "heidi_net_worth_regression.png"
HEIDI_SALARY_PLOT = ACTIVITY_DIR / "heidi_salary_regression.png"
HEIDI_AGE_PLOT = ACTIVITY_DIR / "heidi_age_regression.png"


def to_float(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def to_ordinal(value: str) -> float | None:
    """Convert an ISO date to its ordinal day number."""
    try:
        return float(date.fromisoformat(value).toordinal()) if value.strip() else None
    except ValueError:
        return None


def ordinal_to_iso(value: float) -> str:
    """Convert a predicted ordinal day number back to an ISO date."""
    return date.fromordinal(round(value)).isoformat()


def load_rows() -> list[dict[str, str]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Return [1, x] for linear or [1, x, x^2] for quadratic."""
    return np.column_stack([x**power for power in range(degree + 1)])


def fit(x: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    """Fit regression after scaling x to avoid unstable polynomial powers."""
    centre = float(np.mean(x))
    scale = float(np.std(x)) or 1.0
    scaled_x = (x - centre) / scale
    coefficients, *_ = np.linalg.lstsq(design_matrix(scaled_x, degree), y, rcond=None)
    return coefficients, centre, scale


def predict(x: np.ndarray, model: tuple[np.ndarray, float, float]) -> np.ndarray:
    coefficients, centre, scale = model
    degree = len(coefficients) - 1
    return design_matrix((x - centre) / scale, degree) @ coefficients


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


def save_predictions(
    rows: list[dict[str, str]],
    predictions: dict[str, float],
    selected_model: str,
    date_predictions: dict[str, float],
    selected_date_model: str,
    country_predictions: dict[str, float],
    selected_country_model: str,
    heidi_predictions: dict[str, float],
    selected_heidi_model: str,
    heidi_salary_predictions: dict[str, float],
    selected_heidi_salary_model: str,
    heidi_age_predictions: dict[str, float],
    selected_heidi_age_model: str,
) -> None:
    """Save both model estimates and identify the cross-validated selection."""
    output_rows = [row.copy() for row in rows]
    for row in output_rows:
        if row["Name"] == "David" and not row["Net worth"].strip():
            row["Linear prediction"] = f"{predictions['Linear']:.2f}"
            row["Polynomial prediction"] = f"{predictions['Polynomial (degree 2)']:.2f}"
            row["Selected prediction"] = f"{predictions[selected_model]:.2f}"
            row["Prediction note"] = f"Selected {selected_model} by lowest LOOCV RMSE"
        elif any(not row[column].strip() for column in ("Age", "Net worth", "Salary")):
            row["Linear prediction"] = ""
            row["Polynomial prediction"] = ""
            row["Selected prediction"] = ""
            row["Prediction note"] = "Not predicted: insufficient numeric predictors"
        else:
            row["Linear prediction"] = ""
            row["Polynomial prediction"] = ""
            row["Selected prediction"] = ""
            row["Prediction note"] = "Observed value"

        if row["Name"] == "Charlie" and not row["Join Date"].strip():
            row["Linear Join Date prediction"] = ordinal_to_iso(date_predictions["Linear"])
            row["Polynomial Join Date prediction"] = ordinal_to_iso(
                date_predictions["Polynomial (degree 2)"]
            )
            row["Selected Join Date prediction"] = ordinal_to_iso(
                date_predictions[selected_date_model]
            )
            row["Join Date prediction note"] = (
                f"Selected {selected_date_model} by lowest LOOCV RMSE in days"
            )
        else:
            row["Linear Join Date prediction"] = ""
            row["Polynomial Join Date prediction"] = ""
            row["Selected Join Date prediction"] = ""
            row["Join Date prediction note"] = (
                "Observed date" if row["Join Date"].strip() else "Not predicted"
            )

        if row["Name"] == "Grace" and not row["Country"].strip():
            row["Linear Country score"] = f"{country_predictions['Linear']:.4f}"
            row["Linear Country prediction"] = (
                "AUS" if country_predictions["Linear"] >= 0.5 else "NZ"
            )
            row["Polynomial Country score"] = f"{country_predictions['Polynomial (degree 2)']:.4f}"
            row["Polynomial Country prediction"] = (
                "AUS" if country_predictions["Polynomial (degree 2)"] >= 0.5 else "NZ"
            )
            row["Selected Country prediction"] = (
                "AUS" if country_predictions[selected_country_model] >= 0.5 else "NZ"
            )
            row["Country prediction note"] = (
                f"Selected {selected_country_model} by lowest LOOCV RMSE; NZ=0, AUS=1"
            )
        else:
            row["Linear Country score"] = ""
            row["Linear Country prediction"] = ""
            row["Polynomial Country score"] = ""
            row["Polynomial Country prediction"] = ""
            row["Selected Country prediction"] = ""
            row["Country prediction note"] = (
                "Observed country" if row["Country"].strip() else "Not predicted"
            )

        if row["Name"] == "Heidi" and not row["Net worth"].strip():
            row["Linear Heidi Net worth prediction"] = f"{heidi_predictions['Linear']:.2f}"
            row["Polynomial Heidi Net worth prediction"] = f"{heidi_predictions['Polynomial (degree 2)']:.2f}"
            row["Selected Heidi Net worth prediction"] = f"{heidi_predictions[selected_heidi_model]:.2f}"
            row["Heidi prediction note"] = (
                f"Selected {selected_heidi_model} by lowest LOOCV RMSE; predictor=Join Date"
            )
        else:
            row["Linear Heidi Net worth prediction"] = ""
            row["Polynomial Heidi Net worth prediction"] = ""
            row["Selected Heidi Net worth prediction"] = ""
            row["Heidi prediction note"] = "Observed or not applicable"

        if row["Name"] == "Heidi" and not row["Salary"].strip():
            row["Linear Heidi Salary prediction"] = f"{heidi_salary_predictions['Linear']:.2f}"
            row["Polynomial Heidi Salary prediction"] = f"{heidi_salary_predictions['Polynomial (degree 2)']:.2f}"
            row["Selected Heidi Salary prediction"] = f"{heidi_salary_predictions[selected_heidi_salary_model]:.2f}"
            row["Heidi Salary prediction note"] = (
                f"Selected {selected_heidi_salary_model} by lowest LOOCV RMSE; predictor=Join Date"
            )
        else:
            row["Linear Heidi Salary prediction"] = ""
            row["Polynomial Heidi Salary prediction"] = ""
            row["Selected Heidi Salary prediction"] = ""
            row["Heidi Salary prediction note"] = "Observed or not applicable"

        if row["Name"] == "Heidi" and not row["Age"].strip():
            row["Linear Heidi Age prediction"] = f"{heidi_age_predictions['Linear']:.2f}"
            row["Polynomial Heidi Age prediction"] = f"{heidi_age_predictions['Polynomial (degree 2)']:.2f}"
            row["Selected Heidi Age prediction"] = str(round(heidi_age_predictions[selected_heidi_age_model]))
            row["Heidi Age prediction note"] = (
                f"Selected {selected_heidi_age_model} by lowest LOOCV RMSE; predictor=Join Date"
            )
        else:
            row["Linear Heidi Age prediction"] = ""
            row["Polynomial Heidi Age prediction"] = ""
            row["Selected Heidi Age prediction"] = ""
            row["Heidi Age prediction note"] = "Observed or not applicable"

    fieldnames = list(rows[0]) + [
        "Linear prediction",
        "Polynomial prediction",
        "Selected prediction",
        "Prediction note",
        "Linear Join Date prediction",
        "Polynomial Join Date prediction",
        "Selected Join Date prediction",
        "Join Date prediction note",
        "Linear Country score",
        "Linear Country prediction",
        "Polynomial Country score",
        "Polynomial Country prediction",
        "Selected Country prediction",
        "Country prediction note",
        "Linear Heidi Net worth prediction",
        "Polynomial Heidi Net worth prediction",
        "Selected Heidi Net worth prediction",
        "Heidi prediction note",
        "Linear Heidi Salary prediction",
        "Polynomial Heidi Salary prediction",
        "Selected Heidi Salary prediction",
        "Heidi Salary prediction note",
        "Linear Heidi Age prediction",
        "Polynomial Heidi Age prediction",
        "Selected Heidi Age prediction",
        "Heidi Age prediction note",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_arranged_outputs(rows: list[dict[str, str]], experiments: list[dict[str, str]]) -> None:
    """Save a compact model comparison and a clearly labelled completed dataset."""
    result_fields = [
        "Subject",
        "Missing field",
        "Predictor",
        "Linear result",
        "Linear LOOCV RMSE",
        "Polynomial result",
        "Polynomial LOOCV RMSE",
        "Selected model",
        "Selected result",
        "Interpretation",
    ]
    with OUTPUT.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(experiments)

    completed_rows = [row.copy() for row in rows]
    predicted_lookup = {
        (experiment["Subject"], experiment["Missing field"]): experiment
        for experiment in experiments
    }
    tracked_fields = ("Age", "Net worth", "Country", "Salary", "Join Date")
    for row in completed_rows:
        for field in tracked_fields:
            source_column = f"{field} source"
            experiment = predicted_lookup.get((row["Name"], field))
            if experiment:
                row[field] = experiment["Selected result"]
                row[source_column] = f"Predicted ({experiment['Selected model']})"
            elif row[field].strip():
                row[source_column] = "Observed"
            else:
                row[source_column] = "Missing"

    completed_fields = list(rows[0]) + [f"{field} source" for field in tracked_fields]
    with COMPLETED.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=completed_fields)
        writer.writeheader()
        writer.writerows(completed_rows)


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


def create_date_plot(
    salary: np.ndarray,
    joining_date: np.ndarray,
    models: dict[str, np.ndarray],
    charlie_salary: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    salary_curve = np.linspace(salary.min(), max(salary.max(), charlie_salary), 200)
    observed_dates = [date.fromordinal(round(value)) for value in joining_date]
    plt.figure(figsize=(9, 6))
    plt.scatter(salary, observed_dates, s=70, color="#1677b8", label="Observed records")
    for name, colour in (("Linear", "#e45756"), ("Polynomial (degree 2)", "#54a24b")):
        curve_dates = [date.fromordinal(round(value)) for value in predict(salary_curve, models[name])]
        plt.plot(salary_curve, curve_dates, color=colour, label=f"{name} model")
        plt.scatter(
            charlie_salary,
            date.fromordinal(round(predictions[name])),
            s=110,
            marker="X" if name == "Linear" else "D",
            label=f"Charlie: {name} prediction",
        )

    plt.title("Predicting Charlie's Missing Joining Date from Salary")
    plt.xlabel("Salary")
    plt.ylabel("Joining Date")
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(DATE_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def create_country_plot(
    salary: np.ndarray,
    encoded_country: np.ndarray,
    models: dict[str, np.ndarray],
    grace_salary: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    salary_curve = np.linspace(salary.min(), salary.max(), 200)
    plt.figure(figsize=(9, 6))
    plt.scatter(salary, encoded_country, s=70, color="#1677b8", label="Known countries")
    plt.axhline(0.5, color="grey", linestyle="--", label="Classification threshold")
    for name, colour in (("Linear", "#e45756"), ("Polynomial (degree 2)", "#54a24b")):
        plt.plot(salary_curve, predict(salary_curve, models[name]), color=colour, label=f"{name} model")
        plt.scatter(
            grace_salary,
            predictions[name],
            s=110,
            marker="X" if name == "Linear" else "D",
            label=f"Grace: {name} score",
        )

    plt.title("Estimating Grace's Missing Country from Salary")
    plt.xlabel("Salary")
    plt.ylabel("Encoded Country (NZ = 0, AUS = 1)")
    plt.yticks([0, 0.5, 1], ["NZ (0)", "Threshold", "AUS (1)"])
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(COUNTRY_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def create_heidi_plot(
    joining_date: np.ndarray,
    net_worth: np.ndarray,
    models: dict[str, np.ndarray],
    heidi_date: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    date_curve = np.linspace(joining_date.min(), joining_date.max(), 200)
    plot_dates = [date.fromordinal(round(value)) for value in joining_date]
    curve_dates = [date.fromordinal(round(value)) for value in date_curve]
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_dates, net_worth, s=70, color="#1677b8", label="Observed records")
    for name, colour in (("Linear", "#e45756"), ("Polynomial (degree 2)", "#54a24b")):
        plt.plot(curve_dates, predict(date_curve, models[name]), color=colour, label=f"{name} model")
        plt.scatter(
            date.fromordinal(round(heidi_date)),
            predictions[name],
            s=110,
            marker="X" if name == "Linear" else "D",
            label=f"Heidi: {name} prediction",
        )

    plt.title("Predicting Heidi's Missing Net Worth from Joining Date")
    plt.xlabel("Joining Date")
    plt.ylabel("Net worth")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, ha="right")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(HEIDI_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def create_heidi_salary_plot(
    joining_date: np.ndarray,
    salary: np.ndarray,
    models: dict[str, np.ndarray],
    heidi_date: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    date_curve = np.linspace(joining_date.min(), joining_date.max(), 200)
    plot_dates = [date.fromordinal(round(value)) for value in joining_date]
    curve_dates = [date.fromordinal(round(value)) for value in date_curve]
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_dates, salary, s=70, color="#1677b8", label="Observed records")
    for name, colour in (("Linear", "#e45756"), ("Polynomial (degree 2)", "#54a24b")):
        plt.plot(curve_dates, predict(date_curve, models[name]), color=colour, label=f"{name} model")
        plt.scatter(
            date.fromordinal(round(heidi_date)),
            predictions[name],
            s=110,
            marker="X" if name == "Linear" else "D",
            label=f"Heidi: {name} prediction",
        )

    plt.title("Predicting Heidi's Missing Salary from Joining Date")
    plt.xlabel("Joining Date")
    plt.ylabel("Salary")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, ha="right")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(HEIDI_SALARY_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def create_heidi_age_plot(
    joining_date: np.ndarray,
    age: np.ndarray,
    models: dict[str, np.ndarray],
    heidi_date: float,
    predictions: dict[str, float],
) -> bool:
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    date_curve = np.linspace(joining_date.min(), joining_date.max(), 200)
    plot_dates = [date.fromordinal(round(value)) for value in joining_date]
    curve_dates = [date.fromordinal(round(value)) for value in date_curve]
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_dates, age, s=70, color="#1677b8", label="Observed records")
    for name, colour in (("Linear", "#e45756"), ("Polynomial (degree 2)", "#54a24b")):
        plt.plot(curve_dates, predict(date_curve, models[name]), color=colour, label=f"{name} model")
        plt.scatter(
            date.fromordinal(round(heidi_date)),
            predictions[name],
            s=110,
            marker="X" if name == "Linear" else "D",
            label=f"Heidi: {name} prediction",
        )

    plt.title("Predicting Heidi's Missing Age from Joining Date")
    plt.xlabel("Joining Date")
    plt.ylabel("Age")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30, ha="right")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(HEIDI_AGE_PLOT, dpi=300, bbox_inches="tight")
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

    date_complete = [
        row
        for row in rows
        if to_float(row["Salary"]) is not None and to_ordinal(row["Join Date"]) is not None
    ]
    date_salary = np.array([to_float(row["Salary"]) for row in date_complete], dtype=float)
    joining_dates = np.array([to_ordinal(row["Join Date"]) for row in date_complete], dtype=float)
    charlie = next(row for row in rows if row["Name"] == "Charlie")
    charlie_salary = float(charlie["Salary"])
    date_models = {
        name: fit(date_salary, joining_dates, degree)
        for name, degree in specifications.items()
    }
    date_scores = {
        name: loocv(date_salary, joining_dates, degree)
        for name, degree in specifications.items()
    }
    date_predictions = {
        name: float(predict(np.array([charlie_salary]), coefficients)[0])
        for name, coefficients in date_models.items()
    }
    selected_date_model = min(date_scores, key=lambda name: date_scores[name]["RMSE"])

    country_complete = [
        row
        for row in rows
        if to_float(row["Salary"]) is not None and row["Country"].strip() in {"NZ", "AUS"}
    ]
    country_salary = np.array([to_float(row["Salary"]) for row in country_complete], dtype=float)
    encoded_country = np.array([0.0 if row["Country"] == "NZ" else 1.0 for row in country_complete])
    grace = next(row for row in rows if row["Name"] == "Grace")
    grace_salary = float(grace["Salary"])
    country_models = {
        name: fit(country_salary, encoded_country, degree)
        for name, degree in specifications.items()
    }
    country_scores = {
        name: loocv(country_salary, encoded_country, degree)
        for name, degree in specifications.items()
    }
    country_predictions = {
        name: float(predict(np.array([grace_salary]), coefficients)[0])
        for name, coefficients in country_models.items()
    }
    selected_country_model = min(country_scores, key=lambda name: country_scores[name]["RMSE"])

    heidi_complete = [
        row
        for row in rows
        if to_ordinal(row["Join Date"]) is not None and to_float(row["Net worth"]) is not None
    ]
    heidi_dates = np.array([to_ordinal(row["Join Date"]) for row in heidi_complete], dtype=float)
    heidi_net_worth = np.array([to_float(row["Net worth"]) for row in heidi_complete], dtype=float)
    heidi = next(row for row in rows if row["Name"] == "Heidi")
    heidi_date = float(to_ordinal(heidi["Join Date"]))
    heidi_models = {
        name: fit(heidi_dates, heidi_net_worth, degree)
        for name, degree in specifications.items()
    }
    heidi_scores = {
        name: loocv(heidi_dates, heidi_net_worth, degree)
        for name, degree in specifications.items()
    }
    heidi_predictions = {
        name: float(predict(np.array([heidi_date]), coefficients)[0])
        for name, coefficients in heidi_models.items()
    }
    selected_heidi_model = min(heidi_scores, key=lambda name: heidi_scores[name]["RMSE"])

    heidi_salary_complete = [
        row
        for row in rows
        if to_ordinal(row["Join Date"]) is not None and to_float(row["Salary"]) is not None
    ]
    heidi_salary_dates = np.array(
        [to_ordinal(row["Join Date"]) for row in heidi_salary_complete], dtype=float
    )
    observed_salaries = np.array(
        [to_float(row["Salary"]) for row in heidi_salary_complete], dtype=float
    )
    heidi_salary_models = {
        name: fit(heidi_salary_dates, observed_salaries, degree)
        for name, degree in specifications.items()
    }
    heidi_salary_scores = {
        name: loocv(heidi_salary_dates, observed_salaries, degree)
        for name, degree in specifications.items()
    }
    heidi_salary_predictions = {
        name: float(predict(np.array([heidi_date]), coefficients)[0])
        for name, coefficients in heidi_salary_models.items()
    }
    selected_heidi_salary_model = min(
        heidi_salary_scores, key=lambda name: heidi_salary_scores[name]["RMSE"]
    )

    heidi_age_complete = [
        row
        for row in rows
        if to_ordinal(row["Join Date"]) is not None and to_float(row["Age"]) is not None
    ]
    heidi_age_dates = np.array(
        [to_ordinal(row["Join Date"]) for row in heidi_age_complete], dtype=float
    )
    observed_ages = np.array([to_float(row["Age"]) for row in heidi_age_complete], dtype=float)
    heidi_age_models = {
        name: fit(heidi_age_dates, observed_ages, degree)
        for name, degree in specifications.items()
    }
    heidi_age_scores = {
        name: loocv(heidi_age_dates, observed_ages, degree)
        for name, degree in specifications.items()
    }
    heidi_age_predictions = {
        name: float(predict(np.array([heidi_date]), coefficients)[0])
        for name, coefficients in heidi_age_models.items()
    }
    selected_heidi_age_model = min(
        heidi_age_scores, key=lambda name: heidi_age_scores[name]["RMSE"]
    )

    experiments = [
        {
            "Subject": "David", "Missing field": "Net worth", "Predictor": "Salary",
            "Linear result": f"{predictions['Linear']:.2f}",
            "Linear LOOCV RMSE": f"{scores['Linear']['RMSE']:.2f}",
            "Polynomial result": f"{predictions['Polynomial (degree 2)']:.2f}",
            "Polynomial LOOCV RMSE": f"{scores['Polynomial (degree 2)']['RMSE']:.2f}",
            "Selected model": selected, "Selected result": f"{predictions[selected]:.2f}",
            "Interpretation": "Uncertain estimate; both cross-validated R2 values are negative",
        },
        {
            "Subject": "Charlie", "Missing field": "Join Date", "Predictor": "Salary",
            "Linear result": ordinal_to_iso(date_predictions['Linear']),
            "Linear LOOCV RMSE": f"{date_scores['Linear']['RMSE']:.2f} days",
            "Polynomial result": ordinal_to_iso(date_predictions['Polynomial (degree 2)']),
            "Polynomial LOOCV RMSE": f"{date_scores['Polynomial (degree 2)']['RMSE']:.2f} days",
            "Selected model": selected_date_model,
            "Selected result": ordinal_to_iso(date_predictions[selected_date_model]),
            "Interpretation": "Uncertain estimate; hiring date is weakly related to Salary",
        },
        {
            "Subject": "Grace", "Missing field": "Country", "Predictor": "Salary",
            "Linear result": "AUS" if country_predictions['Linear'] >= 0.5 else "NZ",
            "Linear LOOCV RMSE": f"{country_scores['Linear']['RMSE']:.3f}",
            "Polynomial result": "AUS" if country_predictions['Polynomial (degree 2)'] >= 0.5 else "NZ",
            "Polynomial LOOCV RMSE": f"{country_scores['Polynomial (degree 2)']['RMSE']:.3f}",
            "Selected model": selected_country_model,
            "Selected result": "AUS" if country_predictions[selected_country_model] >= 0.5 else "NZ",
            "Interpretation": "Experimental category encoding; NZ=0 and AUS=1",
        },
        {
            "Subject": "Heidi", "Missing field": "Net worth", "Predictor": "Join Date",
            "Linear result": f"{heidi_predictions['Linear']:.2f}",
            "Linear LOOCV RMSE": f"{heidi_scores['Linear']['RMSE']:.2f}",
            "Polynomial result": f"{heidi_predictions['Polynomial (degree 2)']:.2f}",
            "Polynomial LOOCV RMSE": f"{heidi_scores['Polynomial (degree 2)']['RMSE']:.2f}",
            "Selected model": selected_heidi_model,
            "Selected result": f"{heidi_predictions[selected_heidi_model]:.2f}",
            "Interpretation": "Uncertain estimate from six observed records",
        },
        {
            "Subject": "Heidi", "Missing field": "Salary", "Predictor": "Join Date",
            "Linear result": f"{heidi_salary_predictions['Linear']:.2f}",
            "Linear LOOCV RMSE": f"{heidi_salary_scores['Linear']['RMSE']:.2f}",
            "Polynomial result": f"{heidi_salary_predictions['Polynomial (degree 2)']:.2f}",
            "Polynomial LOOCV RMSE": f"{heidi_salary_scores['Polynomial (degree 2)']['RMSE']:.2f}",
            "Selected model": selected_heidi_salary_model,
            "Selected result": f"{heidi_salary_predictions[selected_heidi_salary_model]:.2f}",
            "Interpretation": "Uncertain estimate based only on Joining Date",
        },
        {
            "Subject": "Heidi", "Missing field": "Age", "Predictor": "Join Date",
            "Linear result": f"{heidi_age_predictions['Linear']:.2f}",
            "Linear LOOCV RMSE": f"{heidi_age_scores['Linear']['RMSE']:.2f} years",
            "Polynomial result": f"{heidi_age_predictions['Polynomial (degree 2)']:.2f}",
            "Polynomial LOOCV RMSE": f"{heidi_age_scores['Polynomial (degree 2)']['RMSE']:.2f} years",
            "Selected model": selected_heidi_age_model,
            "Selected result": str(round(heidi_age_predictions[selected_heidi_age_model])),
            "Interpretation": "Uncertain estimate rounded to a whole year",
        },
    ]
    save_arranged_outputs(rows, experiments)
    plot_created = create_plot(salary, net_worth, models, david_salary, predictions)
    date_plot_created = create_date_plot(
        date_salary,
        joining_dates,
        date_models,
        charlie_salary,
        date_predictions,
    )
    country_plot_created = create_country_plot(
        country_salary,
        encoded_country,
        country_models,
        grace_salary,
        country_predictions,
    )
    heidi_plot_created = create_heidi_plot(
        heidi_dates,
        heidi_net_worth,
        heidi_models,
        heidi_date,
        heidi_predictions,
    )
    heidi_salary_plot_created = create_heidi_salary_plot(
        heidi_salary_dates,
        observed_salaries,
        heidi_salary_models,
        heidi_date,
        heidi_salary_predictions,
    )
    heidi_age_plot_created = create_heidi_age_plot(
        heidi_age_dates,
        observed_ages,
        heidi_age_models,
        heidi_date,
        heidi_age_predictions,
    )

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
    print("Remaining unsupported fields stay blank; model estimates are stored separately.")
    print("\nJOINING DATE PREDICTION FOR CHARLIE")
    print(f"Training observations: {len(date_complete)}")
    print("Target: Joining Date | Predictor: Salary")
    print("Evaluation: leave-one-out cross-validation; errors are measured in days\n")
    for name in specifications:
        print(name)
        print(f"  LOOCV MAE:  {date_scores[name]['MAE']:,.2f} days")
        print(f"  LOOCV RMSE: {date_scores[name]['RMSE']:,.2f} days")
        print(f"  LOOCV R2:   {date_scores[name]['R2']:.3f}")
        print(f"  Charlie prediction: {ordinal_to_iso(date_predictions[name])}\n")
    selected_date = ordinal_to_iso(date_predictions[selected_date_model])
    print(f"Selected date model: {selected_date_model} (lowest LOOCV RMSE)")
    print(f"Predicted missing Joining Date for Charlie: {selected_date}")
    print("Warning: this is an uncertain estimate, not Charlie's known Joining Date.")
    print("\nCOUNTRY ESTIMATION FOR GRACE (NZ=0, AUS=1)")
    print(f"Training observations: {len(country_complete)}")
    print("Target: encoded Country | Predictor: Salary | Threshold: 0.5\n")
    for name in specifications:
        country = "AUS" if country_predictions[name] >= 0.5 else "NZ"
        print(name)
        print(f"  LOOCV MAE:      {country_scores[name]['MAE']:.3f}")
        print(f"  LOOCV RMSE:     {country_scores[name]['RMSE']:.3f}")
        print(f"  LOOCV R2:       {country_scores[name]['R2']:.3f}")
        print(f"  Grace score:    {country_predictions[name]:.3f}")
        print(f"  Grace category: {country}\n")
    selected_country = "AUS" if country_predictions[selected_country_model] >= 0.5 else "NZ"
    print(f"Selected Country model: {selected_country_model} (lowest LOOCV RMSE)")
    print(f"Selected Country estimate for Grace: {selected_country}")
    print("Warning: ordinary regression is only an experimental workaround for a category.")
    print("\nNET WORTH PREDICTION FOR HEIDI")
    print(f"Training observations: {len(heidi_complete)}")
    print("Target: Net worth | Predictor: Joining Date\n")
    for name in specifications:
        print(name)
        print(f"  LOOCV MAE:  {heidi_scores[name]['MAE']:,.2f}")
        print(f"  LOOCV RMSE: {heidi_scores[name]['RMSE']:,.2f}")
        print(f"  LOOCV R2:   {heidi_scores[name]['R2']:.3f}")
        print(f"  Heidi prediction: {heidi_predictions[name]:,.2f}\n")
    print(f"Selected Heidi model: {selected_heidi_model} (lowest LOOCV RMSE)")
    print(f"Selected Net worth estimate for Heidi: {heidi_predictions[selected_heidi_model]:,.2f}")
    print("Warning: this is an uncertain estimate based on only six observed records.")
    print("\nSALARY PREDICTION FOR HEIDI")
    print(f"Training observations: {len(heidi_salary_complete)}")
    print("Target: Salary | Predictor: Joining Date\n")
    for name in specifications:
        print(name)
        print(f"  LOOCV MAE:  {heidi_salary_scores[name]['MAE']:,.2f}")
        print(f"  LOOCV RMSE: {heidi_salary_scores[name]['RMSE']:,.2f}")
        print(f"  LOOCV R2:   {heidi_salary_scores[name]['R2']:.3f}")
        print(f"  Heidi prediction: {heidi_salary_predictions[name]:,.2f}\n")
    print(f"Selected Heidi Salary model: {selected_heidi_salary_model} (lowest LOOCV RMSE)")
    print(f"Selected Salary estimate for Heidi: {heidi_salary_predictions[selected_heidi_salary_model]:,.2f}")
    print("Warning: this is an uncertain estimate based only on Joining Date.")
    print("\nAGE PREDICTION FOR HEIDI")
    print(f"Training observations: {len(heidi_age_complete)}")
    print("Target: Age | Predictor: Joining Date\n")
    for name in specifications:
        print(name)
        print(f"  LOOCV MAE:  {heidi_age_scores[name]['MAE']:.2f} years")
        print(f"  LOOCV RMSE: {heidi_age_scores[name]['RMSE']:.2f} years")
        print(f"  LOOCV R2:   {heidi_age_scores[name]['R2']:.3f}")
        print(f"  Heidi prediction: {heidi_age_predictions[name]:.2f} years\n")
    selected_age = round(heidi_age_predictions[selected_heidi_age_model])
    print(f"Selected Heidi Age model: {selected_heidi_age_model} (lowest LOOCV RMSE)")
    print(f"Selected Age estimate for Heidi: {selected_age} years")
    print("Warning: this is an uncertain estimate based only on Joining Date.")
    print(f"\nPrediction comparison saved to: {OUTPUT}")
    print(f"Completed presentation dataset saved to: {COMPLETED}")
    if plot_created:
        print(f"Comparison graph saved to: {PLOT}")
    if date_plot_created:
        print(f"Joining Date graph saved to: {DATE_PLOT}")
    if country_plot_created:
        print(f"Country graph saved to: {COUNTRY_PLOT}")
    if heidi_plot_created:
        print(f"Heidi Net worth graph saved to: {HEIDI_PLOT}")
    if heidi_salary_plot_created:
        print(f"Heidi Salary graph saved to: {HEIDI_SALARY_PLOT}")
    if heidi_age_plot_created:
        print(f"Heidi Age graph saved to: {HEIDI_AGE_PLOT}")


if __name__ == "__main__":
    main()
