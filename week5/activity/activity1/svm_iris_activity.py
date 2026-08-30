"""Week 5 Activity 1: classify the three Iris species with SVM models."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ACTIVITY_DIR = Path(__file__).resolve().parent
DATA_FILE = ACTIVITY_DIR.parent / "iris" / "iris.data"
OUTPUT_DIR = ACTIVITY_DIR / "outputs"
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


def load_iris_csv(path: Path) -> tuple[list[list[float]], list[str]]:
    """Load the supplied UCI Iris file without changing the source data."""
    features: list[list[float]] = []
    labels: list[str] = []

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(f"Expected 5 columns, found {len(row)}: {row}")
            features.append([float(value) for value in row[:4]])
            labels.append(row[4].removeprefix("Iris-"))

    if not features:
        raise ValueError(f"No observations found in {path}")
    return features, labels


def save_scatter_plot(features: list[list[float]], labels: list[str]) -> None:
    """Plot the two most class-informative features from the dataset notes."""
    colours = {"setosa": "#2563EB", "versicolor": "#F59E0B", "virginica": "#DC2626"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for species, colour in colours.items():
        rows = [row for row, label in zip(features, labels) if label == species]
        ax.scatter(
            [row[2] for row in rows],
            [row[3] for row in rows],
            label=species.title(),
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
        )

    ax.set(title="Iris species by petal measurements", xlabel=FEATURE_NAMES[2], ylabel=FEATURE_NAMES[3])
    ax.legend(title="Species")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "iris_scatter_plot.png", dpi=180)
    plt.close(fig)


def make_pipeline() -> Pipeline:
    """Scale features inside each fold, then fit an SVM classifier."""
    return Pipeline([("scale", StandardScaler()), ("svc", SVC())])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, labels = load_iris_csv(DATA_FILE)

    print(f"Dataset: {DATA_FILE}")
    print(f"Samples: {len(features)} | Features: {len(features[0])}")
    print("Class counts:", {name: labels.count(name) for name in sorted(set(labels))})
    print("Missing values: 0 (validated while parsing numeric fields)\n")
    save_scatter_plot(features, labels)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    # Teacher-style controlled experiment: change only the kernel. All models
    # retain SVC's default C=1 and gamma='scale' settings.
    kernels = ["linear", "rbf", "poly", "sigmoid"]
    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions_by_kernel: dict[str, list[str]] = {}
    for kernel in kernels:
        model = Pipeline([("scale", StandardScaler()), ("svc", SVC(kernel=kernel))])
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")
        model.fit(x_train, y_train)  # Required SVM training step.
        predictions = model.predict(x_test)
        results.append(
            {
                "kernel": kernel,
                "cv_accuracy": cv_scores.mean(),
                "test_accuracy": accuracy_score(y_test, predictions),
                "parameters": "C=1, gamma=scale",
            }
        )
        fitted_models[kernel] = model
        predictions_by_kernel[kernel] = list(predictions)

    # For this classroom experiment, compare the unchanged models on the same
    # held-out test set. Cross-validation is shown as an additional reliability check.
    results.sort(key=lambda row: (row["test_accuracy"], row["cv_accuracy"]), reverse=True)
    best_kernel = str(results[0]["kernel"])
    best_model = fitted_models[best_kernel]
    best_predictions = predictions_by_kernel[best_kernel]

    print("Kernel comparison")
    print("kernel    CV accuracy   test accuracy   fixed parameters")
    for row in results:
        print(
            f"{row['kernel']:<9} {row['cv_accuracy']:.3f}         "
            f"{row['test_accuracy']:.3f}           {row['parameters']}"
        )
    print(f"\nRecommended kernel: {best_kernel.upper()}")
    print("\nClassification report (recommended model):")
    print(classification_report(y_test, best_predictions, digits=3))

    with (OUTPUT_DIR / "kernel_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    classes = best_model.named_steps["svc"].classes_
    matrix = confusion_matrix(y_test, best_predictions, labels=classes)
    display = ConfusionMatrixDisplay(matrix, display_labels=[name.title() for name in classes])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(f"Confusion matrix: {best_kernel.upper()} SVM")
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(display.figure_)


if __name__ == "__main__":
    main()
