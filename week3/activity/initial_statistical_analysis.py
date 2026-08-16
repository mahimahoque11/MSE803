
from __future__ import annotations

import csv
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("Sample_dataset.csv")
CLEANED = Path(__file__).with_name("cleaned_dataset.csv")
NUMERIC_COLUMNS = ("Age", "Net worth", "Salary")


def missing(value: str | None) -> bool:
    return value is None or not value.strip()


def clean_number(value: str, replacements: dict[str, str] | None = None) -> float | None:
    """Convert a numeric field to float, leaving unknown values as None."""
    if missing(value):
        return None
    text = value.strip().lower().replace(",", "")
    if replacements:
        text = replacements.get(text, text)
    try:
        return float(text)
    except ValueError:
        return None


def clean_date(value: str) -> str | None:
    """Return ISO date text; interpret 2019-13-01 as YYYY-DD-MM."""
    if missing(value):
        return None
    for pattern in ("%d/%m/%Y", "%Y-%m-%d", "%Y-%d-%m"):
        try:
            return datetime.strptime(value.strip(), pattern).date().isoformat()
        except ValueError:
            pass
    return None


def load_and_clean() -> tuple[list[dict], dict[str, int]]:
    with SOURCE.open(encoding="utf-8-sig", newline="") as file:
        raw_rows = list(csv.DictReader(file))

    issues = {
        "original rows": len(raw_rows),
        "duplicate rows consolidated": 0,
        "unparseable values converted to missing": 0,
    }
    cleaned_rows: list[dict] = []

    for raw in raw_rows:
        age = clean_number(raw["Age"], {"thirty-eight": "38"})
        salary = clean_number(raw["Salary"], {"sixty five thousand": "65000"})
        net_worth = clean_number(raw["Net worth"])
        if not missing(raw["Age"]) and age is None:
            issues["unparseable values converted to missing"] += 1
        if not missing(raw["Salary"]) and salary is None:
            issues["unparseable values converted to missing"] += 1

        country = raw["Country"].strip().upper() or None
        if country == "AU":
            country = "AUS"

        row = {
            "ID": clean_number(raw["ID"]),
            "Name": raw["Name"].strip() or None,
            "Age": age,
            "Net worth": net_worth,
            "Country": country,
            "Salary": salary,
            "Join Date": clean_date(raw["Join Date"]),
        }


        existing = next((r for r in cleaned_rows if row["ID"] is not None and r["ID"] == row["ID"]), None)
        if existing:
            issues["duplicate rows consolidated"] += 1
            for key, value in row.items():
                if existing[key] is None and value is not None:
                    existing[key] = value
        else:
            cleaned_rows.append(row)

    return cleaned_rows, issues


def save_cleaned(rows: list[dict]) -> bool:
    """Save cleaned data, but do not stop the analysis if Excel locks the file."""
    try:
        with CLEANED.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        print(f"WARNING: Could not update {CLEANED.name}.")
        print("Close the file in Excel and run the program again to update it.\n")
        return False
    return True


def numeric_summary(rows: list[dict], column: str) -> dict[str, float | int]:
    values = [float(row[column]) for row in rows if row[column] is not None]
    quartiles = np.percentile(values, [25, 75])
    return {
        "count": len(values),
        "missing": len(rows) - len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "population variance (N)": statistics.pvariance(values),
        "population standard deviation (N)": statistics.pstdev(values),
        "sample variance (n-1)": statistics.variance(values),
        "sample standard deviation (n-1)": statistics.stdev(values),
        "Q1": float(quartiles[0]),
        "Q3": float(quartiles[1]),
        "IQR": float(quartiles[1] - quartiles[0]),
    }


def print_analysis(rows: list[dict], issues: dict[str, int]) -> None:
    print("INITIAL STATISTICAL ANALYSIS")
    print("=" * 50)
    print("\nDATA QUALITY AND CLEANING")
    for label, value in issues.items():
        print(f"  {label}: {value}")
    print(f"  cleaned rows: {len(rows)}")
    for column in rows[0]:
        count = sum(row[column] is None for row in rows)
        print(f"  missing {column}: {count}")

    print("\nNUMERIC DESCRIPTIVE STATISTICS")
    for column in NUMERIC_COLUMNS:
        print(f"\n{column}")
        for metric, value in numeric_summary(rows, column).items():
            shown = f"{value:,.2f}" if isinstance(value, float) else str(value)
            print(f"  {metric:<26} {shown}")

    countries = [row["Country"] for row in rows if row["Country"] is not None]
    country_counts = Counter(countries)
    print("\nCATEGORICAL SUMMARY")
    print(f"  Country frequencies: {dict(country_counts)}")
    print(f"  Country mode: {statistics.mode(countries)}")

    print("\nCOVARIANCE AND CORRELATION (pairwise complete rows)")
    for left, right in (("Age", "Salary"), ("Age", "Net worth"), ("Net worth", "Salary")):
        pairs = [(row[left], row[right]) for row in rows if row[left] is not None and row[right] is not None]
        x, y = zip(*pairs)
        population_covariance = float(np.cov(x, y, ddof=0)[0, 1])
        sample_covariance = statistics.covariance(x, y)
        correlation = float(np.corrcoef(x, y)[0, 1])
        print(f"\n  {left} vs {right} (complete pairs = {len(pairs)})")
        print(f"    population covariance (N): {population_covariance:,.2f}")
        print(f"    sample covariance (n-1):   {sample_covariance:,.2f}")
        print(f"    Pearson correlation:       {correlation:.3f}")

    print("\nINTERPRETATION GUIDE")
    print("  Count/missing: show how much evidence each result uses.")
    print("  Mean: arithmetic average; sensitive to unusually high or low values.")
    print("  Median: middle value; a robust description of a typical observation.")
    print("  Min/max/range: endpoints and total span; very sensitive to extremes.")
    print("  Population variance/SD: spread when the data are the whole population; divides by N.")
    print("  Sample variance/SD: estimated population spread from a sample; divides by n-1.")
    print("  Q1/Q3: 25th/75th percentiles; the middle half lies between them.")
    print("  IQR: Q3-Q1; robust spread of the middle 50% of observations.")
    print("  Mode/frequency: most common category and counts for each category.")
    print("  Covariance: direction of joint movement; its size depends on measurement units.")
    print("  Pearson r: standardised linear association from -1 to +1; it does not prove causation.")


def main() -> None:
    rows, issues = load_and_clean()
    file_saved = save_cleaned(rows)
    print_analysis(rows, issues)
    if file_saved:
        print(f"\nCleaned data saved to: {CLEANED}")


if __name__ == "__main__":
    main()
