# Week 4 Activity 1.1: Happiness Dashboard

This project analyses a cleaned 20-country Happiness dataset using Pandas,
SQLite, Matplotlib, and Plotly. It identifies the three happiest countries,
summarises Freedom for the least-happy country, and explores relationships
between Happiness and the other features.

## Project files

- `happiness_dashboard.py` - data validation, aggregation, SQL query, analysis,
  and visualisation code.
- `world_happiness_dataset.csv` - cleaned source dataset.
- `dashboard_report.md` - detailed methodology, results, visual explanations,
  findings, and limitations.
- `requirements.txt` - required Python packages.
- `outputs/` - generated static and interactive visualisations.

## Installation

Create or activate a Python virtual environment, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the project

From the repository root:

```powershell
python .\week4\activity\activity1.1\happiness_dashboard.py
```

Or from this activity folder:

```powershell
python .\happiness_dashboard.py
```

The script generates all output files and opens the unified interactive
dashboard in the default browser.

## Dashboard and report

- [Open the interactive Happiness dashboard](outputs/happiness_dashboard.html)
- [Read the complete dashboard report](dashboard_report.md)

![Three happiest countries](outputs/matplotlib_top_three.png)

## Main results

| Rank | Country | Happiness score |
|---:|---|---:|
| 1 | Canada | 7.34 |
| 2 | Brazil | 6.98 |
| 3 | Finland | 6.67 |

South Africa has the lowest Happiness score (3.53) and a Freedom score of 0.90.
See the dashboard report for the complete analysis and interpretation.

## Technologies

- Python
- Pandas
- SQLite
- Matplotlib
- Plotly
