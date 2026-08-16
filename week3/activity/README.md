# Week 3 – Activity 1: Initial statistical analysis

This solution cleans `Sample_dataset.csv`, produces descriptive statistics, and
explains each metric in the console output. It uses Python's `statistics` module
and NumPy's statistical functions, as requested in the activity.

## Run

From this folder:

```powershell
python initial_statistical_analysis.py
```

The program prints the analysis and creates `cleaned_dataset.csv` and
`scatter_plots.png`.

If Matplotlib is not installed, install it before running the program:

```powershell
python -m pip install matplotlib
```

## Cleaning decisions

- The two records with ID 2 are consolidated because they represent Bob and
  contain complementary missing fields.
- `thirty-eight` and `sixty five thousand` are converted to 38 and 65,000.
- Thousands separators are removed from numeric values.
- `AU` is standardised to `AUS`.
- Dates are converted to ISO `YYYY-MM-DD`. The invalid-looking `2019-13-01` is
  interpreted as `YYYY-DD-MM`, giving `2019-01-13`.
- Unknown data stays missing. IDs, names, and other values are not invented.
- Each numeric metric excludes missing values only from that metric. Each
  correlation uses rows where both variables are present (pairwise complete).

## Metric interpretation

- **Count and missing count** reveal the amount of evidence behind a result.
- **Mean** is the arithmetic average and is sensitive to extreme values.
- **Median** is the middle ordered value and is more resistant to extremes.
- **Minimum, maximum, and range** show endpoints and the full spread.
- **Sample variance** measures squared dispersion using `n - 1`.
- **Sample standard deviation** expresses typical spread in the variable's
  original units.
- **Q1, Q3, and IQR** describe the middle 50% and are relatively robust to
  extremes.
- **Frequency and mode** summarise categories; the mode is most frequent.
- **Pearson correlation (`r`)** measures linear association from -1 to +1.
  Association does not establish causation, and the small sample requires care.

## Results from the cleaned dataset

There are 9 unique subjects after consolidating Bob's duplicate record. Because
some observations are missing, each calculation uses the available values for
that variable. Population formulas divide by `N`; sample variance, sample
standard deviation, and sample covariance divide by `n - 1`.

| Variable | Valid subjects | Mean | Population variance | Population SD | Sample variance | Sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Age | 8 | 30.75 | 35.44 | 5.95 | 40.50 | 6.36 |
| Net worth | 7 | 38,571.43 | 171,959,183.67 | 13,113.32 | 200,619,047.62 | 14,164.01 |
| Salary | 8 | 62,625.00 | 27,984,375.00 | 5,290.03 | 31,982,142.86 | 5,655.28 |

Covariance uses pairwise-complete observations: a row is included only when
both variables in that pair are present.

| Variable pair | Complete pairs | Population covariance | Sample covariance (`n - 1`) |
|---|---:|---:|---:|
| Age and Net worth | 7 | 35,877.55 | 41,857.14 |
| Age and Salary | 8 | 19,781.25 | 22,607.14 |
| Net worth and Salary | 7 | 10,510,204.08 | 12,261,904.76 |

All three covariance results are positive, meaning the paired variables tend
to increase together in this dataset. Covariance magnitude depends on the
variables' units, so it should not be interpreted as a standardised strength.
The sample is also very small, so the results should be interpreted cautiously.

## Scatter-plot visualisation

The program creates scatter plots for Age–Salary, Age–Net worth, and Net
worth–Salary. Each dot represents one subject with both required values. The
red fitted line shows the overall direction, while the title reports Pearson's
correlation and the number of complete observations.

An upward line supports a positive relationship and a downward line supports a
negative relationship. Points far from the others may indicate outliers. These
plots are exploratory: the dataset is too small to support strong population
conclusions, and association does not prove causation.

## References

- [NumPy statistical routines](https://numpy.org/doc/2.5/reference/routines.statistics.html)
- [Python `statistics` module](https://docs.python.org/3/library/statistics.html)
