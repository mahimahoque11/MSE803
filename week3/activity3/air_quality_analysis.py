from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ACTIVITY_DIR = Path(__file__).resolve().parent
DATA_FILE = ACTIVITY_DIR / "beijing_air_quality_combined.csv"
OUTPUT_DIR = ACTIVITY_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_FILE)

print("First five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset dimensions:")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# Combine the separate date and time columns
df["datetime"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

# Arrange every station's records chronologically
df = df.sort_values(["station", "datetime"])

print("\nDate range:")
print(f"Start: {df['datetime'].min()}")
print(f"End: {df['datetime'].max()}")

print("\nFirst five datetime values:")
print(df[["station", "datetime"]].head())

missing_report = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing Percentage": df.isna().mean() * 100
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

print("\nMissing-value report:")
print(missing_report.round(2))

# Numerical columns containing measurements
numeric_columns = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

# Interpolate within each station
df[numeric_columns] = (
    df.groupby("station")[numeric_columns]
      .transform(
          lambda column: column.interpolate(
              method="linear",
              limit_direction="both"
          )
      )
)

df["wd"] = (
    df.groupby("station")["wd"]
      .transform(
          lambda column: column.fillna(column.mode().iloc[0])
      )
)

remaining_missing = df.isna().sum()

print("\nMissing values after cleaning:")
print(remaining_missing[remaining_missing > 0])

print(
    f"\nTotal missing values remaining: "
    f"{remaining_missing.sum():,}"
)

# Calculate basic descriptive statistics
statistics = df[numeric_columns].agg(
    ["mean", "median", "min", "max", "std"]
).T

statistics = statistics.round(2)

print("\nBasic descriptive statistics:")
print(statistics)

statistics_file = OUTPUT_DIR / "basic_statistics.csv"
statistics.to_csv(
    statistics_file,
    index_label="Measurement"
)

print(f"\nBasic statistics saved to: {statistics_file}")

pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

station_averages = (
    df.groupby("station")[pollutants]
      .mean()
      .round(2)
)

print("\nAverage pollution levels by station:")
print(station_averages)

station_averages_file = (
    OUTPUT_DIR / "station_pollution_averages.csv"
)

station_averages.to_csv(
    station_averages_file,
    index_label="Station"
)

highest_pm25_station = station_averages["PM2.5"].idxmax()
highest_pm25_value = station_averages["PM2.5"].max()

print(
    f"\nStation with the highest average PM2.5: "
    f"{highest_pm25_station} ({highest_pm25_value:.2f})"
)

print(
    f"\nStation averages saved to: "
    f"{station_averages_file}"
)

# Create a histogram showing the PM2.5 distribution
plt.figure(figsize=(9, 5))

plt.hist(
    df["PM2.5"],
    bins=50,
    color="steelblue",
    edgecolor="black"
)

plt.title("Distribution of PM2.5 Measurements")
plt.xlabel("PM2.5 Concentration (µg/m³)")
plt.ylabel("Frequency")
plt.tight_layout()

histogram_file = OUTPUT_DIR / "pm25_histogram.png"
plt.savefig(histogram_file, dpi=200)
plt.close()

print(f"\nPM2.5 histogram saved to: {histogram_file}")

# Calculate monthly average PM2.5 across all stations
monthly_pm25 = (
    df.set_index("datetime")["PM2.5"]
      .resample("MS")
      .mean()
)

plt.figure(figsize=(11, 5))

plt.plot(
    monthly_pm25.index,
    monthly_pm25.values,
    color="darkred",
    linewidth=1.8
)

plt.title("Monthly Average PM2.5 Over Time")
plt.xlabel("Date")
plt.ylabel("Average PM2.5 Concentration (µg/m³)")
plt.grid(alpha=0.3)
plt.tight_layout()

time_plot_file = OUTPUT_DIR / "pm25_over_time.png"
plt.savefig(time_plot_file, dpi=200)
plt.close()

print(f"PM2.5 time-series graph saved to: {time_plot_file}")

# Create boxplots for the pollution variables
plt.figure(figsize=(10, 6))

df[pollutants].boxplot(
    showfliers=False,
    grid=False
)

plt.yscale("log")
plt.title("Distribution of Air Pollutants")
plt.xlabel("Pollutant")
plt.ylabel("Concentration — logarithmic scale")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

boxplot_file = OUTPUT_DIR / "pollutant_boxplot.png"
plt.savefig(boxplot_file, dpi=200)
plt.close()

print(f"Pollutant boxplot saved to: {boxplot_file}")

# Select pollution and weather measurements for correlation
correlation_columns = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

correlation_matrix = (
    df[correlation_columns]
    .corr()
    .round(3)
)

correlation_file = OUTPUT_DIR / "correlation_matrix.csv"

correlation_matrix.to_csv(
    correlation_file,
    index_label="Variable"
)

print("\nCorrelation matrix:")
print(correlation_matrix)

# Extract and rank correlations with PM2.5
pm25_correlations = (
    correlation_matrix["PM2.5"]
    .drop("PM2.5")
    .sort_values(
        key=lambda values: values.abs(),
        ascending=False
    )
)

print("\nVariables correlated with PM2.5:")
print(pm25_correlations)

strongest_variable = pm25_correlations.abs().idxmax()
strongest_value = pm25_correlations[strongest_variable]

temperature_correlation = correlation_matrix.loc[
    "TEMP", "PM2.5"
]

print(
    f"\nVariable most strongly correlated with PM2.5: "
    f"{strongest_variable} ({strongest_value:.3f})"
)

print(
    f"Correlation between temperature and PM2.5: "
    f"{temperature_correlation:.3f}"
)

print(f"Correlation matrix saved to: {correlation_file}")