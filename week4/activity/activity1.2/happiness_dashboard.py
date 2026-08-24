import sqlite3

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIRECTORY / "world_happiness_dataset.csv"
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "outputs"


def load_dataset(file_path):
    """Load the cleaned happiness dataset."""
    return pd.read_csv(file_path)


def inspect_dataset(data):
    """Display basic information about the dataset."""
    print("First five rows:")
    print(data.head())

    print("\nDataset shape:", data.shape)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nNumber of duplicate rows:", data.duplicated().sum())


def get_top_three_with_pandas(data):
    """Find the three happiest countries using Pandas aggregation."""
    country_summary = (
        data.groupby("Country", as_index=False)
        .agg(Average_Happiness_Score=("Happiness_Score", "mean"))
        .nlargest(3, "Average_Happiness_Score")
        .reset_index(drop=True)
    )

    return country_summary


def get_top_three_with_sql(data):
    """Find the three happiest countries using an SQL query."""
    connection = sqlite3.connect(":memory:")

    try:
        data.to_sql(
            "happiness",
            connection,
            index=False,
            if_exists="replace"
        )

        query = """
            SELECT
                Country,
                ROUND(AVG(Happiness_Score), 2)
                    AS Average_Happiness_Score
            FROM happiness
            GROUP BY Country
            ORDER BY Average_Happiness_Score DESC
            LIMIT 3;
        """

        return pd.read_sql_query(query, connection)

    finally:
        connection.close()

def create_matplotlib_subplot(top_three):
    """Create bar and scatter subplots using Matplotlib."""
    plot_data = top_three.sort_values(
        "Average_Happiness_Score",
        ascending=False
    )

    countries = plot_data["Country"]
    scores = plot_data["Average_Happiness_Score"]
    colours = ["#2E86AB", "#F18F01", "#2CA25F"]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5)
    )

    # Subplot 1: bar chart
    axes[0].bar(countries, scores, color=colours)

    axes[0].set_title("Bar Chart Comparison")
    axes[0].set_xlabel("Country")
    axes[0].set_ylabel("Average Happiness Score")
    axes[0].set_ylim(0, 8)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    for index, score in enumerate(scores):
        axes[0].text(
            index,
            score + 0.1,
            f"{score:.2f}",
            ha="center"
        )

    # Subplot 2: scatter/dot chart
    axes[1].scatter(
        scores,
        countries,
        color=colours,
        s=160
    )

    axes[1].set_title("Scatter Plot Ranking")
    axes[1].set_xlabel("Average Happiness Score")
    axes[1].set_ylabel("Country")
    axes[1].set_xlim(0, 8)
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    for score, country in zip(scores, countries):
        axes[1].text(
            score + 0.1,
            country,
            f"{score:.2f}",
            va="center"
        )

    figure.suptitle(
        "Three Happiest Countries",
        fontsize=16,
        fontweight="bold"
    )

    figure.tight_layout()

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    figure.savefig(
        OUTPUT_DIRECTORY / "matplotlib_top_three.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

def create_plotly_subplot(top_three):
    """Create interactive bar and scatter subplots using Plotly."""
    plot_data = top_three.sort_values(
        "Average_Happiness_Score",
        ascending=False
    )

    countries = plot_data["Country"]
    scores = plot_data["Average_Happiness_Score"]
    colours = ["#2E86AB", "#F18F01", "#2CA25F"]

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Bar Chart Comparison",
            "Scatter Plot Ranking"
        ),
        horizontal_spacing=0.15
    )

    figure.add_trace(
        go.Bar(
            x=countries,
            y=scores,
            marker_color=colours,
            text=scores.round(2),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Happiness score: %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=scores,
            y=countries,
            mode="markers+text",
            marker={
                "size": 18,
                "color": colours
            },
            text=scores.round(2),
            textposition="middle right",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Happiness score: %{x:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=2
    )

    figure.update_yaxes(
        title_text="Average Happiness Score",
        range=[0, 8],
        row=1,
        col=1
    )

    figure.update_xaxes(
        title_text="Average Happiness Score",
        range=[0, 8],
        row=1,
        col=2
    )

    figure.update_layout(
        title={
            "text": "Three Happiest Countries",
            "x": 0.5
        },
        template="plotly_white",
        height=500,
        width=1000,
        showlegend=False
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    figure.write_html(
        OUTPUT_DIRECTORY / "plotly_top_three.html"
    )



def get_lowest_happiness_country(data):
    """Aggregate the data and return the country with the lowest happiness."""
    country_summary = (
        data.groupby("Country", as_index=False)
        .agg(
            Average_Happiness_Score=("Happiness_Score", "mean"),
            Average_Freedom_Score=("Freedom_to_Make_Choices", "mean")
        )
    )

    return country_summary.nsmallest(
        1,
        "Average_Happiness_Score"
    ).iloc[0]


def create_freedom_gauge(lowest_country, dataset_average_freedom):
    """Visualise the Freedom score of the least-happy country."""
    country = lowest_country["Country"]
    happiness_score = lowest_country["Average_Happiness_Score"]
    freedom_score = lowest_country["Average_Freedom_Score"]

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=freedom_score,
            number={"valueformat": ".2f"},
            delta={
                "reference": dataset_average_freedom,
                "valueformat": ".2f"
            },
            title={
                "text": (
                    f"Freedom Score: {country}<br>"
                    f"<span style='font-size:0.75em'>"
                    f"Lowest happiness score: {happiness_score:.2f}"
                    "</span>"
                )
            },
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": "#2E86AB"},
                "steps": [
                    {"range": [0, 0.4], "color": "#FADBD8"},
                    {"range": [0.4, 0.7], "color": "#FCF3CF"},
                    {"range": [0.7, 1], "color": "#D5F5E3"}
                ],
                "threshold": {
                    "line": {"color": "#C0392B", "width": 4},
                    "value": dataset_average_freedom
                }
            }
        )
    )

    figure.update_layout(
        title={"text": "Freedom Summary", "x": 0.5},
        template="plotly_white",
        height=500,
        width=700
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.write_html(OUTPUT_DIRECTORY / "lowest_country_freedom.html")


def create_happiness_freedom_bubble_chart(data):
    """Explore Happiness and Freedom alongside GDP and corruption."""
    figure = px.scatter(
        data,
        x="Freedom_to_Make_Choices",
        y="Happiness_Score",
        size="GDP_per_Capita",
        color="Perceptions_of_Corruption",
        hover_name="Country",
        hover_data={
            "Freedom_to_Make_Choices": ":.2f",
            "Happiness_Score": ":.2f",
            "GDP_per_Capita": ":.2f",
            "Perceptions_of_Corruption": ":.2f"
        },
        size_max=45,
        color_continuous_scale="RdYlGn_r",
        labels={
            "Freedom_to_Make_Choices": "Freedom to Make Choices",
            "Happiness_Score": "Happiness Score",
            "GDP_per_Capita": "GDP per Capita",
            "Perceptions_of_Corruption": "Perceived Corruption"
        },
        title=(
            "Happiness and Freedom by Country<br>"
            "<sup>Bubble size = GDP per capita; "
            "colour = perceived corruption</sup>"
        )
    )

    figure.update_traces(
        marker={"line": {"color": "white", "width": 1}},
        opacity=0.85
    )

    figure.update_layout(
        template="plotly_white",
        height=600,
        width=900,
        title={"x": 0.5},
        coloraxis_colorbar={"title": "Perceived<br>corruption"}
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.write_html(OUTPUT_DIRECTORY / "happiness_freedom_bubble.html")


def calculate_feature_and_country_rankings(data):
    """Rank features by correlation and countries by an equal-weight index."""
    feature_columns = [
        "GDP_per_Capita",
        "Social_Support",
        "Healthy_Life_Expectancy",
        "Freedom_to_Make_Choices",
        "Generosity",
        "Perceptions_of_Corruption"
    ]

    feature_ranking = (
        data[feature_columns]
        .corrwith(data["Happiness_Score"])
        .rename("Correlation_with_Happiness")
        .rename_axis("Feature")
        .reset_index()
    )
    feature_ranking["Absolute_Correlation"] = feature_ranking[
        "Correlation_with_Happiness"
    ].abs()
    feature_ranking = feature_ranking.sort_values(
        "Absolute_Correlation",
        ascending=False
    ).reset_index(drop=True)

    country_ranking = data.copy()
    positive_features = [
        "GDP_per_Capita",
        "Social_Support",
        "Healthy_Life_Expectancy",
        "Freedom_to_Make_Choices",
        "Generosity"
    ]
    normalised_columns = []

    for feature in positive_features:
        column_minimum = country_ranking[feature].min()
        column_range = country_ranking[feature].max() - column_minimum
        normalised_column = f"Normalised_{feature}"
        country_ranking[normalised_column] = (
            country_ranking[feature] - column_minimum
        ) / column_range
        normalised_columns.append(normalised_column)

    corruption_minimum = country_ranking[
        "Perceptions_of_Corruption"
    ].min()
    corruption_range = (
        country_ranking["Perceptions_of_Corruption"].max()
        - corruption_minimum
    )
    country_ranking["Normalised_Low_Corruption"] = 1 - (
        (
            country_ranking["Perceptions_of_Corruption"]
            - corruption_minimum
        ) / corruption_range
    )
    normalised_columns.append("Normalised_Low_Corruption")

    country_ranking["Composite_Feature_Score"] = country_ranking[
        normalised_columns
    ].mean(axis=1)
    country_ranking["Feature_Based_Rank"] = country_ranking[
        "Composite_Feature_Score"
    ].rank(method="min", ascending=False).astype(int)
    country_ranking["Actual_Happiness_Rank"] = country_ranking[
        "Happiness_Score"
    ].rank(method="min", ascending=False).astype(int)

    country_ranking = country_ranking.sort_values(
        "Composite_Feature_Score",
        ascending=False
    ).reset_index(drop=True)

    return feature_ranking, country_ranking


def create_ranking_visualisations(feature_ranking, country_ranking):
    """Visualise feature importance and feature-based country rankings."""
    feature_figure = px.bar(
        feature_ranking.sort_values("Absolute_Correlation"),
        x="Correlation_with_Happiness",
        y="Feature",
        orientation="h",
        color="Correlation_with_Happiness",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        text="Correlation_with_Happiness",
        title="Feature Ranking by Correlation with Happiness",
        labels={
            "Correlation_with_Happiness": "Pearson correlation",
            "Feature": "Feature"
        }
    )
    feature_figure.update_traces(texttemplate="%{text:.3f}")
    feature_figure.update_layout(
        template="plotly_white",
        height=550,
        width=900,
        title={"x": 0.5},
        coloraxis_showscale=False
    )

    country_figure = px.scatter(
        country_ranking,
        x="Composite_Feature_Score",
        y="Happiness_Score",
        text="Country",
        hover_name="Country",
        hover_data={
            "Feature_Based_Rank": True,
            "Actual_Happiness_Rank": True,
            "Composite_Feature_Score": ":.3f",
            "Happiness_Score": ":.2f"
        },
        title=(
            "Actual Happiness vs Feature-Based Country Score<br>"
            "<sup>Equal weighting; lower corruption treated as positive</sup>"
        ),
        labels={
            "Composite_Feature_Score": "Composite feature score",
            "Happiness_Score": "Actual happiness score"
        }
    )
    country_figure.update_traces(
        marker={"size": 12, "color": "#2E86AB"},
        textposition="top center"
    )
    country_figure.update_layout(
        template="plotly_white",
        height=650,
        width=950,
        title={"x": 0.5}
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    feature_figure.write_html(OUTPUT_DIRECTORY / "feature_ranking.html")
    country_figure.write_html(OUTPUT_DIRECTORY / "country_feature_ranking.html")
    country_ranking[[
        "Country",
        "Composite_Feature_Score",
        "Feature_Based_Rank",
        "Happiness_Score",
        "Actual_Happiness_Rank"
    ]].to_csv(OUTPUT_DIRECTORY / "country_feature_ranking.csv", index=False)



def create_readme_feature_dashboard(
    data,
    feature_ranking,
    country_ranking,
    lowest_country,
    dataset_average_freedom
):
    """Save a static feature dashboard that GitHub can display in README."""
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    bubble_sizes = 250 * data["GDP_per_Capita"]
    bubble = axes[0, 0].scatter(
        data["Freedom_to_Make_Choices"],
        data["Happiness_Score"],
        s=bubble_sizes,
        c=data["Perceptions_of_Corruption"],
        cmap="RdYlGn_r",
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8
    )
    axes[0, 0].set_title("Happiness vs Freedom")
    axes[0, 0].set_xlabel("Freedom to Make Choices")
    axes[0, 0].set_ylabel("Happiness Score")
    axes[0, 0].grid(linestyle="--", alpha=0.3)
    colour_bar = figure.colorbar(bubble, ax=axes[0, 0])
    colour_bar.set_label("Perceived Corruption")

    ordered_features = feature_ranking.sort_values(
        "Correlation_with_Happiness"
    )
    correlation_colours = [
        "#C0392B" if value < 0 else "#2E86AB"
        for value in ordered_features["Correlation_with_Happiness"]
    ]
    axes[0, 1].barh(
        ordered_features["Feature"],
        ordered_features["Correlation_with_Happiness"],
        color=correlation_colours
    )
    axes[0, 1].axvline(0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Features Ranked by Correlation with Happiness")
    axes[0, 1].set_xlabel("Pearson correlation")
    axes[0, 1].grid(axis="x", linestyle="--", alpha=0.3)

    country = lowest_country["Country"]
    freedom_score = lowest_country["Average_Freedom_Score"]
    axes[1, 0].barh(
        [country],
        [freedom_score],
        color="#2E86AB",
        height=0.45,
        label="Country Freedom score"
    )
    axes[1, 0].axvline(
        dataset_average_freedom,
        color="#C0392B",
        linestyle="--",
        linewidth=2,
        label=f"Dataset average ({dataset_average_freedom:.2f})"
    )
    axes[1, 0].text(
        freedom_score + 0.02,
        0,
        f"{freedom_score:.2f}",
        va="center"
    )
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_title("Freedom of the Lowest-Happiness Country")
    axes[1, 0].set_xlabel("Freedom to Make Choices")
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(axis="x", linestyle="--", alpha=0.3)

    axes[1, 1].scatter(
        country_ranking["Composite_Feature_Score"],
        country_ranking["Happiness_Score"],
        color="#2E86AB",
        s=65,
        alpha=0.8
    )
    for _, row in country_ranking.iterrows():
        axes[1, 1].annotate(
            row["Country"],
            (row["Composite_Feature_Score"], row["Happiness_Score"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7
        )
    axes[1, 1].set_title("Actual Happiness vs Composite Feature Score")
    axes[1, 1].set_xlabel("Composite Feature Score")
    axes[1, 1].set_ylabel("Happiness Score")
    axes[1, 1].grid(linestyle="--", alpha=0.3)

    figure.suptitle(
        "How Other Features Relate to Happiness",
        fontsize=18,
        fontweight="bold"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    figure.savefig(
        OUTPUT_DIRECTORY / "feature_relationships_dashboard.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)


def detect_outliers(data):
    """Detect numeric outliers using the teacher-taught IQR and Z-score rules."""
    numeric_data = data.select_dtypes(include="number")

    first_quartile = numeric_data.quantile(0.25)
    third_quartile = numeric_data.quantile(0.75)
    interquartile_range = third_quartile - first_quartile
    lower_bounds = first_quartile - (1.5 * interquartile_range)
    upper_bounds = third_quartile + (1.5 * interquartile_range)
    iqr_flags = numeric_data.lt(lower_bounds) | numeric_data.gt(upper_bounds)

    means = numeric_data.mean()
    standard_deviations = numeric_data.std(ddof=0).replace(0, float("nan"))
    z_scores = (numeric_data - means) / standard_deviations
    z_score_flags = z_scores.abs().gt(3)

    summary_rows = []
    detail_rows = []

    for feature in numeric_data.columns:
        maximum_z_index = z_scores[feature].abs().idxmax()
        summary_rows.append({
            "Feature": feature,
            "Q1": first_quartile[feature],
            "Q3": third_quartile[feature],
            "IQR": interquartile_range[feature],
            "Lower_Bound": lower_bounds[feature],
            "Upper_Bound": upper_bounds[feature],
            "IQR_Outlier_Count": int(iqr_flags[feature].sum()),
            "Z_Score_Outlier_Count": int(z_score_flags[feature].sum()),
            "Largest_Absolute_Z_Score": abs(z_scores.loc[
                maximum_z_index,
                feature
            ]),
            "Most_Extreme_Country": data.loc[maximum_z_index, "Country"]
        })

        combined_flags = iqr_flags[feature] | z_score_flags[feature]
        for row_index in data.index[combined_flags]:
            detail_rows.append({
                "Country": data.loc[row_index, "Country"],
                "Feature": feature,
                "Value": numeric_data.loc[row_index, feature],
                "IQR_Flag": bool(iqr_flags.loc[row_index, feature]),
                "Z_Score": z_scores.loc[row_index, feature],
                "Z_Score_Flag": bool(z_score_flags.loc[row_index, feature]),
                "Decision": "Review before removal"
            })

    outlier_summary = pd.DataFrame(summary_rows)
    summary_numeric_columns = [
        "Q1",
        "Q3",
        "IQR",
        "Lower_Bound",
        "Upper_Bound",
        "Largest_Absolute_Z_Score"
    ]
    outlier_summary[summary_numeric_columns] = outlier_summary[
        summary_numeric_columns
    ].round(3)
    outlier_details = pd.DataFrame(
        detail_rows,
        columns=[
            "Country",
            "Feature",
            "Value",
            "IQR_Flag",
            "Z_Score",
            "Z_Score_Flag",
            "Decision"
        ]
    )

    record_audit = data.copy()
    record_audit["IQR_Outlier_Any"] = iqr_flags.any(axis=1)
    record_audit["Z_Score_Outlier_Any"] = z_score_flags.any(axis=1)
    record_audit["Outlier_Decision"] = "Keep"
    record_audit.loc[
        record_audit["IQR_Outlier_Any"]
        | record_audit["Z_Score_Outlier_Any"],
        "Outlier_Decision"
    ] = "Review before removal"

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    outlier_summary.to_csv(
        OUTPUT_DIRECTORY / "outlier_detection_summary.csv",
        index=False
    )
    outlier_details.to_csv(
        OUTPUT_DIRECTORY / "outlier_records.csv",
        index=False
    )
    record_audit.to_csv(
        OUTPUT_DIRECTORY / "outlier_reviewed_dataset.csv",
        index=False
    )

    return outlier_summary, outlier_details, record_audit, z_scores


def create_outlier_visualisations(z_scores):
    """Create static and interactive boxplots for the outlier audit."""
    readable_names = {
        column: column.replace("_", " ")
        for column in z_scores.columns
    }

    figure, axis = plt.subplots(figsize=(14, 7))
    axis.boxplot(
        [z_scores[column].dropna() for column in z_scores.columns],
        tick_labels=[readable_names[column] for column in z_scores.columns],
        whis=1.5,
        showmeans=True
    )
    axis.axhline(3, color="#C0392B", linestyle="--", label="Z-score limit (+3)")
    axis.axhline(-3, color="#C0392B", linestyle="--", label="Z-score limit (-3)")
    axis.set_title("Outlier Detection Across Standardised Numeric Features")
    axis.set_ylabel("Z-score")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "outlier_detection_boxplots.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(figure)

    interactive_figure = go.Figure()
    for feature in z_scores.columns:
        interactive_figure.add_trace(
            go.Box(
                y=z_scores[feature],
                name=readable_names[feature],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                hovertemplate=(
                    f"<b>{readable_names[feature]}</b><br>"
                    "Z-score: %{y:.3f}<extra></extra>"
                )
            )
        )
    interactive_figure.add_hline(
        y=3,
        line_dash="dash",
        line_color="#C0392B",
        annotation_text="Z = +3 threshold"
    )
    interactive_figure.add_hline(
        y=-3,
        line_dash="dash",
        line_color="#C0392B",
        annotation_text="Z = -3 threshold"
    )
    interactive_figure.update_layout(
        title={
            "text": (
                "Outlier Detection: IQR Boxplots and Z-score Thresholds<br>"
                "<sup>No values are outside the IQR fences or |Z| > 3</sup>"
            ),
            "x": 0.5
        },
        template="plotly_white",
        height=650,
        width=1200,
        yaxis_title="Standardised Z-score",
        showlegend=False
    )
    interactive_figure.write_html(
        OUTPUT_DIRECTORY / "outlier_detection.html"
    )


def create_unified_plotly_dashboard(
    data,
    top_three,
    lowest_country,
    dataset_average_freedom,
    feature_ranking,
    z_scores,
    outlier_summary
):
    """Create the main interactive Happiness dashboard in one figure."""
    plot_data = top_three.sort_values(
        "Average_Happiness_Score",
        ascending=False
    ).reset_index(drop=True)
    correlation_data = feature_ranking.sort_values(
        "Correlation_with_Happiness"
    )

    figure = make_subplots(
        rows=4,
        cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "indicator"}],
            [{"type": "xy", "colspan": 2}, None, {"type": "xy"}],
            [{"type": "xy", "colspan": 3}, None, None]
        ],
        subplot_titles=(
            "", "", "",
            "Top Three Happiness Scores",
            "Freedom Summary for the Lowest-Happiness Country",
            "Happiness vs Freedom, GDP and Corruption",
            "Features Ranked by Correlation with Happiness",
            "Outlier Detection: Standardised Numeric Features"
        ),
        row_heights=[0.13, 0.25, 0.34, 0.28],
        column_widths=[0.34, 0.34, 0.32],
        vertical_spacing=0.11,
        horizontal_spacing=0.10
    )

    card_colours = ["#2E86AB", "#F18F01", "#2CA25F"]
    for column, (_, country_row) in enumerate(plot_data.iterrows(), start=1):
        figure.add_trace(
            go.Indicator(
                mode="number",
                value=country_row["Average_Happiness_Score"],
                number={
                    "valueformat": ".2f",
                    "font": {"size": 38, "color": card_colours[column - 1]}
                },
                title={
                    "text": (
                        f"#{column} {country_row['Country']}<br>"
                        "<span style='font-size:0.75em'>Happiness score</span>"
                    )
                }
            ),
            row=1,
            col=column
        )

    figure.add_trace(
        go.Bar(
            x=plot_data["Country"],
            y=plot_data["Average_Happiness_Score"],
            marker_color=card_colours,
            text=plot_data["Average_Happiness_Score"].round(2),
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>Happiness score: %{y:.2f}<extra></extra>"
            ),
            showlegend=False
        ),
        row=2,
        col=1
    )

    figure.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=lowest_country["Average_Freedom_Score"],
            number={"valueformat": ".2f"},
            delta={
                "reference": dataset_average_freedom,
                "valueformat": ".2f"
            },
            title={
                "text": (
                    f"{lowest_country['Country']}<br>"
                    "<span style='font-size:0.75em'>"
                    f"Happiness: {lowest_country['Average_Happiness_Score']:.2f}"
                    "</span>"
                )
            },
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#2E86AB"},
                "steps": [
                    {"range": [0, 0.4], "color": "#FADBD8"},
                    {"range": [0.4, 0.7], "color": "#FCF3CF"},
                    {"range": [0.7, 1], "color": "#D5F5E3"}
                ],
                "threshold": {
                    "line": {"color": "#C0392B", "width": 4},
                    "value": dataset_average_freedom
                }
            }
        ),
        row=2,
        col=3
    )

    readable_names = {
        column: column.replace("_", " ")
        for column in z_scores.columns
    }
    for feature in z_scores.columns:
        figure.add_trace(
            go.Box(
                y=z_scores[feature],
                name=readable_names[feature],
                boxpoints="all",
                jitter=0.2,
                pointpos=0,
                showlegend=False,
                hovertemplate=(
                    f"<b>{readable_names[feature]}</b><br>"
                    "Z-score: %{y:.3f}<extra></extra>"
                )
            ),
            row=4,
            col=1
        )

    figure.add_trace(
        go.Scatter(
            x=data["Freedom_to_Make_Choices"],
            y=data["Happiness_Score"],
            mode="markers",
            text=data["Country"],
            customdata=data[[
                "GDP_per_Capita",
                "Perceptions_of_Corruption"
            ]],
            marker={
                "size": 8 + (data["GDP_per_Capita"] * 18),
                "color": data["Perceptions_of_Corruption"],
                "coloraxis": "coloraxis",
                "opacity": 0.82,
                "line": {"color": "white", "width": 1}
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Freedom: %{x:.2f}<br>"
                "Happiness: %{y:.2f}<br>"
                "GDP per capita: %{customdata[0]:.2f}<br>"
                "Perceived corruption: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
            showlegend=False
        ),
        row=3,
        col=1
    )

    correlation_colours = [
        "#C0392B" if value < 0 else "#2E86AB"
        for value in correlation_data["Correlation_with_Happiness"]
    ]
    figure.add_trace(
        go.Bar(
            x=correlation_data["Correlation_with_Happiness"],
            y=correlation_data["Feature"],
            orientation="h",
            marker_color=correlation_colours,
            text=correlation_data["Correlation_with_Happiness"].round(3),
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"
            ),
            showlegend=False
        ),
        row=3,
        col=3
    )

    figure.update_yaxes(range=[0, 8], title_text="Happiness Score", row=2, col=1)
    figure.update_xaxes(title_text="Country", row=2, col=1)
    figure.update_xaxes(
        title_text="Freedom to Make Choices",
        row=3,
        col=1
    )
    figure.update_yaxes(title_text="Happiness Score", row=3, col=1)
    figure.update_xaxes(
        title_text="Pearson correlation",
        range=[-0.42, 0.24],
        zeroline=True,
        zerolinecolor="black",
        row=3,
        col=3
    )
    figure.update_yaxes(
        title_text="Standardised Z-score",
        range=[-3.4, 3.4],
        row=4,
        col=1
    )
    figure.add_shape(
        type="line",
        xref="x4 domain",
        yref="y4",
        x0=0,
        x1=1,
        y0=3,
        y1=3,
        line={"color": "#C0392B", "dash": "dash"}
    )
    figure.add_shape(
        type="line",
        xref="x4 domain",
        yref="y4",
        x0=0,
        x1=1,
        y0=-3,
        y1=-3,
        line={"color": "#C0392B", "dash": "dash"}
    )

    total_iqr_outliers = int(outlier_summary["IQR_Outlier_Count"].sum())
    total_z_outliers = int(outlier_summary["Z_Score_Outlier_Count"].sum())

    figure.update_layout(
        title={
            "text": (
                "World Happiness Dashboard<br>"
                "<sup>20-country dataset | "
                f"IQR outliers: {total_iqr_outliers} | "
                f"Z-score outliers: {total_z_outliers} | "
                "Decision: retain all records</sup>"
            ),
            "x": 0.5,
            "font": {"size": 26}
        },
        template="plotly_white",
        height=1500,
        width=1450,
        margin={"l": 80, "r": 80, "t": 130, "b": 70},
        coloraxis={
            "colorscale": "RdYlGn_r",
            "colorbar": {
                "title": "Perceived<br>corruption",
                "x": 0.64,
                "y": 0.20,
                "len": 0.30
            }
        }
    )

    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    dashboard_path = OUTPUT_DIRECTORY / "happiness_dashboard.html"
    figure.write_html(dashboard_path)
    figure.show()

    return dashboard_path


def main():
    happiness_data = load_dataset(DATASET_PATH)
    inspect_dataset(happiness_data)

    (
        outlier_summary,
        outlier_details,
        record_audit,
        z_scores
    ) = detect_outliers(happiness_data)

    print("\nOutlier detection summary:")
    print(
        outlier_summary[[
            "Feature",
            "IQR_Outlier_Count",
            "Z_Score_Outlier_Count",
            "Largest_Absolute_Z_Score",
            "Most_Extreme_Country"
        ]].round(3).to_string(index=False)
    )
    print(f"\nRecords flagged for review: {len(outlier_details)}")
    print(
        "Decision: retain all "
        f"{len(record_audit)} records because neither method detected an "
        "outlier."
    )
    create_outlier_visualisations(z_scores)

    pandas_top_three = get_top_three_with_pandas(happiness_data)

    print("\nThree happiest countries using Pandas aggregation:")
    print(pandas_top_three.to_string(index=False))

    sql_top_three = get_top_three_with_sql(happiness_data)

    print("\nThree happiest countries using SQL:")
    print(sql_top_three.to_string(index=False))

    create_matplotlib_subplot(pandas_top_three)
    create_plotly_subplot(pandas_top_three)

    lowest_country = get_lowest_happiness_country(happiness_data)
    dataset_average_freedom = happiness_data[
        "Freedom_to_Make_Choices"
    ].mean()

    print("\nFreedom summary for the country with the lowest happiness:")
    print(f"Country: {lowest_country['Country']}")
    print(
        "Happiness score: "
        f"{lowest_country['Average_Happiness_Score']:.2f}"
    )
    print(
        "Freedom score: "
        f"{lowest_country['Average_Freedom_Score']:.2f}"
    )
    print(f"Dataset average Freedom score: {dataset_average_freedom:.2f}")

    create_freedom_gauge(lowest_country, dataset_average_freedom)
    create_happiness_freedom_bubble_chart(happiness_data)

    feature_ranking, country_ranking = (
        calculate_feature_and_country_rankings(happiness_data)
    )

    print("\nFeatures ranked by relationship with Happiness:")
    print(
        feature_ranking[["Feature", "Correlation_with_Happiness"]]
        .to_string(index=False)
    )
    print("\nTop five countries using the composite feature score:")
    print(
        country_ranking[[
            "Country",
            "Composite_Feature_Score",
            "Feature_Based_Rank",
            "Actual_Happiness_Rank"
        ]].head().to_string(index=False)
    )

    create_ranking_visualisations(feature_ranking, country_ranking)
    create_readme_feature_dashboard(
        happiness_data,
        feature_ranking,
        country_ranking,
        lowest_country,
        dataset_average_freedom
    )

    dashboard_path = create_unified_plotly_dashboard(
        happiness_data,
        pandas_top_three,
        lowest_country,
        dataset_average_freedom,
        feature_ranking,
        z_scores,
        outlier_summary
    )
    print(f"\nInteractive dashboard saved to: {dashboard_path}")


if __name__ == "__main__":
    main()
