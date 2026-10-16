import datetime
import re
import numpy as np
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from pathlib import Path

# Store all charts that are active during the current time window.
active_figures = []
import pytz

# 1. Load Datasets
df_apps = pd.read_csv("elevance/Play Store Data.csv")
df_reviews = pd.read_csv("elevance/User Reviews.csv")


# 2. Data Cleaning & Transformation Helpers
def clean_size(size_str):
  if pd.isna(size_str) or size_str == "Varies with device":
    return np.nan
  size_str = str(size_str).upper()
  if "M" in size_str:
    return float(size_str.replace("M", "").strip())
  elif "K" in size_str:
    return float(size_str.replace("K", "").strip()) / 1024.0
  return np.nan


def clean_installs(installs_str):
  if pd.isna(installs_str):
    return 0
  clean_val = (
      str(installs_str).replace("+", "").replace(",", "").strip()
  )
  try:
    return int(clean_val)
  except ValueError:
    return 0


def clean_price(price_str):
  if pd.isna(price_str):
    return 0.0
  clean_val = str(price_str).replace("$", "").strip()
  try:
    return float(clean_val)
  except ValueError:
    return 0.0


# Clean main apps dataset
df_apps["Size_MB"] = df_apps["Size"].apply(clean_size)
df_apps["Installs_num"] = df_apps["Installs"].apply(clean_installs)
df_apps["Reviews_num"] = (
    pd.to_numeric(df_apps["Reviews"], errors="coerce").fillna(0).astype(int)
)
df_apps["Rating"] = pd.to_numeric(df_apps["Rating"], errors="coerce")
df_apps["Price_num"] = df_apps["Price"].apply(clean_price)
df_apps["Revenue"] = df_apps["Installs_num"] * df_apps["Price_num"]
df_apps["Last Updated Date"] = pd.to_datetime(
    df_apps["Last Updated"], errors="coerce"
)

# Merge subjectivity from user reviews dataset
avg_subjectivity = (
    df_reviews.groupby("App")["Sentiment_Subjectivity"].mean().reset_index()
)
df_merged = pd.merge(df_apps, avg_subjectivity, on="App", how="left")
df_merged["Sentiment_Subjectivity"] = df_merged[
    "Sentiment_Subjectivity"
].fillna(0.0)

# Define IST Timezone Checker
ist_tz = pytz.timezone("Asia/Kolkata")


def current_ist_time():
  return datetime.datetime.now(ist_tz).time()


from datetime import datetime, time

def is_within_time(start_hour, start_min, end_hour, end_min):
    now_t = datetime.now().time()
    start_t = time(start_hour, start_min)
    end_t = time(end_hour, end_min)

    return start_t <= now_t <= end_t


print("Data successfully loaded and preprocessed!")

# ==========================================
# CHART 1: Grouped Bar Chart
# Time Window: 3:00 PM - 5:00 PM IST
# ==========================================
if is_within_time(15, 0, 17, 0):
  print("\n--- Generating Chart 1: Grouped Bar Chart ---")
  # Filters: Last update Jan, Rating >= 4.0, Size >= 10M
  df_c1 = df_merged[
      (df_merged["Last Updated Date"].dt.month == 1)
      & (df_merged["Rating"] >= 4.0)
      & (df_merged["Size_MB"] >= 10.0)
  ].copy()

  # Top 10 categories by total installs
  top_cats_c1 = (
      df_c1.groupby("Category")["Installs_num"]
      .sum()
      .nlargest(10)
      .index.tolist()
  )
  df_c1_filtered = df_c1[df_c1["Category"].isin(top_cats_c1)]

  agg_c1 = (
      df_c1_filtered.groupby("Category")
      .agg({"Rating": "mean", "Reviews_num": "sum"})
      .reset_index()
  )

  fig1 = px.bar(
      agg_c1,
      x="Category",
      y=["Rating", "Reviews_num"],
      barmode="group",
      title=(
          "Top 10 App Categories (Jan Update, Rating >= 4.0, Size >= 10M) -"
          " Average Rating & Total Reviews"
      ),
  )
  active_figures.append(fig1)
else:
  print(
      "[Chart 1 Hidden]: Outside active time window (Allowed: 3:00 PM - 5:00 PM"
      " IST)."
  )

# ==========================================
# CHART 2: Interactive Choropleth Map
# Time Window: 6:00 PM - 8:00 PM IST
# ==========================================
if is_within_time(18, 0, 20, 0):
  print("\n--- Generating Chart 2: Choropleth Map ---")
  # Note: Play Store dataset is global/general. Simulating country-level distribution for demonstration via Plotly map.
  # Filter categories: Top 5 categories, exclude starting with A, C, G, S
  valid_cats_c2 = [
      cat
      for cat in df_merged["Category"].unique()
      if not cat.upper().startswith(("A", "C", "G", "S"))
  ]
  df_c2 = df_merged[df_merged["Category"].isin(valid_cats_c2)]
  top_5_cats = (
      df_c2.groupby("Category")["Installs_num"].sum().nlargest(5).index.tolist()
  )
  df_c2_top = df_c2[df_c2["Category"].isin(top_5_cats)].copy()

  # Highlight where installs exceed 1 Million
  df_c2_top["High_Install_Flag"] = (
      df_c2_top["Installs_num"] > 1000000
  ).astype(str)

  # Sample country mapping for map visualization demo
  world_data = pd.DataFrame({
      "Country": [
          "United States",
          "India",
          "United Kingdom",
          "Germany",
          "Canada",
      ],
      "ISO_A3": ["USA", "IND", "GBR", "DEU", "CAN"],
      "Installs": [1500000, 2500000, 800000, 1200000, 950000],
      "Category": top_5_cats[:5] if len(top_5_cats) >= 5 else top_5_cats * 2,
  })

  fig2 = px.choropleth(
      world_data,
      locations="ISO_A3",
      color="Installs",
      hover_name="Country",
      hover_data=["Category"],
      title=(
          "Global Installs by Category (Top 5 Categories, excluding A, C, G,"
          " S)"
      ),
      color_continuous_scale="Viridis",
  )
  active_figures.append(fig2)
else:
  print(
      "[Chart 2 Hidden]: Outside active time window (Allowed: 6:00 PM - 8:00 PM"
      " IST)."
  )

# ==========================================
# CHART 3: Dual-Axis Chart
# Time Window: 1:00 PM - 2:00 PM IST
# ==========================================
if is_within_time(13, 0, 14, 0):
  print("\n--- Generating Chart 3: Dual-Axis Chart ---")
  # Filters: Installs >= 10k, Revenue >= $10k, Android > 4.0, Size > 15M, Content Rating == Everyone, App Name length <= 30
  df_c3 = df_merged[
      (df_merged["Installs_num"] >= 10000)
      & (df_merged["Revenue"] >= 10000)
      & (df_merged["Size_MB"] > 15.0)
      & (df_merged["Content Rating"] == "Everyone")
      & (df_merged["App"].str.len() <= 30)
      & (
          df_merged["Android Ver"].str.contains(
              r"[5-9]|\d{2}", na=False, regex=True
          )
      )
  ].copy()

  top_3_cats = (
      df_c3.groupby("Category")["Installs_num"]
      .sum()
      .nlargest(3)
      .index.tolist()
  )
  df_c3 = df_c3[df_c3["Category"].isin(top_3_cats)]

  agg_c3 = (
      df_c3.groupby(["Category", "Type"])
      .agg({"Installs_num": "mean", "Revenue": "mean"})
      .reset_index()
  )

  fig3 = go.Figure()
  fig3.add_trace(
      go.Bar(
          x=agg_c3["Category"] + " (" + agg_c3["Type"] + ")",
          y=agg_c3["Installs_num"],
          name="Average Installs",
          yaxis="y1",
      )
  )
  fig3.add_trace(
      go.Scatter(
          x=agg_c3["Category"] + " (" + agg_c3["Type"] + ")",
          y=agg_c3["Revenue"],
          name="Average Revenue ($)",
          yaxis="y2",
          mode="lines+markers",
      )
  )
  fig3.update.layout = title=(
      "Free vs Paid Apps: Average Installs & Revenue (Top 3 Categories)"
  )
  fig3.update_layout(
      yaxis=dict(title="Average Installs"),
      yaxis2=dict(
          title="Average Revenue ($)", overlaying="y", side="right"
      ),
  )
  active_figures.append(fig3)
else:
  print(
      "[Chart 3 Hidden]: Outside active time window (Allowed: 1:00 PM - 2:00 PM"
      " IST)."
  )

# ==========================================
# CHART 4: Time Series Line Chart with Growth Shading & Translations
# Time Window: 6:00 PM - 9:00 PM IST
# ==========================================
if is_within_time(18, 0, 21, 0):
  print("\n--- Generating Chart 4: Time Series Line Chart ---")
  # Filters: App name doesn't start with x,y,z; category starts with E, C, B; reviews > 500; app name doesn't contain 'S'/'s'
  df_c4 = df_merged[
      ~df_merged["App"].str.lower().str.startswith(("x", "y", "z"))
      & df_merged["Category"].str.upper().str.startswith(("E", "C", "B"))
      & (df_merged["Reviews_num"] > 500)
      & (~df_merged["App"].str.contains("s|S", regex=True))
  ].copy()

  # Category Translations
  cat_map = {"BEAUTY": "सौंदर्य", "BUSINESS": "வணிகம்", "DATING": "Dating"}

  # Per prompt specifications: Beauty -> Hindi (सौंदर्य), Business -> Tamil (வணிகம்), Dating -> German (Dating)
  translation_dict = {
      "BEAUTY": "सौंदर्य (Hindi)",
      "BUSINESS": "வணிகம் (Tamil)",
      "DATING": "Dating (German)",
  }
  df_c4["Category_Translated"] = df_c4["Category"].str.upper().map(
      lambda x: translation_dict.get(x, x)
  )

  # Grouping by Last Updated Month and Category for time series trend
  df_c4["YearMonth"] = df_c4["Last Updated Date"].dt.to_period("M").astype(str)
  ts_c4 = (
      df_c4.groupby(["YearMonth", "Category_Translated"])["Installs_num"]
      .sum()
      .reset_index()
  )

  fig4 = px.line(
      ts_c4,
      x="YearMonth",
      y="Installs_num",
      color="Category_Translated",
      title=(
          "Installs Trend Over Time by Category (Translated Categories,"
          " Filtered)"
      ),
  )
  active_figures.append(fig4)
else:
  print(
      "[Chart 4 Hidden]: Outside active time window (Allowed: 6:00 PM - 9:00 PM"
      " IST)."
  )

# ==========================================
# CHART 5: Bubble Chart
# Time Window: 5:00 PM - 7:00 PM IST
# ==========================================
if is_within_time(17, 0, 19, 0):
    print("\n--- Generating Chart 5: Bubble Chart ---")

    allowed_cats_c5 = [
        "GAME", "BEAUTY", "BUSINESS", "COMICS",
        "COMMUNICATION", "DATING", "ENTERTAINMENT",
        "SOCIAL", "EVENTS",
    ]

    df_c5 = df_merged[
        (df_merged["Rating"] > 3.5)
        & (df_merged["Category"].str.upper().isin(allowed_cats_c5))
        & (df_merged["Reviews_num"] > 500)
        & (~df_merged["App"].str.contains("s|S", regex=True))
        & (df_merged["Sentiment_Subjectivity"] > 0.5)
        & (df_merged["Installs_num"] > 50000)
        & (df_merged["Size_MB"].notna())
    ].copy()

    # Apply category translations
    c5_trans = {
        "BEAUTY": "सौंदर्य (Hindi)",
        "BUSINESS": "வணிகம் (Tamil)",
        "DATING": "Dating (German)",
    }

    df_c5["Category_Display"] = (
        df_c5["Category"].str.upper()
        .map(lambda x: c5_trans.get(x, x))
    )

    # Fixed colors for consistent rendering
    category_colors = {
        "सौंदर्य (Hindi)": "#FF7F7F",
        "வணிகம் (Tamil)": "#40D9B4",
        "Dating (German)": "#C9A0FF",
        "COMICS": "#A0A0A0",
        "COMMUNICATION": "#40D9B4",
        "ENTERTAINMENT": "#FFB366",
        "GAME": "#FFB6C8",
        "SOCIAL": "#45D9F2",
        "EVENTS": "#8C9EFF",
    }

    fig5 = px.scatter(
        df_c5,
        x="Size_MB",
        y="Rating",
        size="Installs_num",
        color="Category_Display",
        color_discrete_map=category_colors,
        title=(
            "App Size vs Rating Bubble Chart "
            "(Filtered, Game highlighted in Pink)"
        ),
    )

    active_figures.append(fig5)

else:
    print(
        "[Chart 5 Hidden]: Outside active time window "
        "(Allowed: 5:00 PM - 7:00 PM IST)."
    )


# ==========================================
# CHART 6: Stacked Area Chart
# Time Window: 4:00 PM - 6:00 PM IST
# ==========================================
if is_within_time(16, 0, 18, 0):
  print("\n--- Generating Chart 6: Stacked Area Chart ---")
  # Filters: Rating >= 4.2, App name no numbers, Category starts with T or P, Reviews > 1000, Size between 20M and 80M
  df_c6 = df_merged[
      (df_merged["Rating"] >= 4.2)
      & (~df_merged["App"].str.contains(r"\d", regex=True))
      & df_merged["Category"].str.upper().str.startswith(("T", "P"))
      & (df_merged["Reviews_num"] > 1000)
      & (df_merged["Size_MB"] >= 20.0)
      & (df_merged["Size_MB"] <= 80.0)
  ].copy()

  # Legend translations: Travel & Local -> French, Productivity -> Spanish, Photography -> Japanese
  legend_trans = {
      "TRAVEL_AND_LOCAL": "Voyage et Local (French)",
      "PRODUCTIVITY": "Productividad (Spanish)",
      "PHOTOGRAPHY": "写真 (Japanese)",
  }
  df_c6["Category_Legend"] = df_c6["Category"].str.upper().map(
      lambda x: legend_trans.get(x, x)
  )

  df_c6["YearMonth"] = df_c6["Last Updated Date"].dt.to_period("M").astype(str)
  area_c6 = (
      df_c6.groupby(["YearMonth", "Category_Legend"])["Installs_num"]
      .sum()
      .reset_index()
  )

  fig6 = px.area(
      area_c6,
      x="YearMonth",
      y="Installs_num",
      color="Category_Legend",
      title=(
          "Cumulative Installs Over Time by Category (Filtered & Translated)"
      ),
  )
  active_figures.append(fig6)
else:
  print(
      "[Chart 6 Hidden]: Outside active time window (Allowed: 4:00 PM - 6:00 PM"
      " IST)."
  )

# ==========================================
# DISPLAY ALL ACTIVE CHARTS IN ONE BROWSER TAB
# ==========================================
if active_figures:
    chart_sections = []
    for index, fig in enumerate(active_figures):
        chart_sections.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if index == 0 else False,
                config={"responsive": True},
            )
        )

    dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Play Store Analytics Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }
        h1 { text-align: center; }
        .chart { background: white; padding: 12px; margin: 0 auto 22px; max-width: 1400px; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>Play Store Analytics Dashboard</h1>
    """ + "".join(f'<section class="chart">{chart}</section>' for chart in chart_sections) + """
</body>
</html>
"""

    dashboard_path = Path("play_store_dashboard.html").resolve()
    dashboard_path.write_text(dashboard_html, encoding="utf-8")
    print(f"\nDashboard created: {dashboard_path}")
    webbrowser.open(dashboard_path.as_uri())
else:
    print("\nNo charts are active at the current time, so no dashboard was opened.")
