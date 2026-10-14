import pandas as pd
import numpy as np

INPUT_FILE = "humidity_curated_data.csv"
OUTPUT_FILE = "humidity_training_data.csv"
SLOPE_WINDOW = 5
PREDICTION_HORIZON = 5

df = pd.read_csv(INPUT_FILE)

slopes = []
for i in range(len(df)):
    if i < SLOPE_WINDOW - 1:
        slopes.append(np.nan)
        continue

    recent = df.iloc[i - SLOPE_WINDOW + 1:i + 1]
    x = recent["time_s"].values
    y = recent["humidity_pct"].values

    n = len(x)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    slopes.append(slope)

df["humidity_slope"] = slopes
df["future_humidity_5s"] = df["humidity_pct"].shift(-PREDICTION_HORIZON)

training_df = df[
    ["humidity_pct", "temperature_c", "humidity_slope", "future_humidity_5s"]
].copy()

training_df.columns = [
    "current_humidity",
    "temperature",
    "humidity_slope",
    "future_humidity_5s"
]

training_df = training_df.dropna().reset_index(drop=True)
training_df.to_csv(OUTPUT_FILE, index=False)

print(training_df.head(10))
print()
print("Raw samples:", len(df))
print("Training samples:", len(training_df))
print("Saved to:", OUTPUT_FILE)
