import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# File ======================================
INPUT_FILE = "humidity_training_data.csv"

# Columns ===================================
FEATURES = ["current_humidity", "temperature", "humidity_slope"]
TARGET = "future_humidity_5s"

# Load data =================================
df = pd.read_csv(INPUT_FILE)

print("Dataset loaded")
print("Rows:", len(df))
print()
print(df.head())

# Inputs and target ==========================
X = df[FEATURES]
y = df[TARGET]

# Chronological 80/20 split =================
split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model ==============================
model = LinearRegression()

# Train / fit model =========================
model.fit(X_train, y_train)

# Predict ===================================
predictions = model.predict(X_test)

# Evaluate ==================================
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print()
print("MODEL RESULTS")
print("======================================")
print("MAE:", round(mae, 3), "%RH")
print("R^2:", round(r2, 3))

# Model coefficients ========================
print()
print("MODEL COEFFICIENTS")
print("======================================")
print("Intercept:", model.intercept_)

for feature, coefficient in zip(FEATURES, model.coef_):
    print(feature, ":", coefficient)

# ESP32 equation ============================
print()
print("ESP32 EQUATION")
print("======================================")

print(
    f"predictedHumidity = {model.intercept_:.6f} "
    f"+ ({model.coef_[0]:.6f} * currentHumidity) "
    f"+ ({model.coef_[1]:.6f} * currentTemperature) "
    f"+ ({model.coef_[2]:.6f} * humiditySlope);"
)

# Show some predictions =====================
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print()
print("FIRST 10 TEST PREDICTIONS")
print("======================================")
print(results.head(10))

# Plot 1: Actual vs Predicted ===============
plt.figure(figsize=(7, 5))

plt.scatter(y_test, predictions, alpha=0.6)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")

plt.xlabel("Actual Humidity 5 Seconds Later (%RH)")
plt.ylabel("Predicted Humidity 5 Seconds Later (%RH)")
plt.title("Actual vs Predicted Humidity")
plt.grid(True)

plt.show()

# Plot 2: Predictions over test samples =====
plt.figure(figsize=(10, 5))

plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")

plt.xlabel("Test Sample")
plt.ylabel("Humidity (%RH)")
plt.title("Actual vs Predicted Future Humidity")
plt.legend()
plt.grid(True)

plt.show()