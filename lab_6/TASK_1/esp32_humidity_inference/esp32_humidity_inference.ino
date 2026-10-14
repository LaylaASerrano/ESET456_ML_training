#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <Adafruit_AHTX0.h>
#include <SPI.h>
#include <Wire.h>

// TFT pins ==================================
#define TFT_CS 17
#define TFT_DC 21
#define TFT_RES 4
#define TFT_MOSI 23
#define TFT_SCLK 18

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RES);
Adafruit_AHTX0 aht;

// Model settings ============================
#define SLOPE_WINDOW 5
#define HUMIDITY_THRESHOLD 60.0

float humiditySamples[SLOPE_WINDOW];
int sampleCount = 0;

// Trained model coefficients ================
const float MODEL_INTERCEPT = 0.9103225471097147;
const float COEF_HUMIDITY = 0.9760348336779218;
const float COEF_TEMPERATURE = 0.017807561113276735;
const float COEF_SLOPE = 2.5867813741136674;

// Store humidity sample =====================
void addHumiditySample(float newHumidity) {
  if (sampleCount < SLOPE_WINDOW) humiditySamples[sampleCount++] = newHumidity;
  else {
    for (int i = 0; i < SLOPE_WINDOW - 1; i++) humiditySamples[i] = humiditySamples[i + 1];
    humiditySamples[SLOPE_WINDOW - 1] = newHumidity;
  }
}

// Calculate recent humidity slope ===========
float calculateSlope() {
  float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

  for (int i = 0; i < SLOPE_WINDOW; i++) {
    float x = i;
    float y = humiditySamples[i];

    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumX2 += x * x;
  }

  return (SLOPE_WINDOW * sumXY - sumX * sumY) / (SLOPE_WINDOW * sumX2 - sumX * sumX);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // AHT10 ===================================
  Wire.begin(27, 33);
  if (!aht.begin(&Wire)) {
    Serial.println("Could not find AHT10!");
    while (1) delay(10);
  }

  // LCD =====================================
  tft.initR(INITR_MINI160x80);
  tft.setRotation(3);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(2);

  Serial.println("ESP32 Linear Regression Model Ready");
}

void loop() {
  sensors_event_t humidity, temp;
  aht.getEvent(&humidity, &temp);

  float currentHumidity = humidity.relative_humidity;
  float currentTemperature = temp.temperature;

  // Store humidity ==========================
  addHumiditySample(currentHumidity);

  // Collect first 5 samples =================
  if (sampleCount < SLOPE_WINDOW) {
    Serial.print("Collecting Samples: ");
    Serial.print(sampleCount);
    Serial.print("/");
    Serial.println(SLOPE_WINDOW);

    tft.fillScreen(ST77XX_BLACK);
    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(2);
    tft.setCursor(5, 15);
    tft.println("Collecting");
    tft.setCursor(5, 45);
    tft.print(sampleCount);
    tft.print("/");
    tft.print(SLOPE_WINDOW);

    delay(1000);
    return;
  }

  // Calculate slope =========================
  float humiditySlope = calculateSlope();

  // AI model inference ======================
float predictedHumidity = MODEL_INTERCEPT
                        + (COEF_HUMIDITY * currentHumidity)
                        + (COEF_TEMPERATURE * currentTemperature)
                        + (COEF_SLOPE * humiditySlope);

  predictedHumidity = constrain(predictedHumidity, 0.0, 100.0);

  // Trend ===================================
  String trend = "STABLE";
  if (humiditySlope > 0.1) trend = "RISING";
  else if (humiditySlope < -0.1) trend = "FALLING";

  // Decision =================================
  String status = "NORMAL";
  if (currentHumidity >= HUMIDITY_THRESHOLD) status = "HIGH NOW";
  else if (predictedHumidity >= HUMIDITY_THRESHOLD) status = "HIGH SOON";

  // Serial ==================================
  Serial.println("-----------------------");

  Serial.print("Temperature: ");
  Serial.print(currentTemperature, 1);
  Serial.println(" C");

  Serial.print("Current Humidity: ");
  Serial.print(currentHumidity, 1);
  Serial.println(" %RH");

  Serial.print("Humidity Slope: ");
  Serial.print(humiditySlope, 3);
  Serial.println(" %RH/sec");

  Serial.print("Predicted Humidity in 5 sec: ");
  Serial.print(predictedHumidity, 1);
  Serial.println(" %RH");

  Serial.print("Trend: ");
  Serial.println(trend);

  Serial.print("Status: ");
  Serial.println(status);

  // LCD =====================================
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_WHITE);

  tft.setCursor(5, 5);
  tft.print("RH:");
  tft.print(currentHumidity, 1);
  tft.print("%");

  tft.setCursor(5, 30);
  tft.print("Pred:");
  tft.print(predictedHumidity, 1);
  tft.print("%");

  tft.setCursor(5, 55);

  if (status == "HIGH NOW") {
    tft.setTextColor(ST77XX_RED);
    tft.print("HIGH NOW");
  } else if (status == "HIGH SOON") {
    tft.setTextColor(ST77XX_YELLOW);
    tft.print("HIGH SOON");
  } else {
    tft.setTextColor(ST77XX_GREEN);
    tft.print(trend);
  }

  delay(1000);
}