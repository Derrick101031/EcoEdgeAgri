/********************************************************************
   Dual-prediction demo  |  ESP32-S3  |  ThingSpeak push-on-error
   ------------------------------------------------------------------
   • Reads soil-moisture (ADC → %VWC), air temperature (°C) and RH (%)
   • Uses a TCN model (TensorFlow Lite Micro) to PREDICT the next sample
   • When the live reading deviates from yesterday’s prediction by more
     than a threshold, it sends the live values **and the error** to
     ThingSpeak (Fields 1-6).
   • Also prints human-readable alerts when a value strays outside the
     ‘healthy bell-pepper’ bands.
 ********************************************************************/

#include <Arduino.h>        // Core Arduino functions
#include "DHT.h"            // Library for DHT11/22 family
#include <WiFi.h>           // ESP32 Wi-Fi driver
#include <ThingSpeak.h>     // Upload helper from MathWorks
#include <MicroTFLite.h>    // “C-style” wrapper around TFL-Micro
#include "tcn_model_int8.h" // Your quantised TCN model as a C array

/* ─────────────── 1. Wi-Fi & ThingSpeak credentials ───────────────
   Replace the SSID, password, channel ID and API-key with *your* data.
   Wi-FiClient is the TCP socket object ThingSpeak will use.          */
const char *WIFI_SSID   = "TP-Link_3FE6";
const char *WIFI_PASS   = "61332074";
const unsigned long TS_CHANNEL_ID  = 2968337;
const char *TS_API_KEY = "YKEGOU8LYUSY02GU";
WiFiClient tsClient;

/* ─────────────── 2. Pin assignments / sensor objects ───────────── */
#define DHTPIN  5                 // GPIO5 → DHT data
#define DHTTYPE DHT11             // Change to DHT22 if you ever upgrade
const uint8_t MOIST_PIN = 2;      // Soil-moisture sensor into ADC2
DHT dht(DHTPIN, DHTTYPE);         // Create the sensor object

/* ─────────────── 3. “Healthy band” for bell-pepper crop ───────────
   If the live value falls outside these ranges an *alert* is printed.
   They can be fine-tuned later from agronomy literature / field data. */
const float MOIST_LOW = 25.0, MOIST_HIGH = 35.0;   // % volumetric water
const float TEMP_LOW  = 15.0, TEMP_HIGH  = 30.0;   // °C
const float HUM_LOW   = 60.0, HUM_HIGH   = 80.0;   // % RH

/* ─────────────── 4. ADC-to-VWC calibration constants ──────────────
   Adjust SENSOR_DRY / SENSOR_WET after you push the probe into DRY
   and SATURATED soil and read the raw ADC numbers.                   */
const int SENSOR_DRY = 3000;   // raw ADC ≈ 0 % VWC
const int SENSOR_WET = 1500;   // raw ADC ≈ 100 % VWC

/* ─────────────── 5. Model-input buffer & tensor arena ─────────────
   • input_buf holds the last 24 samples (Moist, Temp, Hum)
   • tensor_arena is the scratch-RAM the micro-interpreter needs       */
constexpr int SEQ_LEN  = 24;
constexpr int NUM_FEAT = 3;
float   input_buf[SEQ_LEN * NUM_FEAT] = {0};   // circular window
uint8_t buf_idx = 0;                           // how full the buffer is

constexpr size_t kArena = 12 * 1024;           // 12 kB fits ESP32-S3
uint8_t tensor_arena[kArena];

/* ─────────────── 6. Storage for the “prediction for t+1” ──────────
   predNext.valid tells us if we have already run at least one
   inference (otherwise we can’t compare prediction to ground-truth). */
struct Pred {
  float m = 0;   // predicted Moisture %
  float t = 0;   // predicted Temp °C
  float h = 0;   // predicted Hum %
  bool  valid = false;
} predNext;

/* ─────────────── 7. Error thresholds that trigger an upload ───────
   If *any* sensor deviates from its prediction by more than these
   margins we push a ThingSpeak update.                               */
const float ERR_MOIST = 3.0;   // % VWC
const float ERR_TEMP  = 1.5;   // °C
const float ERR_HUM   = 4.0;   // % RH

/* ─────────────── 8. NEW: variables that hold *last* error value ───
   They are uploaded to ThingSpeak Fields 4-6 so you can graph them.  */
float errM = 0;   // Moisture error
float errT = 0;   // Temperature error
float errH = 0;   // Humidity error

unsigned long lastUpload = 0;  // so we respect ThingSpeak 15-s limit

/* ══════════════════════════════════════════════════════════════════
   ===============           SET-UP SECTION            ===============
   ══════════════════════════════════════════════════════════════════*/

void wifiConnect() {
  WiFi.mode(WIFI_STA);                 // Station mode (just connect)
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print('.');
  }
  Serial.println(" ✅");
}

void setup() {
  Serial.begin(115200);
  dht.begin();

  wifiConnect();
  ThingSpeak.begin(tsClient);          // binds client to library

  if (!ModelInit(tcn_model_int8_tflite, tensor_arena, kArena)) {
    Serial.println("❌ TensorFlow-Lite model failed to load");
    while (true);                      // freeze here so user notices
  }
}

/* ══════════════════════════════════════════════════════════════════
   ===============            MAIN  LOOP              ===============
   ══════════════════════════════════════════════════════════════════*/
void loop() {
  /* ───── 1. LIVE SENSOR READS ───── */
  int   adcRaw = analogRead(MOIST_PIN);    // 0-4095
  float tempC  = dht.readTemperature();    // °C
  float rhPct  = dht.readHumidity();       // % RH
  if (isnan(tempC) || isnan(rhPct)) {
    Serial.println("⚠️  DHT read failed—skipping cycle");
    delay(2000); return;
  }

  // map() returns long int, so we cast back to float afterwards
  float vwcPct = map(adcRaw, SENSOR_DRY, SENSOR_WET, 0, 100);
  vwcPct = constrain(vwcPct, 0, 100);      // clip any overshoot

  /* ───── 2. LIVE ALERTS (rule-based) ───── */
  if      (vwcPct < MOIST_LOW)  Serial.println("🚨 Soil too dry (LIVE)");
  else if (vwcPct > MOIST_HIGH) Serial.println("🚨 Soil too wet (LIVE)");

  if      (tempC  < TEMP_LOW)   Serial.println("🚨 Temperature low (LIVE)");
  else if (tempC  > TEMP_HIGH)  Serial.println("🚨 Temperature high (LIVE)");

  if      (rhPct  < HUM_LOW)    Serial.println("🚨 Humidity low (LIVE)");
  else if (rhPct  > HUM_HIGH)   Serial.println("🚨 Humidity high (LIVE)");

  /* ───── 3. PREDICTION ERROR CHECK ─────
       Compare what we *predicted last cycle* (predNext) with what
       we actually measured *this* cycle. */
  bool needUpload = false;
  if (predNext.valid) {
    errM = fabs(vwcPct - predNext.m);        // ← store for TS
    errT = fabs(tempC  - predNext.t);
    errH = fabs(rhPct  - predNext.h);

    Serial.printf("ERR ▲ Moist:%.1f  Temp:%.1f  Hum:%.1f\n",
                  errM, errT, errH);

    if (errM > ERR_MOIST || errT > ERR_TEMP || errH > ERR_HUM)
      needUpload = true;
  }

  /* ───── 4. CONDITIONAL THINGSPEAK UPLOAD ───── */
  if (needUpload && millis() - lastUpload >= 15*1000UL) {
    ThingSpeak.setField(1, vwcPct);   // live soil-moisture
    ThingSpeak.setField(2, tempC);    // live temperature
    ThingSpeak.setField(3, rhPct);    // live humidity

    ThingSpeak.setField(4, errM);     // NEW: error values
    ThingSpeak.setField(5, errT);
    ThingSpeak.setField(6, errH);

    int httpCode = ThingSpeak.writeFields(TS_CHANNEL_ID, TS_API_KEY);
    if (httpCode == 200) {
      Serial.println("✓ ThingSpeak updated");
      lastUpload = millis();
    } else {
      Serial.printf("✗ ThingSpeak error %d (see docs)\n", httpCode);
    }
  }

  /* ───── 5. PUSH LATEST SAMPLE INTO MODEL BUFFER & INFER t+1 ─────
         • Convert real-world units → 0-1 normalised values
         • Shift the sliding window left, append new sample
         • If window is full, run the TCN & store its predictions     */
  float nm = vwcPct / 100.0;           // 0-1
  float nt = (tempC + 10.0) / 60.0;    // −10→50 mapped to 0→1
  float nh = rhPct / 100.0;            // 0-1

  // slide the array left by one row (3 floats) ➜ memmove is fast
  memmove(input_buf, input_buf + NUM_FEAT,
          (SEQ_LEN - 1) * NUM_FEAT * sizeof(float));

  // append new sample
  input_buf[(SEQ_LEN - 1)*NUM_FEAT + 0] = nm;
  input_buf[(SEQ_LEN - 1)*NUM_FEAT + 1] = nt;
  input_buf[(SEQ_LEN - 1)*NUM_FEAT + 2] = nh;

  buf_idx = min<uint8_t>(buf_idx + 1, SEQ_LEN);

  if (buf_idx == SEQ_LEN) {            // only after first 24 readings
    for (int i = 0; i < SEQ_LEN * NUM_FEAT; i++)
      ModelSetInput(input_buf[i], i);

    if (!ModelRunInference()) {
      Serial.println("❌ Inference failed");
    } else {
      /* De-normalise prediction so we can compare next loop */
      predNext.m = ModelGetOutput(0) * 100.0;         // back to %
      predNext.t = ModelGetOutput(1) * 60.0 - 10.0;   // back to °C
      predNext.h = ModelGetOutput(2) * 100.0;         // back to %
      predNext.valid = true;

      Serial.printf("PRED ▼ Moist:%.1f%%  Temp:%.1f°C  Hum:%.1f%%\n",
                    predNext.m, predNext.t, predNext.h);
    }
  }

  /* ───── 6. RAW DEBUG PRINTOUT ───── */
  Serial.printf("RAW → ADC:%d  Moist:%.1f%%  Temp:%.1f°C  Hum:%.1f%%\n\n",
                adcRaw, vwcPct, tempC, rhPct);

  delay(2000);   // 2-second sample rate (fits TS 15-s rule easily)
}
