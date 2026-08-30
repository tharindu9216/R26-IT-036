// ============================================================
// ESP32 + MAX30102 (Heart Rate) + CJMCU-6701 (GSR/EDA)
// → Firebase Cloud Firestore
// C1 Component — Fixed version
// ============================================================

#include <Wire.h>
#include <WiFi.h>
#include <SPI.h>
#include "MAX30105.h"
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"
#include <math.h>

// ─── FILL IN YOUR CREDENTIALS ────────────────────────────────
#define WIFI_SSID     "Dhanuka 11"       // <-- fill in
#define WIFI_PASSWORD "Nirosh2002"   // <-- fill in
#define API_KEY       "AIzaSyAsDJcVQF6gjj4iHiz15L1O2Fgizxz8wcw"
#define PROJECT_ID    "research-410a6"
#define COLLECTION    "heart_rate_readings"
// ─── CJMCU-6701 GSR Pins ─────────────────────────────────────
#define GSR_CS_PIN    5
#define GSR_MISO_PIN  19
#define GSR_SCK_PIN   18
#define GSR_OUT_PIN   34   // analog OUT

// ─── Firebase ────────────────────────────────────────────────
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// ─── MAX30102 ────────────────────────────────────────────────
MAX30105 particleSensor;

// ─── Beat Detection ──────────────────────────────────────────
float slowEMA     = 0;
float fastEMA     = 0;
bool  initialized = false;
bool  above       = false;
float cycleMaxGap = 0;
unsigned long lastBeatTime = 0;
const unsigned long REFRACTORY_MS     = 350;
const float         MIN_PULSE_AMPLITUDE = 800.0;
const unsigned long minRR = 400;
const unsigned long maxRR = 1500;

// ─── BPM ─────────────────────────────────────────────────────
const byte RATE_SIZE = 8;
float  rates[RATE_SIZE];
byte   rateSpot  = 0;
byte   rateCount = 0;
float  instantBPM = 0;
float  beatAvg    = 0;

// ─── RR + HRV ────────────────────────────────────────────────
const byte RR_SIZE = 8;
float  rrBuffer[RR_SIZE];
byte   rrSpot  = 0;
byte   rrCount = 0;

// ─── GSR Variables ───────────────────────────────────────────
float gsrRaw      = 0;
float gsrFiltered = 0;
float gsrBaseline = 0;
float gsrPhasic   = 0;
float gsrPrev     = 0;
int   gsrPeakCount = 0;

// FIX 2: baseline starts AFTER setup() completes
bool  gsrBaselineSet   = false;
bool  gsrBaselineReady = false; // true when we can start baseline timer
unsigned long gsrBaselineTimer = 0;
const unsigned long GSR_BASELINE_MS = 10000;

// ─── Timing ──────────────────────────────────────────────────
unsigned long lastSendTime = 0;
const unsigned long SEND_INTERVAL = 5000;
bool signupOK = false;

// ============================================================
float calculateRMSSD() {
  if (rrCount < 2) return 0.0;
  float sumSq = 0.0; byte n = 0;
  for (byte i = 1; i < RR_SIZE; i++) {
    if (rrBuffer[i] > 0 && rrBuffer[i-1] > 0) {
      float d = rrBuffer[i] - rrBuffer[i-1];
      sumSq += d * d; n++;
    }
  }
  return n == 0 ? 0.0 : sqrt(sumSq / n);
}

float calculateSDNN() {
  if (rrCount < 2) return 0.0;
  float sum = 0, sumSq = 0; byte n = 0;
  for (byte i = 0; i < RR_SIZE; i++) {
    if (rrBuffer[i] > 0) {
      sum += rrBuffer[i];
      sumSq += rrBuffer[i] * rrBuffer[i];
      n++;
    }
  }
  if (n < 2) return 0.0;
  float mean = sum / n;
  return sqrt((sumSq / n) - (mean * mean));
}

bool isPlausibleRR(float newRR) {
  if (rrCount == 0) return true;
  byte lastIdx = (rrSpot == 0) ? RR_SIZE - 1 : rrSpot - 1;
  float lastRR = rrBuffer[lastIdx];
  if (lastRR <= 0) return true;
  return fabs(newRR - lastRR) / lastRR < 0.40;
}

// ============================================================
bool detectBeat(long irValue) {
  float ir = (float)irValue;
  if (!initialized) {
    slowEMA = ir; fastEMA = ir;
    initialized = true;
    return false;
  }
  slowEMA = 0.02 * ir + 0.98 * slowEMA;
  fastEMA = 0.4  * ir + 0.6  * fastEMA;
  float gap = fastEMA - slowEMA;
  bool currentlyAbove = (gap > 0);
  if (currentlyAbove && gap > cycleMaxGap) cycleMaxGap = gap;

  bool beatDetected = false;
  if (currentlyAbove && !above) {
    above = true;
    cycleMaxGap = gap;
  } else if (!currentlyAbove && above) {
    above = false;
    unsigned long now   = millis();
    unsigned long delta = now - lastBeatTime;
    bool refractoryOK = (delta > REFRACTORY_MS);
    bool amplitudeOK  = (cycleMaxGap > MIN_PULSE_AMPLITUDE);

    if (lastBeatTime > 0 && refractoryOK && amplitudeOK &&
        delta > minRR && delta < maxRR) {
      instantBPM = 60000.0 / delta;
      if (isPlausibleRR((float)delta)) {
        rates[rateSpot++] = instantBPM;
        rateSpot %= RATE_SIZE;
        if (rateCount < RATE_SIZE) rateCount++;
        float s = 0;
        for (byte i = 0; i < rateCount; i++) s += rates[i];
        beatAvg = s / rateCount;
        rrBuffer[rrSpot++] = (float)delta;
        rrSpot %= RR_SIZE;
        if (rrCount < RR_SIZE) rrCount++;
        beatDetected = true;
      }
      lastBeatTime = now;
    } else if (!refractoryOK || !amplitudeOK) {
      // noise — skip
    } else {
      lastBeatTime = now;
    }
  }
  return beatDetected;
}

// ============================================================
void readGSR() {
  // FIX 1: configure ADC attenuation for GPIO34
  // This fixes Raw always reading 0
  analogSetPinAttenuation(GSR_OUT_PIN, ADC_11db);

  gsrRaw      = analogRead(GSR_OUT_PIN);
  gsrFiltered = 0.1 * gsrRaw + 0.9 * gsrFiltered;

  // FIX 2: only start baseline timer after setup is done
  if (!gsrBaselineReady) return;

  if (!gsrBaselineSet) {
    if (millis() - gsrBaselineTimer > GSR_BASELINE_MS) {
      gsrBaseline    = gsrFiltered;
      gsrBaselineSet = true;
      // FIX 3: reset peak count when baseline is set
      gsrPeakCount   = 0;
      Serial.println("GSR baseline set: " + String(gsrBaseline));
    } else {
      Serial.printf("GSR calibrating... Raw:%.0f Filtered:%.0f (%.0f sec left)\n",
                    gsrRaw, gsrFiltered,
                    (GSR_BASELINE_MS - (millis() - gsrBaselineTimer)) / 1000.0);
    }
    return;
  }

  gsrPhasic = gsrFiltered - gsrBaseline;

  // Peak detection
  float peakThreshold = 30.0;
  if (gsrPhasic > peakThreshold && gsrPrev <= peakThreshold) {
    gsrPeakCount++;
  }
  gsrPrev = gsrPhasic;
}

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(2000);

  memset(rrBuffer, 0, sizeof(rrBuffer));
  memset(rates,    0, sizeof(rates));

  // ── Init GSR pins ─────────────────────────────────────────
  Serial.println("Setting up CJMCU-6701 GSR...");
  pinMode(GSR_CS_PIN, OUTPUT);
  digitalWrite(GSR_CS_PIN, HIGH);
  SPI.begin(GSR_SCK_PIN, GSR_MISO_PIN, -1, GSR_CS_PIN);
  pinMode(GSR_OUT_PIN, INPUT);
  // FIX 1: set ADC attenuation once at startup too
  analogSetPinAttenuation(GSR_OUT_PIN, ADC_11db);
  Serial.println("GSR pins ready.");

  // ── Init MAX30102 ─────────────────────────────────────────
  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found! Check wiring.");
    while (true);
  }
  particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 ready.");

  // ── WiFi ─────────────────────────────────────────────────
  Serial.print("Connecting to WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  // ── Firebase ──────────────────────────────────────────────
  config.api_key = API_KEY;
  config.token_status_callback = tokenStatusCallback;
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase OK");
    signupOK = true;
  } else {
    Serial.println("Firebase error!");
  }
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  // FIX 2: start baseline timer AFTER everything is ready
  gsrBaselineReady = true;
  gsrBaselineTimer = millis();

  Serial.println("==========================================");
  Serial.println("C1 Component Ready!");
  Serial.println("Step 1: Wait 10 sec — GSR calibrating");
  Serial.println("Step 2: Place finger on MAX30102");
  Serial.println("Step 3: Hold completely still");
  Serial.println("==========================================");
}

// ============================================================
void loop() {

  // Read GSR every loop
  readGSR();

  // Read MAX30102
  long irValue = particleSensor.getIR();

  if (irValue < 20000) {
    Serial.println("No finger on MAX30102. Place finger flat...");
    initialized  = false;
    above        = false;
    lastBeatTime = 0;
    cycleMaxGap  = 0;
    delay(500);
    return;
  }

  bool beat = detectBeat(irValue);
  if (beat) {
    byte lastIdx = (rrSpot == 0) ? RR_SIZE - 1 : rrSpot - 1;
    Serial.printf("*** BEAT *** BPM:%.1f | Avg:%.1f | RR:%.0fms\n",
                  instantBPM, beatAvg, rrBuffer[lastIdx]);
  }

  float rmssd    = calculateRMSSD();
  float sdnn     = calculateSDNN();
  float latestRR = (rrCount > 0)
                   ? rrBuffer[(rrSpot == 0 ? RR_SIZE-1 : rrSpot-1)]
                   : 0;

  Serial.printf(
    "ECG→ BPM:%.1f Avg:%.1f RR:%.0f RMSSD:%.2f SDNN:%.2f | "
    "GSR→ Raw:%.0f Filtered:%.0f Phasic:%.0f Peaks:%d\n",
    instantBPM, beatAvg, latestRR, rmssd, sdnn,
    gsrRaw, gsrFiltered, gsrPhasic, gsrPeakCount
  );

  // Send to Firestore every 5 seconds
  if (Firebase.ready() && signupOK &&
      (millis() - lastSendTime > SEND_INTERVAL) &&
      beatAvg > 0 && rrCount >= 2 && gsrBaselineSet) {

    lastSendTime = millis();

    FirebaseJson content;
    content.set("fields/avg_bpm/doubleValue",        String(beatAvg, 1));
    content.set("fields/instant_bpm/doubleValue",    String(instantBPM, 1));
    content.set("fields/rr_interval_ms/doubleValue", String(latestRR, 1));
    content.set("fields/hrv_rmssd/doubleValue",      String(rmssd, 2));
    content.set("fields/hrv_sdnn/doubleValue",       String(sdnn, 2));
    content.set("fields/eda_raw/doubleValue",        String(gsrRaw, 1));
    content.set("fields/eda_filtered/doubleValue",   String(gsrFiltered, 1));
    content.set("fields/eda_phasic/doubleValue",     String(gsrPhasic, 1));
    content.set("fields/eda_peaks/integerValue",     String(gsrPeakCount));
    content.set("fields/eda_baseline/doubleValue",   String(gsrBaseline, 1));
    content.set("fields/timestamp/integerValue",     String(millis()));
    content.set("fields/device_id/stringValue",      "ESP32_C1_NODE");

    String docPath = String(COLLECTION) + "/" + String(millis());
    Serial.print("Sending to Firestore... ");
    if (Firebase.Firestore.createDocument(
          &fbdo, PROJECT_ID, "", docPath.c_str(), content.raw())) {
      Serial.printf("SUCCESS | BPM:%.1f | GSR:%.0f | Phasic:%.0f\n",
                    beatAvg, gsrFiltered, gsrPhasic);
    } else {
      Serial.println("FAILED: " + String(fbdo.errorReason().c_str()));
    }
  }
}
