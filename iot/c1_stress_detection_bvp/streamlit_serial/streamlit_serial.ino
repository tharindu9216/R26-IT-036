// ============================================================
// ESP32 RAW SERIAL STREAM FOR STREAMLIT MODEL TESTING
// MAX30102 + CJMCU-6701 analog OUT
// Output: DATA,<millis>,<MAX30102_IR>,<GSR_ADC>
// ============================================================

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;
#define GSR_OUT_PIN 34

const byte LED_BRIGHTNESS = 0x1F;
const byte SAMPLE_AVERAGE  = 4;
const byte LED_MODE        = 2;
const int  SAMPLE_RATE     = 100;
const int  PULSE_WIDTH     = 411;
const int  ADC_RANGE       = 4096;

void setup() {
  Serial.begin(230400);
  delay(1000);

  pinMode(GSR_OUT_PIN, INPUT);
  analogSetPinAttenuation(GSR_OUT_PIN, ADC_11db);
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR,MAX30102_NOT_FOUND");
    while (true) delay(1000);
  }

  particleSensor.setup(
    LED_BRIGHTNESS,
    SAMPLE_AVERAGE,
    LED_MODE,
    SAMPLE_RATE,
    PULSE_WIDTH,
    ADC_RANGE
  );

  particleSensor.setPulseAmplitudeRed(LED_BRIGHTNESS);
  particleSensor.setPulseAmplitudeGreen(0);

  Serial.println("READY");
}

void loop() {
  particleSensor.check();

  while (particleSensor.available()) {
    long irValue = particleSensor.getFIFOIR();
    int gsrRaw = analogRead(GSR_OUT_PIN);
    unsigned long t = millis();

    Serial.print("DATA,");
    Serial.print(t);
    Serial.print(",");
    Serial.print(irValue);
    Serial.print(",");
    Serial.println(gsrRaw);

    particleSensor.nextSample();
  }
}


// #define GSR_OUT_PIN 34

// void setup() {
//   Serial.begin(115200);
//   analogSetPinAttenuation(GSR_OUT_PIN, ADC_11db);
// }

// void loop() {
//   int raw = analogRead(GSR_OUT_PIN);

//   Serial.print("EDA raw: ");
//   Serial.println(raw);

//   delay(200);
// }
