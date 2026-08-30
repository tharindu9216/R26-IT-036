# Component 1: BVP + EDA Stress Model Streamlit Tester

This is a simple local test harness for the saved BVP + EDA stress models.

## Why the separate ESP32 test sketch?

Your current project sketch sends summary fields to Firestore every 5 seconds. That is not enough to reproduce the raw 60-second `[EDA, BVP]` window required by CNN-LSTM, and it does not preserve every pulse interval needed for the engineered-feature models.

The project sketch at
`iot/c1_stress_detection_bvp/streamlit_serial/streamlit_serial.ino` streams raw
sensor samples over USB serial so the PC can reconstruct the same 60-second
processing pipeline used during training.

## Model folder

The app automatically discovers notebook output runs anywhere below the
project-level `models/c1_stress_detection_bvp/` folder. The folders may be
nested; each individual run must have this structure:

```text
models/c1_stress_detection_bvp/
└── BVP_EDA/any-run/
    ├── run_manifest.json
    ├── dl_params.json
    └── models/
        ├── logistic_regression.pkl
        ├── random_forest.pkl
        ├── svm_rbf.pkl
        ├── histgradientboosting.pkl
        ├── feature_mlp.pt
        ├── feature_mlp_prep.pkl
        ├── cnn_lstm.pt
        └── cnn_lstm_prep.pkl
```

Only complete model artifacts that actually exist are loaded. If multiple runs
are found, choose one from the **Model run** selector in the sidebar; the run
with the most model artifacts is selected by default.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r streamlit_app/c1_stress_detection_bvp/requirements.txt
python -m streamlit run streamlit_app/c1_stress_detection_bvp/app.py
```

The requirements pin scikit-learn 1.6.1 because that is the version used to
serialize the included classical model pipelines.

Then choose the ESP32 serial port in the sidebar.

## Live flow

1. Flash `iot/c1_stress_detection_bvp/streamlit_serial/streamlit_serial.ino`.
2. Open the Streamlit app.
3. Connect to the ESP32 serial port at 230400 baud.
4. Stay quiet/still for 5 minutes for personal calibration.
5. Collect another 60 seconds.
6. The app runs every loaded model and displays stress probability + threshold + class.
7. New predictions are generated about every 5 seconds from the latest 60-second window.

## Important limitation

The notebook's EDA peak detector uses a fixed phasic prominence of `0.01` in WESAD/Empatica EDA units. The CJMCU-6701 provides ADC units. Personal z-score normalization helps many features, but it does not make this pre-feature-extraction peak threshold unit-invariant.

So treat live predictions as prototype results until the EDA peak features or CJMCU scaling are validated on real hardware.

## Safety

When body-contact EDA electrodes are attached, use a battery-powered laptop/power source where practical and avoid mains-grounded test equipment connected to the body-contact circuit.
