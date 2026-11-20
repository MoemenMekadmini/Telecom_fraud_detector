# Telecom Fraud Detector

A Streamlit web app that classifies telecom call detail records (CDRs) as genuine or fraudulent, using an XGBoost model. Upload a CSV, get a per-record fraud prediction, review the flagged calls, and download the results.
App url : https://telecomfrauddetector.streamlit.app/

## Background

Telecom fraud has rapidly become a major problem within the telecommunications industry, costing companies billions of dollars annually all around the globe from dangerous scams such as: SIMBox, Subscription fraud, IRSF...

With the rise of technology and various sophisticated schemes, detecting these types of fraud using basicrule-based systems has become more challenging. Therefore, we propose building a Machine Learning (ML) model that successfully detects these types of fraud more efficiently using algorithms such as XGBoost and Random Forest to prevent further security breaches and minimize financial losses.
The provided solution is expected to improve detection accuracy while reducing false positives compared to rule-based systems.

## What the app does

1. **Upload** a CSV of call detail records.
2. **Engineer features** automatically: call counts per SIM and per device, and their ratio.
3. **Predict** a class for every record with the trained XGBoost model.
4. **Review** results: predictions are highlighted in the table, with separate tables for each fraud type and a summary chart.
5. **Export** the full results as a CSV.

### Predicted classes

| Class | Meaning |
|---|---|
| `none` | Genuine traffic |
| `sim_box_fraud` | Traffic routed through a SIM box to bypass international termination fees |
| `subscription_fraud` | Service obtained with false or stolen identity, typically with no intent to pay |

### Input format

The uploaded CSV needs these columns:

| Column | Description |
|---|---|
| `Duration` | Call length in seconds |
| `IMSI` | Subscriber (SIM) identifier |
| `IMEI` | Device identifier |
| `Call_type` | `local` or `international` |

A ready-made sample is in [`sample_data/test_cdr.csv`](sample_data/test_cdr.csv).

## Model

| | |
|---|---|
| **Algorithm** | XGBoost classifier (Random Forest used as a baseline) |
| **Dataset** | 24,543 CDRs, reduced to 18,408 after removing the `random_fraud` and `call_masking` classes |
| **Split** | 70/30 train/test, stratified, `random_state=42` |
| **Features** | `Duration`, `IMSI_call_traffic`, `IMEI_call_traffic`, `IMSI/IMEI`, one-hot `Call_type` |
| **Scaling** | MinMax scaling on the numeric features |
| **Test accuracy** | 0.86 |
| **Macro F1** | 0.81 (Random Forest baseline: 0.79) |

Feature engineering:
- `IMSI_call_traffic`: number of calls made by the same SIM
- `IMEI_call_traffic`: number of calls made from the same device
- `IMSI/IMEI`: ratio of the two, which helps expose one device cycling through many SIMs

The full workflow (exploration, cleaning, feature engineering, training, evaluation) is in [`Telecom_fraud_detector.ipynb`](Telecom_fraud_detector.ipynb).

## Tech stack

- **Language:** Python 3.13
- **App framework:** Streamlit
- **Machine learning:** XGBoost, scikit-learn
- **Data handling:** pandas, NumPy
- **Model persistence:** joblib
- **Deployment:** Streamlit Community Cloud

## Run locally

```bash
git clone https://github.com/MoemenMekadmini/Telecom_fraud_detector.git
cd Telecom_fraud_detector
pip install -r requirements.txt
streamlit run app1.py
```

The app opens at `http://localhost:8501`. Upload `sample_data/test_cdr.csv` to try it.

## Project structure

```
├── app1.py                          # Streamlit app
├── Telecom_fraud_detector.ipynb     # Data exploration and model training
├── model.joblib                     # Trained XGBoost model
├── scaler.joblib                    # Fitted MinMaxScaler
├── label_encoder.joblib             # Fitted label encoder for the classes
├── requirements.txt                 # Python dependencies
├── sample_data/test_cdr.csv         # Sample input for testing the app
└── getwireless.png                  # App logo
```

## Limitations

- The model was trained on a single labelled dataset and may not generalise to other operators' traffic without retraining.
- Call counts are computed within the uploaded file, so predictions depend on the file containing enough of each SIM's and device's activity.
- Only SIM box and subscription fraud are detected. Other fraud types such as call masking and IRSF are not covered.
- Predictions are decision support and should be reviewed before action is taken on a subscriber.

## License

Released under the [MIT License](LICENSE).
