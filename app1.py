import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')


FEATURE_NAMES = [
    'Duration',
    'IMSI_call_traffic',
    'IMEI_call_traffic',
    'IMSI/IMEI',
    'Call_type_international',
    'Call_type_local',
]
SCALE_FEATURES = ['Duration', 'IMSI_call_traffic', 'IMEI_call_traffic', 'IMSI/IMEI']


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    if 'IMSI' in df.columns and 'IMSI_call_traffic' not in df.columns:
        df['IMSI_call_traffic'] = df.groupby('IMSI')['IMSI'].transform('count')
    if 'IMEI' in df.columns and 'IMEI_call_traffic' not in df.columns:
        df['IMEI_call_traffic'] = df.groupby('IMEI')['IMEI'].transform('count')

    
    if 'IMSI_call_traffic' in df.columns and 'IMEI_call_traffic' in df.columns:
        df['IMSI/IMEI'] = df['IMSI_call_traffic'] / df['IMEI_call_traffic'].replace(0, np.nan)
        df['IMSI/IMEI'] = df['IMSI/IMEI'].fillna(0)

    if 'Call_type' in df.columns:
        call_type_dummies = pd.get_dummies(df['Call_type'], prefix='Call_type')
        for col in ['Call_type_international', 'Call_type_local']:
            if col not in call_type_dummies.columns:
                call_type_dummies[col] = 0
        df = pd.concat([df, call_type_dummies[['Call_type_international', 'Call_type_local']]], axis=1)

    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        raise ValueError(
            'Missing required feature columns. Upload a CSV with these columns please :Duration(seconds), IMSI, IMEI and Call_type.  '
            + ', '.join(missing_features)
        )

    final_df = df[FEATURE_NAMES].copy()
    final_df[SCALE_FEATURES] = scaler.transform(final_df[SCALE_FEATURES])
    return df, final_df


col1, col2 = st.columns([3, 1])

with col1:
    st.title('Telecom Fraud Detector')

with col2:
    st.image('getwireless.png')

st.write('Upload a CSV file containing these call detail records: Duration(seconds), IMSI, IMEI and Call_type. The app will predict the type of fraud for each record using an XGBoost model trained on historical CDRs with a 0.81 f1-score.')

uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f'Unable to read CSV: {exc}')
    else:
        st.write('Data Preview:')
        st.dataframe(df.head())

        try:
            raw_df, features_df = preprocess(df)
        except Exception as exc:
            st.error(str(exc))
        else:
            preds = model.predict(features_df.values)
            if hasattr(label_encoder, 'inverse_transform'):
                try:
                    preds = label_encoder.inverse_transform(preds)
                except Exception:
                    pass

            raw_df['Predicted_Fraud_Type'] = preds
            st.write('Predictions:')
            
            def highlight_fraud_types(row):
                if row['Predicted_Fraud_Type'] in ['sim_box_fraud', 'subscription_fraud']:
                    return ['background-color: red'] * len(row)
                return [''] * len(row)
            
            styled_df = raw_df.style.apply(highlight_fraud_types, axis=1)
            st.dataframe(styled_df)

            simbox_fraud_df = raw_df[raw_df['Predicted_Fraud_Type'] == 'sim_box_fraud']
            subscription_fraud_df = raw_df[raw_df['Predicted_Fraud_Type'] == 'subscription_fraud']
            
            st.write('Simbox Fraud Records:')
            st.dataframe(simbox_fraud_df)
            
            st.write('Subscription Fraud Records:')
            st.dataframe(subscription_fraud_df)

            fraud_counts = raw_df['Predicted_Fraud_Type'].value_counts()
            st.write('Fraud Detection Summary:')
            st.bar_chart(fraud_counts)

            result_csv = raw_df.to_csv(index=False)
            st.download_button(
                label='Download results as CSV',
                data=result_csv,
                file_name='fraud_predictions.csv',
                mime='text/csv',
            )
