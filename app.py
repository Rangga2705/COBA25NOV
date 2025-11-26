import streamlit as st
import pandas as pd
import joblib

# --------------------------
# Load model
# --------------------------
data = joblib.load('best_model.pkl')
model = data['model']
feature_columns = data['columns']
scaler = data['scaler']

st.title('Prediksi Klasifikasi Kemiskinan')
st.write('Aplikasi untuk memprediksi kelas kemiskinan berdasarkan input numerik.')

# --------------------------
# Input user
# --------------------------
def user_input_features():
    data = {}
    for col in feature_columns:
        val = st.sidebar.number_input(f"{col}", value=0.0)
        data[col] = val
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()
st.subheader('Input User:')
st.write(df_input)

# --------------------------
# Prediksi
# --------------------------
if st.sidebar.button('Prediksi'):
    try:
        # scaling
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)
        st.subheader('Hasil Prediksi:')
        st.write(pred[0])
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.exception(e)
