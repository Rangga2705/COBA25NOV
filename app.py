
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('best_model.pkl')

st.title('Prediksi Klasifikasi Kemiskinan')
st.write('Aplikasi untuk memprediksi kelas kemiskinan berdasarkan input numerik.')

# Input user
def user_input_features():
    data = {}
    for col in model.feature_names_in_:
        val = st.sidebar.number_input(col, value=0.0)
        data[col] = val
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()
st.subheader('Input User:')
st.write(df_input)

if st.sidebar.button('Prediksi'):
    pred = model.predict(df_input)
    st.subheader('Hasil Prediksi:')
    st.write(pred[0])
