import streamlit as st
import numpy as np
import joblib

model =joblib.load('rf_model.pkl')

st.title('Laptop Price Predicter')

st.divider()

st.write('here you predict price of laptop using given feature')

st.divider()

processor_speed = st.number_input('Enter Processor Speed',value = 2.50,step = 0.50)
ram_size = st.number_input('Enter RAM Size',value = 16,step = 8)
storage_capacity = st.number_input('Enter Storage Capacity',value = 512, step = 256)

x = [processor_speed,ram_size,storage_capacity]

st.divider()

prediction = st.button('Price Estimation Button')

st.divider()


if prediction:
    x1 = np.array(x)
    prediction =model.predict([x1])[0]
    st.write(f"Price Estimation For Laptop Is {prediction:.2f}")

    
