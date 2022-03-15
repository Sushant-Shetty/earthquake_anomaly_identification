import streamlit as st

st.title("Earthquake Anomaly Detection using Delegated Regressors")

window_size = st.selectbox('Please select Window Size', (1, 2, 3, 4))
st.write('Selected window size:', window_size)