import streamlit as st
import time 
import csv
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd

st.title("Earthquake Anomaly Detection using Delegated Regressors")

window_size = st.selectbox('Please select Window Size', (1, 2))
st.write('Selected window size:', window_size)

df = pd.read_csv("2_year_data_30min.csv")
data = df['Radon']
look_back = 0
if window_size == 1: data = data.iloc[:48]
elif window_size == 2: data = data.iloc[:96]


now = time.time()
plotting_data = []
plotting_row = []
index = 0

while True:
  temp_time = time.time()
  if abs(now - temp_time) >= 10:
    plotting_row.append(index)
#     with open("2_year_data_30_min.csv") as data_file:
#       reader = csv.reader(data_file)
#       data_to_be_loaded = [data_row for idx, data_row in enumerate(reader) if idx == index]
    index += 1
  
    json_file = open(f'model_{window_size}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"model_{window_size}.h5")
    prediction = loaded_model.predict(data_to_be_loaded)
    plotting_data.append(prediction)
  
    st.subheader("Predicted radon values")
    st.line_chart(plotting_data, plotting_row)
  
  
      
    
  
