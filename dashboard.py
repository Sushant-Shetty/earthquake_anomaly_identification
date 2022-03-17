import streamlit as st
import time 
import csv
from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotting_data = []
plotting_row = []
index = 0
look_back = 0
 

df = pd.read_csv("2_year_data_30min.csv")
data = df[['Radon']]
prediction = data.mean()

st.title("Earthquake Anomaly Detection using LSTM")
window_size = st.selectbox('Please select Window Size', (1, 2))
st.write('Selected window size:', window_size)

if window_size == 1: look_back = 48
elif window_size == 2: look_back = 96
  
if st.button('Fetch data'):
  data_to_be_loaded = (data.iloc[index:look_back].T).to_numpy()
  data_to_be_loaded = np.reshape(data_to_be_loaded, (data_to_be_loaded.shape[0], 1, data_to_be_loaded.shape[1]))
  index += 1
  look_back += 1
  json_file = open(f'model_{window_size}.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  loaded_model.load_weights(f"model_{window_size}.h5")
  prediction = loaded_model.predict(data_to_be_loaded)
  plotting_data.append(prediction)
  
  
st.subheader("Predicted radon values")
st.write('Selected window size:', prediction)
st.write('Selected window size:', plotting_row)
# st.line_chart(pd.DataFrame(pd.DataFrame(plotting_data), pd.DataFrame(plotting_row)))
  
  
  
  

# now = time.time()


# while True:
#   temp_time = time.time()
#   if abs(now - temp_time) >= 10:
    
#     plotting_row.append(index)
#     data_to_be_loaded = data.iloc[index:look_back]
#     data_to_be_loaded = data_to_be_loaded.T
#     data_to_be_loaded = data_to_be_loaded.to_numpy()
#     data_to_be_loaded = np.reshape(data_to_be_loaded, (data_to_be_loaded.shape[0], 1, data_to_be_loaded.shape[1]))
# #     with open("2_year_data_30_min.csv") as data_file:
# #       reader = csv.reader(data_file)
# #       data_to_be_loaded = [data_row for idx, data_row in enumerate(reader) if idx == index]
#     index += 1
#     look_back += 1
  
#     json_file = open(f'model_{window_size}.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(f"model_{window_size}.h5")
#     prediction = loaded_model.predict(data_to_be_loaded)
# #     prediction = 
#     plotting_data.append(prediction.tolist())
  
# st.subheader("Predicted radon values")
# st.write('Selected window size:', type(prediction))
# st.write('Selected window size:', type(plotting_row))
# st.line_chart(pd.DataFrame(pd.DataFrame(plotting_data), pd.DataFrame(plotting_row)))
  
  
      
    
  
