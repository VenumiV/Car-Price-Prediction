import pandas as pd
import numpy as np 
import pickle
import pickle as pk
import streamlit as st
from sklearn.preprocessing import LabelEncoder


model = pk.load(open('model.pkl','rb'))


st.header('Car Prediction ML Model')

cars_data = pd.read_csv('car_price_dataset.csv')

brand = st.selectbox('Select Car Brand',cars_data['Brand'].unique())

model = st.selectbox('Select Car Model',cars_data['Model'].unique())

yom = st.slider ('Car Manufactured Year',1950,2025)

engine = st.slider ('Engine Capacity',500,2000)

gear = st.selectbox('Select Gear Type',cars_data['Gear'].unique())

fuel_type = st.selectbox('Select Fuel Type',cars_data['Fuel Type'].unique())

millage = st.slider ('Milege (km)',11,1000000)

town = st.selectbox('Select Town ',cars_data['Town'].unique())

leasing = st.selectbox('Select Leasing Type',cars_data['Leasing'].unique())

condition = st.selectbox('Select Condition',cars_data['Condition'].unique())

air_condition = st.selectbox('Select Air Condition Type',cars_data['AIR CONDITION'].unique())

power_steering = st.selectbox('Select Steering Type',cars_data['POWER STEERING'].unique())

power_mirror = st.selectbox('Select Mirro Type',cars_data['POWER MIRROR'].unique())

power_window = st.selectbox('Select Window Type',cars_data['POWER WINDOW'].unique())

if st.button("Predict"):
      input_data_model = pd.DataFrame(
    [[
    brand, model, yom, engine , gear, fuel_type,
    millage, town, leasing, condition, air_condition,
    power_steering, power_mirror, power_window
]],

    columns = [
    'Brand', 'Model', 'YOM', 'Engine (cc)', 'Gear', 'Fuel Type',
    'Millage(KM)', 'Town', 'Leasing', 'Condition', 'AIR CONDITION',
    'POWER STEERING', 'POWER MIRROR', 'POWER WINDOW'
])

      input_data_model['Fuel Type'] = input_data_model['Fuel Type'].replace(
    ['Petrol', 'Diesel', 'Hybrid', 'Electric'], 
    [1, 2,3,4]
)
      input_data_model['Gear'] = input_data_model['Gear'].replace(
    ['Automatic', 'Manual'],  # ✅ Comma between items
    [1,2]
)
      input_data_model['Leasing'] = input_data_model['Leasing'].replace(
    ['No Leasing', 'Ongoing Lease'], 
    [1, 2]
)
      input_data_model['Condition'] = input_data_model['Condition'].replace(
    ['USED', 'NEW'], 
    [1, 2]
)
      input_data_model['AIR CONDITION'] = input_data_model['AIR CONDITION'].replace(
    ['Available', 'Not_Available'], 
    [1, 2]
)
      input_data_model['POWER STEERING'] = input_data_model['POWER STEERING'].replace(
    ['Available', 'Not_Available'], 
    [1, 2]
)
      input_data_model['POWER WINDOW'] = input_data_model['POWER WINDOW'].replace(
    ['Available', 'Not_Available'], 
    [1, 2]
)
      input_data_model['POWER MIRROR'] = input_data_model['POWER MIRROR'].replace(
    ['Available', 'Not_Available'], 
    [1, 2]

)
      le_brand = LabelEncoder()
      input_data_model['Brand'] = le_brand.fit_transform(input_data_model['Brand'])

      le_model = LabelEncoder()
      input_data_model['Model'] = le_model.fit_transform(input_data_model['Model'])

      le_town = LabelEncoder()
      input_data_model['Town'] = le_town.fit_transform(input_data_model['Town'])
   
with open('model.pkl', 'rb') as file:
    model = pk.load(file)
    prediction = model.predict(input_data_model)
    st.markdown("**Predicted Car Price:** " + str(prediction[0]))

