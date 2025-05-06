import os

import pandas as pd
import streamlit as st
from src.utils import prepare_data, train_model, read_model

st.set_page_config(
    page_title="Real estate price prediction",
)

model_path = 'lr_fitted.pkl'

floor = st.sidebar.number_input("What is the floor?", 1, 70, 1)
rooms_select = st.sidebar.selectbox(
    "How many rooms?",
    (1, 2, 3, 4, 5, 6, 7, 8),
)
lon = st.sidebar.number_input("What is the longitude of the apartment location?", 37.2, 38.0, 37.5)
lat = st.sidebar.number_input("What is the latitude of the apartment location?", 55.5, 56.5, 56.0)


# create input DataFrame
inputDF = pd.DataFrame(
    {
        "rooms": rooms_select,
        "floor": floor,
        "lat": lat,
        "lon": lon
    },
    index=[0],
)


if not os.path.exists(model_path):
    train_data = prepare_data()
    train_data.to_csv('data.csv')
    train_model(train_data)

model = read_model('lr_fitted.pkl')

preds = model.predict(inputDF)

st.write(f"Approximate price of real estate: {int(preds)} rubles.")
