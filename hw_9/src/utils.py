import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def prepare_data():
    df = pd.read_csv('data/realty_data.csv')
    df = df[['price', 'rooms', 'floor', 'lat', 'lon']]
    df.dropna(inplace=True)
    return df


def train_model(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns='price'),
        df['price'],
        random_state=2025,
        test_size=0.3
    )

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    with open('lr_fitted.pkl', 'wb') as file:
        pickle.dump(lr, file)


def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not exists")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model
