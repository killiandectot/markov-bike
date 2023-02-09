import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import math

def preprocess_stations_data(df):
    numeric_features = ['latitude', 'longitude', 'capacity']
    categorical_features = ['rental_methods', 'eightd_has_key_dispenser', 'is_installed', 'is_renting', 'is_returning', 'eightd_has_available_keys']
    boolean_features = ['eightd_has_key_dispenser', 'is_installed', 'is_renting', 'is_returning', 'eightd_has_available_keys']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))
    ])

    boolean_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X = pipeline.fit_transform(df)

    categorical_onehot_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    boolean_onehot_features = preprocessor.named_transformers_['bool'].named_steps['onehot'].get_feature_names_out(boolean_features)
    transformed_columns = numeric_features + list(categorical_onehot_features) + list(boolean_onehot_features)

    return X, transformed_columns

class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_age = True, add_sine_cosine = True):
        self.add_age = add_age
        self.add_sine_cosine = add_sine_cosine

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=["tripduration", "starttime", "stoptime", "start_station_latitude",
                                      "start_station_longitude", "end_station_latitude", "end_station_longitude",
                                      "bikeid", "usertype", "birth_year", "gender"])

        if self.add_age:
            df['age'] = (datetime.now().year - df['birth_year'])

        if self.add_sine_cosine:
            df['starttime_sin'] = np.sin(2*np.pi*df['starttime'].dt.hour/24)
            df['starttime_cos'] = np.cos(2*np.pi*df['starttime'].dt.hour/24)

        return df


def preprocess_trips_data(df):
    # Define the columns to be encoded, scaled and imputed
    categorical_columns = ["usertype", "gender"]
    numerical_columns = ["tripduration", "start_station_latitude", "start_station_longitude", "end_station_latitude", "end_station_longitude", "bikeid", "birth_year", "age", "starttime_sin", "starttime_cos"]
    boolean_columns = []

    # Create the preprocessing pipelines for the categorical, numerical and boolean columns
    categorical_transformer = OneHotEncoder()
    numerical_transformer = StandardScaler()
    custom_features_adder = CustomFeaturesAdder()

    # Use the ColumnTransformer to apply the transformations to the appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_columns),
            ('num', numerical_transformer, numerical_columns)
        ])

    # Create the final pipeline that includes the preprocessor and custom feature adder
    pipeline = Pipeline(steps=[('custom_features_adder', custom_features_adder),
                               ('preprocessor', preprocessor)])

    # Fit the pipeline to the training data
    pipeline.fit(df)

    columns = categorical_columns + numerical_columns + ['age', 'starttime_sin', 'starttime_cos']

    return pipeline.transform(df), columns
