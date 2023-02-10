import pandas as pd
from markovBike.manager.manager import Manager

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import math


def preprocess_stations_data(df, drops=[], verbose=True):

    numerical_features, categorical_features, boolean_features = Manager.split_dataframe(
        df, drops=drops, verbose=verbose)

    numeric_transformer = Pipeline(
        steps=[('imputer',
                SimpleImputer(strategy='most_frequent')),
               ('scaler',
                StandardScaler())
               ])

    categorical_transformer = Pipeline(
        steps=[('onehot',
                OneHotEncoder(handle_unknown='ignore', drop='if_binary')),
               ('imputer',
                SimpleImputer(strategy='most_frequent', fill_value='missing')
                )])

    boolean_transformer = Pipeline(steps=[('onehot',
                                           OneHotEncoder(drop='if_binary'))])

    preprocessor = ColumnTransformer(
        transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('bool', boolean_transformer, boolean_features)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X = pipeline.fit_transform(df)

    categorical_onehot_features = preprocessor.named_transformers_[
        'cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

    print(categorical_onehot_features)

    boolean_onehot_features = preprocessor.named_transformers_[
        'bool'].named_steps['onehot'].get_feature_names_out(boolean_features)

    print(boolean_onehot_features)

    transformed_columns = list(numerical_features) + list(categorical_onehot_features) + list(boolean_onehot_features)

    print(transformed_columns)

    return X, transformed_columns


class CustomFeaturesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_age=True, add_sine_cosine=True):
        self.add_age = add_age
        self.add_sine_cosine = add_sine_cosine

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X,
                          columns=[
                              "tripduration", "starttime", "stoptime",
                              "start_station_latitude",
                              "start_station_longitude",
                              "end_station_latitude", "end_station_longitude",
                              "bikeid", "usertype", "birth_year", "gender"
                          ])

        if self.add_age:
            df['age'] = (datetime.now().year - df['birth_year'])

        if self.add_sine_cosine:
            df['starttime_sin'] = np.sin(2 * np.pi * df['starttime'].dt.hour /
                                         24)
            df['starttime_cos'] = np.cos(2 * np.pi * df['starttime'].dt.hour /
                                         24)

        print(df.columns)
        return df


def preprocess_trips_data(df, drops=[], verbose=True):
    # Define the columns to be encoded, scaled and imputed
    numerical_features, categorical_features, boolean_features = Manager.split_dataframe(df, drops=drops, verbose=verbose)

    # Create the preprocessing pipelines for the categorical, numerical and boolean columns
    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder())])
    numerical_transformer = Pipeline(
        steps=[('scaler', StandardScaler())])
    custom_features_adder = Pipeline(
        steps=[('adder', CustomFeaturesAdder())])

    # Use the ColumnTransformer to apply the transformations to the appropriate columns
    preprocessor = ColumnTransformer(transformers=[(
        'num', numerical_transformer, numerical_features
    ), ('cat', categorical_transformer,
        categorical_features)])

    # Create the final pipeline that includes the preprocessor and custom feature adder
    pipeline = Pipeline(
        steps=[('custom_features_adder', custom_features_adder),
               ('preprocessor', preprocessor)
               ])

    # Fit the pipeline to the training data
    pipeline.fit(df)

    X = pipeline.transform(df)

    categorical_onehot_features = preprocessor.named_transformers_[
        'cat'].named_steps['onehot'].get_feature_names_out(
            categorical_features)


    transformed_columns = list(numerical_features) + list(categorical_onehot_features) + [
            'age', 'starttime_sin', 'starttime_cos'
        ]

    return X, transformed_columns
