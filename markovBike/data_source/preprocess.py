from datetime import datetime
from colorama import Fore, Style
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def preprocess_stations_data(
    dataframe: pd.DataFrame,
    index: str = None,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
    boolean_features: List[str] = None,
    drops: List[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocesses a dataframe containing information on stations.

    Args:
        dataframe: The input dataframe.
        index: The name of the column to use as the index of the dataframe.
        numerical_features: A list of the names of the numerical features in the dataframe.
        categorical_features: A list of the names of the categorical features in the dataframe.
        boolean_features: A list of the names of the boolean features in the dataframe.
        drops: A list of columns to drop from the dataframe.
        verbose: Whether to print out information about the dataframe and features.

    Returns:
        A tuple containing the preprocessed dataframe and a list of the names of the transformed columns.
    """
    # Set the index of the dataframe
    if index:
        dataframe.set_index(index, inplace=True)

    # Drop specified columns
    if drops:
        dataframe.drop(drops, axis=1, inplace=True)

    # Print out information about the dataframe and features
    if verbose:
        print(f"\n 📶 {Fore.CYAN}Columns {list(dataframe.columns)}{Style.RESET_ALL}")
        print(f"\n 📶 {Fore.CYAN}Numericals {numerical_features}{Style.RESET_ALL}")
        print(f"\n 📶 {Fore.CYAN}Categoricals {categorical_features}{Style.RESET_ALL}")
        print(f"\n 📶 {Fore.CYAN}Booleans {boolean_features}{Style.RESET_ALL}")

    # Define the transformers for numerical, categorical, and boolean features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
            ("imputer", SimpleImputer(strategy="most_frequent", fill_value="missing")),
        ]
    )

    boolean_transformer = Pipeline(steps=[("onehot", OneHotEncoder(drop="if_binary"))])

    # Combine the transformers using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
            ("bool", boolean_transformer, boolean_features),
        ]
    )

    # Create a pipeline with the preprocessor
    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Fit and transform the pipeline on the input dataframe
    X = pipeline.fit_transform(dataframe)

    # Get the names of the one-hot encoded columns for categorical and boolean features
    categorical_onehot_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )
    boolean_onehot_features = (
        preprocessor.named_transformers_["bool"]
        .named_steps["onehot"]
        .get_feature_names_out(boolean_features)
    )

    # Combine the numerical, categorical, and boolean features into a single list
    transformed_columns = (
        list(numerical_features)
        + list(categorical_onehot_features)
        + list(boolean_onehot_features)
    )

    return X, transformed_columns


class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer to add new features to a Pandas DataFrame.

    Parameters:
    ----------
    add_age: bool, default=True
        Whether to add the age of the user based on their birth year.
    add_sine_cosine: bool, default=True
        Whether to add the sine and cosine of the start time hour.

    Returns:
    -------
    df: pandas DataFrame
        A pandas DataFrame with added features.
    """

    def __init__(self, add_age=True, add_sine_cosine=True):
        self.add_age = add_age
        self.add_sine_cosine = add_sine_cosine

    def fit(self, X, y=None):
        """
        Returns the transformer object.

        Parameters:
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target values.

        Returns:
        -------
        self: object
        """
        return self

    def transform(self, X, y=None):
        """
        Adds new features to the input DataFrame.

        Parameters:
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples to be transformed.
        y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target values.

        Returns:
        -------
        df: pandas DataFrame
            A pandas DataFrame with added features.
        """
        # Create a pandas DataFrame from the input data
        df = pd.DataFrame(
            X,
            columns=[
                "tripduration",
                "starttime",
                "stoptime",
                "start_station_latitude",
                "start_station_longitude",
                "end_station_latitude",
                "end_station_longitude",
                "bikeid",
                "usertype",
                "birth_year",
                "gender",
                "start_station_id",
                "end_station_id",
            ],
        )

        # Add user age if specified
        if self.add_age:
            df["age"] = datetime.now().year - df["birth_year"]

        # Add sine and cosine of start time hour if specified
        if self.add_sine_cosine:
            df["starttime_sin"] = np.sin(2 * np.pi * df["starttime"].dt.hour / 24)
            df["starttime_cos"] = np.cos(2 * np.pi * df["starttime"].dt.hour / 24)

        # Return the pandas DataFrame with added features
        return df


def preprocess_trips_data(
    dataframe,
    index=None,
    numerical_features=None,
    categorical_features=None,
    boolean_features=None,
    drops=None,
    verbose=True,
):
    """
    Preprocesses a given pandas DataFrame by scaling numerical features, encoding categorical features, and adding custom features.

    Args:
    - dataframe: pandas DataFrame to preprocess.
    - index: name of the index column to set (default None).
    - numerical_features: list of column names containing numerical features (default None).
    - categorical_features: list of column names containing categorical features (default None).
    - boolean_features: list of column names containing boolean features (default None).
    - drops: list of column names to drop from the DataFrame (default None).
    - verbose: whether to print information about the preprocessed DataFrame (default True).

    Returns:
    - X: the preprocessed DataFrame.
    - transformed_columns: list of column names of the preprocessed DataFrame.
    """

    if index:
        dataframe.set_index(index, inplace=True)

    if drops:
        dataframe.drop(drops, axis=1, inplace=True)

    if verbose:
        print(
            "\n 📶 Columns " + Fore.CYAN + str(list(dataframe.columns)) + Style.RESET_ALL
        )

        print(
            "\n 📶 Numericals "
            + Fore.CYAN
            + str(list(numerical_features))
            + Style.RESET_ALL
        )

        print(
            "\n 📶 Categoricals "
            + Fore.CYAN
            + str(list(categorical_features))
            + Style.RESET_ALL
        )

        print(
            "\n 📶 Booleans " + Fore.CYAN + str(list(boolean_features)) + Style.RESET_ALL
        )

    # Create the preprocessing pipelines for the categorical, numerical and boolean columns
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(sparse=False, drop="if_binary"))]
    )
    numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    custom_features_adder = Pipeline(steps=[("adder", CustomFeaturesAdder())])

    # Use the ColumnTransformer to apply the transformations to the appropriate columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Create the final pipeline that includes the preprocessor and custom feature adder
    pipeline = Pipeline(
        steps=[
            ("custom_features_adder", custom_features_adder),
            ("preprocessor", preprocessor),
        ]
    )

    print(repr(dataframe.columns))
    dataframe.columns = list(dataframe.columns)
    print(repr(dataframe.columns))

    # Fit the pipeline to the training data
    pipeline.fit(dataframe)
    print("2")
    X = pipeline.transform(dataframe)

    categorical_onehot_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_features)
    )

    transformed_columns = (
        list(numerical_features)
        + list(categorical_onehot_features)
        + ["age", "starttime_sin", "starttime_cos"]
    )

    return X, transformed_columns
