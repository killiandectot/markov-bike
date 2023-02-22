import pandas as pd
import requests
from typing import Optional
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


def calculate_daily_rides(
    dataframe: pd.DataFrame, date_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Given a Pandas DataFrame containing bike ride data, calculates the number of rides per day.

    Args:
        dataframe: A Pandas DataFrame containing bike ride data. Must contain a 'starttime' column.
        date_column: Optional. The name of the column to use for the date. If not provided, a new column called 'date'
                     will be added to the DataFrame.

    Returns:
        A Pandas DataFrame with two columns: 'date', which contains the date of each ride, and 'daily_rides',
        which contains the number of rides on that date.
    """
    # Convert the starttime column to a Pandas datetime object
    dataframe["starttime"] = pd.to_datetime(dataframe["starttime"])

    # Add a new column with just the date portion of the starttime column
    if date_column is None:
        dataframe["date"] = dataframe["starttime"].dt.date
        date_column = "date"
    else:
        dataframe[date_column] = dataframe["starttime"].dt.date

    # Count the number of rows (i.e. bike rides) per day
    daily_rides = (
        dataframe.groupby(date_column).size().reset_index(name="daily_n_rides")
    )

    return daily_rides


def get_daily_rides_with_weather(df, api_key):
    """
    Given a DataFrame containing daily bike ride data, this function adds weather information for each day
    by calling the OpenWeatherMap API.

    Args:
    - df: DataFrame containing daily bike ride data
    - api_key: OpenWeatherMap API key

    Returns:
    - DataFrame with weather information added
    """

    # Define a function to retrieve weather data for a given date
    def get_weather_data(date):
        url = f"https://api.openweathermap.org/data/2.5/weather?q=New%20York&units=imperial&appid={api_key}&dt={date}"
        response = requests.get(url)

        # Check if the API request was successful
        if response.status_code == 200:
            data = response.json()
            weather_data = {
                "date": date,
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
            }
            print(f"Weather for {date} retrieved", end="\r")

            # Wait for one second to limit API requests to 60 per minute
            time.sleep(1)

            return weather_data
        else:
            # If there was an error, print the status code and return None
            print(f"Error: {response.status_code} - {date}", end="\n")
            return None

    # Apply the get_weather_data function to each unique date in the DataFrame and create a new column with the results
    df["weather"] = df["date"].apply(get_weather_data)

    # Extract weather information from the 'weather' column and add it as separate columns in the DataFrame
    df["temperature"] = df["weather"].apply(
        lambda x: x["temperature"] if x is not None else None
    )
    df["description"] = df["weather"].apply(
        lambda x: x["description"] if x is not None else None
    )
    df["humidity"] = df["weather"].apply(
        lambda x: x["humidity"] if x is not None else None
    )
    df["wind_speed"] = df["weather"].apply(
        lambda x: x["wind_speed"] if x is not None else None
    )

    # Drop the 'weather' column
    df.drop("weather", axis=1, inplace=True)

    return df


def forecast_number_users(
    df, X=None, y=None, test_size=0.2, lstm_units=64, lstm_epochs=50
):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[X], df[y], test_size=test_size, shuffle=False
    )

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape the input data to be 3-dimensional for the LSTM model
    X_train_reshaped = X_train_scaled.reshape(
        X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
    )
    X_test_reshaped = X_test_scaled.reshape(
        X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
    )

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(1, X_train_scaled.shape[1])))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")

    # Fit the model on the training set
    model.fit(X_train_reshaped, y_train, epochs=lstm_epochs, batch_size=1, verbose=0)

    # Make predictions on the testing set
    predictions = model.predict(X_test_reshaped)

    # Calculate the mean absolute error
    mae = np.mean(np.abs(predictions - y_test))

    return model, mae
