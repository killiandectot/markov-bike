from markovBike.data_source.source import database_queries, get_stations_data, get_trips_data
from markovBike.data_source.preprocess import preprocess_stations_data, preprocess_trips_data

from markovBike.forecast.forecast import forecast_number_users

queries = database_queries(12)

stations_raw_dataframe = get_stations_data(queries['stations'])

trips_raw_dataframe = get_trips_data(queries['trips'])

stations_preproc_dataframe = preprocess_stations_data(stations_raw_dataframe) # TO DO

trips_preproc_dataframe = preprocess_trips_data(trips_raw_dataframe) # TO DO

number_of_users = forecast_number_users()

# TO THE MODELING
