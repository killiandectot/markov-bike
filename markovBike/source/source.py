import pandas_gbq

def database_query(n_samples):


    sql_stations_query = f"""
                        select *
                        from `bigquery-public-data.new_york.citibike_stations`
                        LIMIT {n_samples}
                        """

    sql_trips_query = f"""
                    SELECT *
                    FROM `bigquery-public-data.new_york.citibike_trips`
                    LIMIT {n_samples}
                    """

    return sql_stations_query, sql_trips_query

def get_stations_data(n_samples):

    sql_stations = database_query(n_samples)[0]

    if sql_stations:

        stations = pandas_gbq.read_gbq(sql_stations)

        print(f'Bike stations table with {n_samples} samples and {len(stations.columns)} columns loaded, columns are: \n\n{stations.dtypes}')

    else:
        print('No query found')

    return stations

def get_trips_data(n_samples):

    sql_trips = database_query(n_samples)[1]

    if sql_trips:

        trips = pandas_gbq.read_gbq(sql_trips)

        print(
            f'Bike trips table with {n_samples} samples and {len(trips.columns)} columns loaded, columns are: \n\n{trips.dtypes}'
        )

    else:
        print('No query found')

    return trips

def preprocess_stations_data():
    pass

def preprocess_trips_data():
    pass

def merge_data():
    pass

def source_samples():
    pass
