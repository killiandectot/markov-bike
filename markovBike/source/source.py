from markovBike.manager.manager import Manager
import argparse
from google.cloud import bigquery

def database_queries(n_samples):

    sql_stations_query = f"SELECT * FROM bigquery-public-data.new_york.citibike_stations LIMIT {n_samples}"

    sql_trips_query = f"SELECT * FROM bigquery-public-data.new_york.citibike_trips LIMIT {n_samples}"

    return dict(stations= sql_stations_query, trips = sql_trips_query)

def get_stations_data(query, verbose=True):

    if query:
        client = bigquery.Client()

        dataframe_stations = client.query(query).result().to_dataframe()

        if verbose:
            print(
                f'Bike trips table with shape {dataframe_stations.shape}. Columns are: \n\n{dataframe_stations.dtypes}'
            )

        return dataframe_stations

    else:
        print('No query')


def get_trips_data(query, verbose=True):

    if query:
        client = bigquery.Client()

        dataframe_trips = client.query(query).result().to_dataframe()

        if verbose:
            print(
                f'Bike trips table with shape {dataframe_trips.shape}. Columns are: \n\n{dataframe_trips.dtypes}'
            )

        return dataframe_trips

    else:
        print('No query')


def merge_data():

    pass


def source_samples():

    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run a BigQuery query')

    parser.add_argument('--query',
                        type=str,
                        help='The query to run',
                        required=True)

    args = parser.parse_args()

    dataframe_stations = get_stations_data(args.query, verbose=False)

    Manager.print_table(dataframe_stations.values.tolist())
