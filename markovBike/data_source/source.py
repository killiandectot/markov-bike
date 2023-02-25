from markovBike.manager.manager import Manager
import argparse
from google.cloud import bigquery


def database_queries(n_samples):
    """
    Generate SQL queries to fetch station and trip data from BigQuery.
    :param n_samples: The number of samples to retrieve for each table.
    :return: A dictionary containing the SQL queries for stations and trips tables.
    """
    sql_stations_query = f"SELECT * FROM bigquery-public-data.new_york.citibike_stations LIMIT {n_samples}"
    sql_trips_query = (
        f"SELECT * FROM bigquery-public-data.new_york.citibike_trips LIMIT {n_samples}"
    )
    return {"stations": sql_stations_query, "trips": sql_trips_query}


def get_stations_data(query, verbose=True):
    """
    Retrieve station data from BigQuery and convert it into a Pandas DataFrame.
    :param query: The SQL query to retrieve station data.
    :param verbose: If True, print information about the retrieved data.
    :return: A Pandas DataFrame containing station data.
    """
    if query:
        client = bigquery.Client()
        dataframe_stations = client.query(query).to_dataframe()
        if verbose:
            print(
                f"Bike station table with shape {dataframe_stations.shape}. Columns are: \n\n{dataframe_stations.dtypes}\n"
            )
        return dataframe_stations
    else:
        print("No query")
        return None


def get_trips_data(query, verbose=True):
    """
    Retrieve trip data from BigQuery and convert it into a Pandas DataFrame.
    :param query: The SQL query to retrieve trip data.
    :param verbose: If True, print information about the retrieved data.
    :return: A Pandas DataFrame containing trip data.
    """
    if query:
        client = bigquery.Client()
        dataframe_trips = client.query(query).to_dataframe()
        if verbose:
            print(
                f"Bike trips table with shape {dataframe_trips.shape}. Columns are: \n\n{dataframe_trips.dtypes}\n"
            )
        return dataframe_trips
    else:
        print("No query")
        return None


def merge_data():
    pass


def source_samples():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a BigQuery query")

    parser.add_argument("--query", type=str, help="The query to run", required=True)

    args = parser.parse_args()

    dataframe_stations = get_stations_data(args.query, verbose=False)

    Manager.print_table(dataframe_stations.values.tolist())
