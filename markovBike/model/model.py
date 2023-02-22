import numpy as np
import pandas as pd

def calculate_probability_matrix(dataframe):
    # Get a list of all unique station IDs
    station_ids = dataframe[['start_station_id',
                             'end_station_id']].stack().unique()

    # Create an empty matrix to store trip counts
    trip_counts = np.zeros((len(station_ids), len(station_ids)))

    # Loop over each row in the DataFrame and update the corresponding entry in the trip counts matrix
    for _, row in dataframe.iterrows():
        start_station_id = row['start_station_id']
        end_station_id = row['end_station_id']
        start_index = np.where(station_ids == start_station_id)[0][0]
        end_index = np.where(station_ids == end_station_id)[0][0]
        trip_counts[start_index, end_index] += 1

    # Calculate the total number of trips from each station
    total_trips = trip_counts.sum(axis=1)

    # Calculate the probability matrix
    probability_matrix = trip_counts / total_trips[:, np.newaxis]

    # Fill NaN values with zeros
    probability_matrix = np.nan_to_num(probability_matrix)

    # Convert to a pandas DataFrame and add row and column labels
    probability_matrix = pd.DataFrame(probability_matrix,
                                      index=station_ids,
                                      columns=station_ids)

    print('returned')

    return probability_matrix

def BSiBSj(trips_raw):
    # Create dictionary to map station ids to indices
    station_dict = {
        id: i
        for i, id in enumerate(
            set(trips_raw['start_station_id']).union(
                set(trips_raw['end_station_id'])))
    }

    # Get number of stations
    N = len(station_dict)

    # Initialize matrix with zeros
    bsibsj = np.zeros((N, N))

    # Loop over rows in dataframe
    for index, row in trips_raw.iterrows():
        start_station = station_dict[row['start_station_id']]
        end_station = station_dict[row['end_station_id']]

        print(
            f'Trip number {index} from station {start_station} to station {end_station}',
            end='\r')

        if start_station == end_station:
            bsibsj[start_station][end_station] += 1
        else:
            pass

    # Calculate probabilities
    for i in range(N):
        total = sum(bsibsj[i])
        if total > 0:
            bsibsj[i] /= total

    return bsibsj

def BSiBSj2(trips_raw):
    # create dictionary mapping station ids to indices
    station_dict = {
        id: i
        for i, id in enumerate(
            np.unique(
                np.concatenate([
                    trips_raw['start_station_id'].values,
                    trips_raw['end_station_id'].values
                ])))
    }

    # initialize matrix with zeros
    num_stations = len(station_dict)
    bsibsj = np.zeros((num_stations, num_stations))

    # populate matrix with probabilities
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row['start_station_id']]
        end_station = station_dict[row['end_station_id']]
        if start_station == end_station:
            bsibsj[start_station][end_station] += 1

    # normalize matrix along the diagonal
    for i in range(num_stations):
        bsibsj[i][i] /= np.sum(bsibsj[i])

    return bsibsj


def BSiTBj(trips_raw):
    # Create dictionary to map station id to matrix index
    station_dict = {
        id: i
        for i, id in enumerate(np.unique(trips_raw['start_station_id']))
    }

    # Initialize matrix with zeros
    n = len(station_dict)
    bsitbj = np.zeros((n, n))

    # Count number of trips starting at each station
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row['start_station_id']]
        bsitbj[start_station][start_station] += 1

    # Calculate probability values
    bsitbj = bsitbj / np.sum(bsitbj, axis=1, keepdims=True)

    return bsitbj


def TBiTBj(trips_raw):
    station_dict = {
        id: i
        for i, id in enumerate(trips_raw['start_station_id'].unique())
    }
    num_stations = len(station_dict)
    tbitbj = np.zeros((num_stations, num_stations))
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row['start_station_id']]
        end_station = station_dict[row['end_station_id']]
        tbitbj[start_station][end_station] += 1
    tb = np.sum(tbitbj, axis=1)
    tbitbj /= tb[:, np.newaxis]
    np.fill_diagonal(tbitbj, 0)
    return tbitbj
