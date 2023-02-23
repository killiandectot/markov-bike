import numpy as np
import pandas as pd


def get_station_to_station_matrix(trips_raw, station_dict):
    """
    Computes the transition matrix of trips between bike-sharing stations.

    Args:
        trips_data (pandas.DataFrame): A DataFrame containing bike trips data with columns
            "start_station_id" and "end_station_id".

    Returns:
        numpy.ndarray: A square matrix where each row and column represents a unique station ID.
            The (i, j) element of the matrix contains the probability of transitioning from station i to j.
    """

    # Create dictionary to map station ids to indices
    station_dict = {
        station_id: i
        for i, station_id in enumerate(
            np.unique(trips_raw[["start_station_id", "end_station_id"]].values)
        )
    }

    # Get number of stations
    num_stations = len(station_dict)

    # Initialize matrix with zeros
    station_to_station_matrix = np.zeros((num_stations, num_stations))

    # Loop over rows in dataframe and count trips between stations
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row["start_station_id"]]
        end_station = station_dict[row["end_station_id"]]
        if start_station != end_station:
            station_to_station_matrix[start_station][end_station] += 1

    # Calculate probabilities
    for i in range(num_stations):
        total_trips = sum(station_to_station_matrix[i])
        if total_trips > 0:
            station_to_station_matrix[i] /= total_trips

    return station_to_station_matrix


def get_station_to_trip_matrix(trips_raw, station_dict):
    """
    Generates a matrix representing the proportion of trips starting at each station.

    Args:
        trips_raw: Pandas DataFrame containing information about each bike trip.
        station_dict: Dictionary mapping station IDs to matrix indices.

    Returns:
        A square numpy ndarray representing the proportion of trips starting at each station.
    """

    # Initialize matrix with zeros
    num_stations = len(station_dict)
    bsi_tbj = np.zeros((num_stations, num_stations))

    # Count number of trips starting at each station
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row["start_station_id"]]
        bsi_tbj[start_station][start_station] += 1

    # Calculate probability values
    row_sums = np.sum(bsi_tbj, axis=1, keepdims=True)
    nonzero_row_indices = np.nonzero(row_sums)[0]
    bsi_tbj[nonzero_row_indices, :] /= row_sums[nonzero_row_indices, :]

    return bsi_tbj


def get_trip_to_station_matrix(trips_raw, station_dict):
    """
    Computes the transition matrix between bike stations based on the number of trips between each station.

    Args:
    - trips_raw: a pandas DataFrame containing information about bike trips.

    Returns:
    - tbs_matrix: a numpy array representing the transition matrix between bike stations based on the number of trips
                  between each station.
    """
    # Create dictionary to map station ids to indices
    station_dict = {
        id: i
        for i, id in enumerate(
            np.unique(trips_raw[["start_station_id", "end_station_id"]].values)
        )
    }

    # Initialize matrix with zeros
    num_stations = len(station_dict)
    tbs_matrix = np.zeros((num_stations, num_stations))

    # Count number of trips ending at each station
    for _, row in trips_raw.iterrows():
        start_station = station_dict[row["start_station_id"]]
        end_station = station_dict[row["end_station_id"]]
        tbs_matrix[end_station][start_station] += 1

    # Calculate probability values, handling divide-by-zero error
    row_sums = np.sum(tbs_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # replace 0s with 1s to avoid division by zero
    tbs_matrix /= row_sums

    return tbs_matrix


def full_transition_matrix(trips_raw):
    """
    This function takes in a pandas dataframe of trip data and calculates a full
    transition matrix between all stations. The matrix is constructed by using
    three submatrices: a matrix of trips from station i to station j, a matrix of
    trips from station i to any other station, and a matrix of trips from any
    station to station j. These submatrices are combined to form the full
    transition matrix. The matrix is normalized such that the rows sum to 1.

    Args:
    - trips_raw (pandas dataframe): dataframe of trip data with the following
      columns: ['start_station_id', 'end_station_id']. Each row represents a
      single trip taken by a user.

    Returns:
    - full_transition_matrix (numpy array): a square matrix with dimensions
      (n_stations, n_stations) where n_stations is the number of unique stations
      in the trip data. The (i, j) entry in the matrix represents the probability
      of transitioning from station i to station j.
    """

    # Create dictionary to map station ids to indices
    station_dict = {
        id: i
        for i, id in enumerate(
            np.unique(trips_raw[["start_station_id", "end_station_id"]].values)
        )
    }

    print(f"The length of the dictionary is {len(station_dict)}")

    # Get number of stations
    n_stations = len(station_dict)

    # Get submatrices
    bsibsj = get_station_to_station_matrix(trips_raw, station_dict)
    print(bsibsj.shape)
    bsitbj = get_station_to_trip_matrix(trips_raw, station_dict)
    print(bsitbj.shape)
    tbitbj = get_trip_to_station_matrix(trips_raw, station_dict)
    print(tbitbj.shape)

    # Initialize full transition matrix with zeros
    full_transition_matrix = np.zeros((n_stations, n_stations))

    # Calculate full transition matrix
    for i in range(n_stations):
        for j in range(n_stations):
            if i == j:
                # If i and j are the same, the probability of transitioning from
                # station i to station j is 0
                full_transition_matrix[i][j] = 0
            else:
                # Otherwise, use the submatrices to calculate the probability of
                # transitioning from station i to station j
                bsit = bsitbj[i][j]
                tbit = tbitbj[i][j]
                bsib = bsibsj[i][j]
                total_trips_from_i = bsit + bsib
                if total_trips_from_i == 0:
                    # If there are no trips from station i to any other station,
                    # the probability of transitioning from station i to station j
                    # is 0
                    full_transition_matrix[i][j] = 0
                else:
                    # Otherwise, calculate the probability of transitioning from
                    # station i to station j using the submatrices and normalize by
                    # the total number of trips from station i
                    prob_it = bsit / total_trips_from_i
                    prob_ti = tbit / total_trips_from_i
                    prob_ib = bsib / total_trips_from_i
                    full_transition_matrix[i][j] = prob_it + prob_ti + prob_ib

    return bsibsj, bsitbj, tbitbj, full_transition_matrix
