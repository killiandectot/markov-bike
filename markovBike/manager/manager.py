import folium


class Manager():

    def __init__(self) -> None:
        pass

    @staticmethod
    def print_table(rows):
        # Get the maximum length of each column
        max_lengths = [
            max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))
        ]

        # Create the header and row separator strings
        header = "-+-".join("-" * (length + 2) for length in max_lengths)
        row_separator = "+".join("-" * (length + 1) for length in max_lengths)

        # Print the header
        print(row_separator)
        for row in rows:
            print(" |".join(
                str(col).ljust(length)
                for col, length in zip(row, max_lengths)))
            print(row_separator)

    @staticmethod
    def plot_latitude_longitude_map(latitudes, longitudes):
        # Create a folium map centered on New York City
        nyc_latitude = 40.730610
        nyc_longitude = -73.935242
        m = folium.Map(location=[nyc_latitude, nyc_longitude], zoom_start=11)

        # Add markers for each latitude and longitude
        for lat, lng in zip(latitudes, longitudes):
            folium.Marker(location=[lat, lng], ).add_to(m)

        # Display the map
        return m

    @staticmethod
    def split_dataframe(df, drops=[], verbose= True):

        df.set_index([col for col in df.columns if 'id' in col], inplace=True)

        df.drop(drops, axis=1, inplace=True)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        numerical_features = df.select_dtypes(include=numerics).columns

        categorical_features = df.select_dtypes(include='object').columns

        boolean_features = df.select_dtypes(include='bool').columns

        if verbose:

            print(numerical_features)

            print(categorical_features)

            print(boolean_features)

        return numerical_features, categorical_features, boolean_features
