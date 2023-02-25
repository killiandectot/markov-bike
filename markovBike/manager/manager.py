import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import seaborn as sns
import folium


class Manager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def print_terminal_table(rows):
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
            print(
                " |".join(
                    str(col).ljust(length) for col, length in zip(row, max_lengths)
                )
            )
            print(row_separator)

    @staticmethod
    def plot_nodes(latitudes, longitudes):
        # Create a new plot
        fig, ax = plt.subplots(
            figsize=(16, 16),
        )

        # Remove left and top spines
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Create reference lines from the ticks
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")

        ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5)

        # Set number of ticks for x and y axes
        ax.xaxis.set_major_locator(ticker.MaxNLocator(40))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(40))

        # Plot the nodes as red dots
        ax.scatter(longitudes, latitudes, s=50, c="red", alpha=0.8, edgecolors="none")

        # Set the axis limits and add a title
        ax.set_xlim(min(longitudes) - 0.01, max(longitudes) + 0.025)
        # print(min(longitudes) - 0.025)
        # print(max(longitudes) + 0.025)

        ax.set_ylim(min(latitudes) - 0.01, max(latitudes) + 0.025)
        # print(min(latitudes) - 0.025)
        # print(max(latitudes) + 0.025)
        ax.set_title("Bike Sharing System")

        # Set the font size for the tick labels
        ax.tick_params(axis="both", labelsize=4)

        # Show the plot
        plt.show()

    @staticmethod
    def plot_subgraphs(dataframe, stations, number_of_plots):
        # Select a random subset of starting stations
        stations = np.random.choice(stations, number_of_plots, replace=False)

        # Calculate the number of subplots needed
        num_subplots = len(stations)

        # Determine the number of rows and columns of subplots
        num_cols = min(max(num_subplots, 3), 5)
        num_rows = (num_subplots - 1) // num_cols + 1

        # Loop over each starting station and plot the corresponding graph
        for i, start_station in enumerate(stations):
            start_latitudes = list(
                dataframe[dataframe["start_station_id"] == start_station][
                    "start_station_latitude"
                ]
            )
            start_longitudes = list(
                dataframe[dataframe["start_station_id"] == start_station][
                    "start_station_longitude"
                ]
            )
            end_latitudes = list(
                dataframe[dataframe["start_station_id"] == start_station][
                    "end_station_latitude"
                ]
            )
            end_longitudes = list(
                dataframe[dataframe["start_station_id"] == start_station][
                    "end_station_longitude"
                ]
            )
            trip_counts = list(dataframe["trip_count"])

            line_thickness = [0.02 * count for count in trip_counts]

            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 20))

            # Remove left and top spines
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Create reference lines from the ticks
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

            ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.5)
            ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5)

            # Set number of ticks for x and y axes
            ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(20))

            for j in range(len(start_latitudes)):
                ax.plot(
                    [start_longitudes[j], end_longitudes[j]],
                    [start_latitudes[j], end_latitudes[j]],
                    "-",
                    color="black",
                    alpha=0.1,
                    linewidth=line_thickness[j],
                )

            ax.scatter(
                start_longitudes,
                start_latitudes,
                s=200,
                c="red",
                alpha=0.1,
                edgecolors="none",
            )

            ax.scatter(
                end_longitudes,
                end_latitudes,
                s=100,
                c="blue",
                alpha=0.1,
                edgecolors="none",
            )

            # ax.set_xlim(
            #    min(start_longitudes + end_longitudes) - 0.01,
            #    max(start_longitudes + end_longitudes) + 0.01)

            ax.set_xlim(-74.11170067787171, -73.85645)

            # ax.set_ylim(
            #    min(start_latitudes + end_latitudes) - 0.01,
            #    max(start_latitudes + end_latitudes) + 0.01)

            ax.set_ylim(40.608385, 40.90726)

            ax.set_title(f"Trips starting at station {start_station}")

            # Save the plot
            fig.savefig(f"plots/trips_starting_at_station_{start_station}.png")

            # Close the plot
            plt.close(fig)

        # Create subplots of all the saved plots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))

        # Flatten the axes array for ease of use
        axes = axes.flatten()

        # Loop over each saved plot and add it to the subplots
        for i, start_station in enumerate(stations):
            img = mpimg.imread(f"plots/trips_starting_at_station_{start_station}.png")
            axes[i].imshow(img)
            axes[i].axis("off")

        # Show the subplots
        plt.show()

        # Loop over each starting station and plot the corresponding graph
        for i, start_station in enumerate(stations):
            start_latitudes = list(
                dataframe[dataframe["end_station_id"] == start_station][
                    "end_station_latitude"
                ]
            )
            start_longitudes = list(
                dataframe[dataframe["end_station_id"] == start_station][
                    "end_station_longitude"
                ]
            )
            end_latitudes = list(
                dataframe[dataframe["end_station_id"] == start_station][
                    "start_station_latitude"
                ]
            )
            end_longitudes = list(
                dataframe[dataframe["end_station_id"] == start_station][
                    "start_station_longitude"
                ]
            )
            trip_counts = list(dataframe["trip_count"])

            line_thickness = [0.02 * count for count in trip_counts]

            # Create the plot
            fig, ax = plt.subplots(figsize=(20, 20))

            # Remove left and top spines
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Create reference lines from the ticks
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

            ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.5)
            ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5)

            # Set number of ticks for x and y axes
            ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(20))

            for j in range(len(start_latitudes)):
                ax.plot(
                    [start_longitudes[j], end_longitudes[j]],
                    [start_latitudes[j], end_latitudes[j]],
                    "-",
                    color="black",
                    alpha=0.1,
                    linewidth=line_thickness[j],
                )

            ax.scatter(
                start_longitudes,
                start_latitudes,
                s=200,
                c="red",
                alpha=0.1,
                edgecolors="none",
            )

            ax.scatter(
                end_longitudes,
                end_latitudes,
                s=100,
                c="blue",
                alpha=0.1,
                edgecolors="none",
            )

            # ax.set_xlim(
            #    min(start_longitudes + end_longitudes) - 0.01,
            #    max(start_longitudes + end_longitudes) + 0.01)

            ax.set_xlim(-74.11170067787171, -73.85645)

            # ax.set_ylim(
            #    min(start_latitudes + end_latitudes) - 0.01,
            #    max(start_latitudes + end_latitudes) + 0.01)

            ax.set_ylim(40.608385, 40.90726)

            ax.set_title(f"Trips ending at station {start_station}")

            # Save the plot
            fig.savefig(f"plots/trips_ending_at_station_{start_station}.png")

            # Close the plot
            plt.close(fig)

        # Create subplots of all the saved plots
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))

        # Flatten the axes array for ease of use
        axes = axes.flatten()

        # Loop over each saved plot and add it to the subplots
        for i, start_station in enumerate(stations):
            img = mpimg.imread(f"plots/trips_ending_at_station_{start_station}.png")
            axes[i].imshow(img)
            axes[i].axis("off")

        # Show the subplots
        plt.show()

    @staticmethod
    def transition_matrix_heatmaps(matrix):
        # Plot a heatmap of the probability matrix
        fig, ax = plt.subplots(figsize=(20, 20))

        num_ticks = 10
        # the index of the position of yticks
        xticks = np.linspace(0, matrix.shape[0] - 1, num_ticks, dtype=np.int)
        yticks = np.linspace(0, matrix.shape[1] - 1, num_ticks, dtype=np.int)

        # the content of labels of these yticks
        xticklabels = [str(idx) for idx in xticks]
        yticklabels = [str(idx) for idx in yticks]

        sns.heatmap(matrix,
                    cmap="rocket",
                    ax=ax,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                    alpha=0.95)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_title("Station Transition Probability Matrix")
        ax.set_xlabel("End Station ID")
        ax.set_ylabel("Start Station ID")

        plt.show()

    @staticmethod
    def folium_latitude_longitude_map(latitudes, longitudes):
        # Create a folium map centered on New York City
        nyc_latitude = 40.730610
        nyc_longitude = -73.935242
        m = folium.Map(location=[nyc_latitude, nyc_longitude], zoom_start=11)

        # Add markers for each latitude and longitude
        for lat, lng in zip(latitudes, longitudes):
            folium.Marker(
                location=[lat, lng],
            ).add_to(m)

        # Display the map
        return m
