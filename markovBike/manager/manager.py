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
