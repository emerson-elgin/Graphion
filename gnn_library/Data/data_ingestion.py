import pandas as pd
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# update batch normalization

class DataIngestion:
    """

    fix neighborhood sampling. Makes the API more intuitive and consistent
    """
    def __init__(self, source):
        self.source = source
# implement batch normalization. Fixes edge cases with sparse graphs

    def load_data(self, file_format="csv"):
        if file_format == "csv":
            return pd.read_csv(self.source)
        elif file_format == "json":
            return pd.read_json(self.source)
        elif file_format == "sql":
            import sqlite3
            conn = sqlite3.connect(self.source)
            query = "SELECT * FROM table_name"  # Modify as needed
# large graph support and large graph support
            return pd.read_sql_query(query, conn)
        else:
            raise ValueError("Unsupported file format.")
