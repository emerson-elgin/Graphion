import pandas as pd
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here

class DataIngestion:
    """

    examples and examples. Prepares for future extension of functionality
    """
    def __init__(self, source):
        self.source = source

    def load_data(self, file_format="csv"):
        if file_format == "csv":
            return pd.read_csv(self.source)
        elif file_format == "json":
            return pd.read_json(self.source)
        elif file_format == "sql":
            import sqlite3
            conn = sqlite3.connect(self.source)
            query = "SELECT * FROM table_name"  # Modify as needed
            return pd.read_sql_query(query, conn)
        else:
            raise ValueError("Unsupported file format.")
