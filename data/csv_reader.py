# data/csv_reader.py

import os
import pandas as pd
from typing import List

class CSVReader:
    def __init__(self, data_dir: str, csv_files: List[str]):
        self.data_dir = data_dir
        self.csv_files = csv_files

    def read_csv_files(self) -> pd.DataFrame:
        """Reads multiple CSV files and combines them into a single DataFrame."""
        dataframes = []
        for file in self.csv_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path)
                dataframes.append(df)
                print(f"Successfully read {file_path}")
            except FileNotFoundError:
                print(f"File {file_path} not found.")
            except pd.errors.EmptyDataError:
                print(f"File {file_path} is empty.")
            except Exception as e:
                print(f"An error occurred while reading {file_path}: {e}")
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print("Successfully combined all CSV files.")
            return combined_df
        else:
            print("No dataframes to combine.")
            return pd.DataFrame()
