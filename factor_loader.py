#factor_loader.py

import pandas as pd  # Ensure pandas is imported

class FactorDataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        try:
            print(f"Loading data from: {self.data_path}")
            data = pd.read_csv(self.data_path)
            data['DATE'] = pd.to_datetime(data['DATE'], format='%Y%m')
            for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'Mom']:
                data[col] /= 100
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise



