#test loader

from factor_loader import FactorDataLoader

# Use a raw string to handle backslashes and enclose the path in triple quotes to handle spaces
data_file_path = r"""C:\Users\frost\sde_project_v2\raw data\factors dataset final.csv"""

# Initialize the loader with the correct parameter name and file path
loader = FactorDataLoader(data_path=data_file_path)

# Test loading the data
try:
    data = loader.load_data()
    print(data.head())  # Print the first few rows to confirm it loaded successfully
except Exception as e:
    print(f"Error loading data: {str(e)}")


