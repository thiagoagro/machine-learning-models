import pandas as pd
import numpy as np

def load_data(file_obj) -> pd.DataFrame:
    """
    Load data from an Excel or CSV file object.
    """
    if file_obj.name.endswith('.csv'):
        return pd.read_csv(file_obj)
    elif file_obj.name.endswith('.xlsx') or file_obj.name.endswith('.xls'):
        return pd.read_excel(file_obj)
    else:
        raise ValueError("File not supported. Use CSV or Excel.")

def generate_eda(df: pd.DataFrame) -> dict:
    """
    Generate basic Exploratory Data Analysis statistics.
    Returns a dictionary with numeric description, missing values, etc.
    """
    missing_data = df.isnull().sum()
    
    eda_results = {
        'shape': df.shape,
        'head': df.head(),
        'describe': df.describe(),
        'missing_values': missing_data[missing_data > 0],
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(exclude=[np.number]).columns.tolist()
    }
    return eda_results
