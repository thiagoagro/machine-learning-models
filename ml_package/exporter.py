import pandas as pd
import io

def export_to_excel(results_df, predictions_df, model_params_df=None):
    """
    Takes multiple dataframes and writes them to an Excel file buffer.
    Returns the bytes of the file for Streamlit download.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        if results_df is not None:
            results_df.to_excel(writer, sheet_name='Metrics', index=False)
        if predictions_df is not None:
            predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        if model_params_df is not None:
            model_params_df.to_excel(writer, sheet_name='Model_Parameters', index=False)
    return buffer.getvalue()
