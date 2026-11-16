import pandas as pd
import numpy as np

# Assuming the export functions are imported or defined in the scope
# from your_module import export_it_series_to_csv, export_results_to_csv

from export import export_it_series_to_csv, export_results_to_csv

example_path = r"Test Data\23092026_New_Columbia_PP_Au\Au_elect\12\03_Xray_150kV_500muA_Vds-10_Vgs-2.dat"

def test_export_it_series_should_pad_shorter_arrays_with_nan():
    """
    Verify that shorter arrays are correctly 
    padded with NaNs to match max length.
    """
    # example dictionary
    data_dict = {
        "Time": [0.0, 1.0, 2.0, 3.0],
        "Short_Signal": [10.0, 20.0]  # Length 2 vs Max length 4
    }
    
    file_path = example_path.replace(".dat", "_padded_test.csv")
    #fails if data are not in .dat format
    export_it_series_to_csv(data_dict, str(file_path), sep=';')
    
    # Read back the saved CSV to verify content
    df_result = pd.read_csv(file_path, sep=';')
    
    assert len(df_result) == 4
    assert df_result["Time"].tolist() == [0.0, 1.0, 2.0, 3.0]
    
    expected_series = pd.Series([10.0, 20.0, np.nan, np.nan], name="Short_Signal")
    pd.testing.assert_series_equal(df_result["Short_Signal"], expected_series)    
    # Assert series equality (handles NaNs automatically)



def test_export_it_series_should_handle_empty_dictionary_gracefully( capsys):
    """
    Verify that exporting an empty dictionary 
    prints a warning and skips file creation.
    """
    empty_dict = {}
    
    export_it_series_to_csv(empty_dict, str(example_path), sep=';')
    
    # Check captured stdout for the warning message
    captured = capsys.readouterr()
    assert "Warning: Dictionary" in captured.out


def test_export_results_should_save_equal_length_data_without_padding():
    """Verify that results dictionaries with equal lengths are exported cleanly without padding."""
    data_dict = {
        "Sample": ["P4", "P6"],
        "Sensitivity": [1.2e5, 1.5e5]
    }
    
    file_path = example_path.replace(".dat", "_results_test.csv")
    #Fails if .dat is missing
    
    export_results_to_csv(data_dict, str(file_path), sep='\t')
    
    df_result = pd.read_csv(file_path, sep='\t')
    
    assert len(df_result) == 2
    assert "NaN" not in df_result.values.astype(str)
    assert df_result["Sample"].tolist() == ["P4", "P6"]
    assert df_result["Sensitivity"].tolist() == [1.2e5, 1.5e5]