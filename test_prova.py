import numpy as np
import pandas as pd
from prova import load_It_data


def test_type_load_It_data():
    """Testing the load_It_data function  with a file path.
       This test only asserts the loading of the correct columns and dataframe building.
       Column and conversion to seconds chose based on the settings of the instrumentation.
    """
    file_path = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\03_Xray_150kV_500muA_Vds-10_Vgs-7.dat"
    time, Ids = load_It_data(file_path)
    # Checking the type of the array extracted using Pandas
    assert isinstance(time, pd.Series) and isinstance(Ids, pd.Series)
    # Check the data types (int, float) because the time data are also floats
    assert all(isinstance(i, (int, float)) for i in time)
    assert all(isinstance(i, (int, float)) for i in Ids)  


def test_range_load_It_data():
    """
    Testing data loaded are within expected ranges.
    """
    file_path = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\03_Xray_150kV_500muA_Vds-10_Vgs-7.dat"
    time, Ids = load_It_data(file_path)
    abs_ids= np.abs(Ids)
    #Checking range of values of the arrays
    #print(time, abs_ids)
    #assert print(all(0 <= t < 2500 for t in time))
    assert all(0 <= t < 1500 for t in time)
    assert all(1e-12 < i < 1e-5 for i in abs_ids)