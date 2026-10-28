import numpy as np

from prova import load_It_data


def test_load_It_data():
    """Testing the load_It_data function  with a file path.
       This test only asserts the loading of the correct columns and dataframe building.
       Column and conversion to seconds chose based on the settings of the instrumentation.
    """
    file_path = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\03_Xray_150kV_500muA_Vds-10_Vgs-7.dat"
    time, Ids = load_It_data(file_path)
    # Checking the type of the array extracted using Pandas
    assert isinstance(time, list)
    assert isinstance(Ids, list)
    # Check the data types (int, float) because the time data are also floats
    assert all(isinstance(i, (int, float)) for i in time)
    assert all(isinstance(i, ( float)) for i in Ids)  
    #Checking range of values of the arrays
    assert all(0 < t < 1000 for t in time)
    assert all(-1e-11 < i < -1e-6 for i in Ids)