import numpy as np
import pandas as pd
import pytest

# Importing functions from the main code to be tested
from prova import load_It_data
from prova import normalize_It_manual
from prova import photocurrent

# ==========================================
# TEST PER `load_It_data`
# ==========================================

file_path = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\03_Xray_150kV_500muA_Vds-10_Vgs-7.dat"
time, ids = load_It_data(file_path)
abs_ids= np.abs(ids)

def test_type_load_It_data():
    """Testing the load_It_data function  with a file path.
       This test only asserts the loading of the correct columns and dataframe building.
       Column and conversion to seconds chose based on the settings of the instrumentation.
    """

    # Checking the type of the array extracted using Pandas
    assert isinstance(time, pd.Series) and isinstance(ids, pd.Series)
    # Check the data types (int, float) because the time data are also floats
    assert all(isinstance(i, (int, float)) for i in time)
    assert all(isinstance(i, (int, float)) for i in ids)  


def test_range_load_It_data():
    """
    Testing the load_It_data function  with a file path.
    Testing data loaded are within expected ranges.
    """
    #Checking range of values of the arrays
    #print(time, abs_ids)
    #assert print(all(0 <= t < 2500 for t in time))
    assert all(0 <= t < 1500 for t in time)
    assert all(1e-12 < i < 1e-5 for i in abs_ids)



# ==========================================
# TEST PER `normalize_It_manual`
# ==========================================

min_points = [240, 360, 480]  # Punti nominali per il test
it_normalized, par_fit, mae = normalize_It_manual(time, abs_ids, min_points)

def test_normalize_It_manual_output_shapes():
    """Verify that output shapes and types are correct."""

    # Mock mean_absolute_error to avoid heavy external dependencies if unnecessary, 
    # or use the real sklearn if installed. Here we use real sklearn.
    from sklearn.metrics import mean_absolute_error

    #Checking types 
    assert isinstance(it_normalized, np.ndarray)
    assert isinstance(par_fit, list)
    assert isinstance(mae, float)

   #Checking length of the output arrays
    assert len(it_normalized) == len(ids)
    assert mae >= 0.0


# ==========================================
# TEST PER `photocurrent`
# ==========================================

# General data photocurrent
prec=1e-9
phcur, err_phcur = photocurrent(ids, prec, mae)

def test_photocurrent_success():
    """
    Verifies that the photocurrent function returns valid outputs for a typical input.
    """
    assert isinstance(phcur, float)
    assert isinstance(err_phcur, float)
    assert err_phcur > 0.0


def test_photocurrent_no_peaks_raises_error():
    """Verifica che venga sollevato un ValueError se il segnale è piatto (nessun picco)."""
    Ids = np.ones(100)  # Segnale costante, nessun picco/minimo
    prec = 0.1
    mae = 0.0
    
    with pytest.raises(ValueError, match="Nessun picco trovato"):
        photocurrent(Ids, prec, mae)


def test_photocurrent_too_few_cycles():
    """Verifica che venga sollevato un errore se i cicli completi sono inferiori a 2."""
    # Segnale con un solo picco accennato
    Ids = np.array([0, 1, 0, -1, 0, 0, 0])
    prec = 0.05
    mae = 0.0
    
    with pytest.raises(ValueError, match="Troppi pochi cicli completi"):
        photocurrent(Ids, prec, mae)


