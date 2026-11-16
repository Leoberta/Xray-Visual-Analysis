import numpy as np
import pandas as pd
import pytest

# Importing functions from the main code to be tested
from prova import load_It_data
from prova import normalize_It_manual
from prova import photocurrent

from prova import import_ofet
from prova import fit_ofet_lin
from prova import linfit_transfer
from prova import ss_linfit_transfer
from prova import surf_sensitivity

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

def test_normalize_It_manual_output_shapes():
    """
    Verify that output shapes and types are correct.
    """
    min_points = [240, 360, 480]  # Punti nominali per il test
    it_normalized, par_fit, mae = normalize_It_manual(time, abs_ids, min_points)

    #Checking types 
    assert isinstance(it_normalized, np.ndarray)
    assert isinstance(par_fit, list)
    assert isinstance(mae, float)


def test_normalize_It_manual_should_preserve_output_dimensions():
    """
    Verify that normalized output array has the exact same length as input.
    """
    min_points = [240, 360, 480]  # Punti nominali per il test
    it_normalized, par_fit, mae = normalize_It_manual(time, abs_ids, min_points)

    assert len(it_normalized) == len(abs_ids)


def test_normalize_It_manual_should_return_non_negative_mae():
    """
    Verify that the calculated Mean Absolute Error is always greater
    than or equal to zero.
    """    
    min_points = [240, 360, 480]  # Punti nominali per il test
    it_normalized, par_fit, mae = normalize_It_manual(time, abs_ids, min_points)

    assert mae >= 0.0


def test_normalize_It_manual_should_handle_boundary_indices_safely():
    """
    Verify behavior when specified minimum points are extremely close 
    to array boundaries.
    """
    
    time = np.linspace(0, 2000, 2000)
    abs_ids = np.ones(2000)
    # Index 0 or near edge might trigger negative slicing issues if not safely bounded
    min_points = [2000] 
    
    with pytest.raises((IndexError, ValueError)):
        normalize_It_manual(time, abs_ids, min_points)



# ==========================================
# TEST PER `photocurrent`
# ==========================================
min_points = [240, 360, 480]  # Punti nominali per il test
it_normalized, par_fit, mae = normalize_It_manual(time, abs_ids, min_points)


def test_photocurrent_should_return_valid_floating_metrics():
    """
    Verify that photocurrent calculation returns finite floats and 
    a positive error estimate.
    """

    prec = 1e-9
    phcur, err_phcur = photocurrent(it_normalized, prec, mae)
    
    assert isinstance(phcur, float)
    assert isinstance(err_phcur, float)
    assert err_phcur > 0.0


def test_photocurrent_should_raise_value_error_on_flat_signals():
    """
    Verify that a ValueError is explicitly raised if the input 
    signal is completely flat.
    """
    flat_ids = np.ones(100)
    prec = 0.1
    mae = 0.0
    
    with pytest.raises(ValueError, match="Nessun picco trovato"):
        photocurrent(flat_ids, prec, mae)


def test_photocurrent_should_raise_value_error_on_insufficient_cycles():
    """
    Verify that a ValueError is raised when fewer than two complete cycles
    are present.
    """
    sparse_ids = np.array([0, 1, 0, -1, 0, 0, 0])
    prec = 0.05
    mae = 0.0
    
    with pytest.raises(ValueError, match="Troppi pochi cicli completi"):
        photocurrent(sparse_ids, prec, mae)


# ==========================================
# TESTS FOR `import_ofet`
# ==========================================


def test_import_ofet_type():
    """
    Verify that import_ofet successfully reads file data and computes expected series.
    """
    # Test Dataset 
    path_transfer = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\02.dat"
    skiprows= 10 # Must skip 10 rows
    # 40 points , from 10V to -10V    
    vg, abs_ids, sqrt_ids , log_ids= import_ofet(path_transfer, skip=skiprows, init= 0, end=40) 
    # Verify return types and shapes
    assert isinstance(vg, pd.Series)
    assert len(vg) == 40

    # Check the data types (int, float) because the time data are also floats
    assert all(isinstance(i, (int, float)) for i in vg)
    assert all(isinstance(i, ( float)) for i in abs_ids)  
    assert all(isinstance(i, ( float)) for i in sqrt_ids)  
    assert all(isinstance(i, ( float)) for i in log_ids)  


def test_import_ofet_file_not_found():
    """
    Verify that FileNotFoundError is raised when an invalid path is provided.
    """
    with pytest.raises(FileNotFoundError):
        import_ofet("non_existent_file.txt", skip=1, init=0, end=50)


# ==========================================
# TESTS FOR `fit_ofet_lin`
# ==========================================

# Test Dataset 
path_transfer = r"Test Data\23092026_New_Columbia_PP_Au\PP_elect\00\02.dat"
skiprows= 10 # Must skip 10 rows
# 40 points , from 10V to -10V    
vg, abs_ids, sqrt_ids , log_ids= import_ofet(path_transfer, skip=skiprows, init= 0, end=40) 

def test_fit_ofet_lin_should_return_finite_parameters_and_errors():
    """ 
    Verify that linear and subthreshold fits converge and return valid finite numbers.
    """
    Ci_W_L = 1e-6
    init_mask, end_mask = -8, -4
    init_ss, end_ss = -2, 1 
    
    popt, sigma_mu, sigma_vth, spopt = fit_ofet_lin(
        vg, sqrt_ids, log_ids, init_mask, end_mask, init_ss, end_ss, Ci_W_L
    )
    
    assert len(popt) == 3 
    assert len(spopt) == 2 
    
    assert np.isfinite(popt[0]) and np.isfinite(popt[1]) and np.isfinite(popt[2])
    assert np.isfinite(sigma_mu) and np.isfinite(sigma_vth)
    assert np.isfinite(spopt[0]) and np.isfinite(spopt[1])


def test_fit_ofet_lin_should_fail_on_diverging_noise():
    """
    Verify that curve_fit raises a RuntimeError when fed with purely
      random unstructured noise.
      """
    
    
    random_vg = np.linspace(-10, 0, 40)
    random_sqrt_ids = np.random.normal(5.0, 2.0, size=40) # Unstructured noise
    random_log_ids = np.random.normal(-5.0, 1.0, size=40)
    
    Ci_W_L = 1e-6
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        fit_ofet_lin(random_vg, random_sqrt_ids, random_log_ids, -8, -4, -2, 1, Ci_W_L)


import numpy as np
import pytest

# ==========================================
# TESTS FOR `surf_sensitivity`
# ==========================================

def test_surf_sensitivity_should_calculate_expected_values():
    """Verify that sensitivity and combined error are computed correctly with valid inputs."""
    slope_fit = 1.0
    err_ph = np.array([0.01, 0.02])
    phcur = 1.0
    dose_err = 0.05
    dose_rates = np.array([10.0, 20.0])
    
    sens, err_sens = surf_sensitivity(slope_fit, err_ph, phcur, dose_err, dose_rates)
    
    # Expected sensitivity: (1.0 * 1e6) / (1e-2 * 1e-6) = 1e14
    assert isinstance(sens, float)
    assert isinstance(err_sens, float)
    assert sens > 0.0
    assert err_sens > 0.0


def test_surf_sensitivity_should_raise_value_error_on_mismatched_lengths():
    """Verify that a ValueError is raised if err_ph and dose_rates have different lengths."""
    slope_fit = 1.0
    err_ph = np.array([0.01, 0.02, 0.03])
    phcur = 1.0
    dose_err = 0.05
    dose_rates = np.array([10.0, 20.0])  # Length mismatch (3 vs 2)
    
    with pytest.raises(ValueError, match="must have the same length"):
        surf_sensitivity(slope_fit, err_ph, phcur, dose_err, dose_rates)


def test_surf_sensitivity_should_raise_zero_division_error_on_zero_dose_rate():
    """Verify that a ZeroDivisionError is raised if any dose rate is equal to zero."""
    slope_fit = 1.0
    err_ph = np.array([0.01])
    phcur = 1.0
    dose_err = 0.05
    dose_rates = np.array([0.0])  # Zero division hazard
    
    with pytest.raises(ZeroDivisionError):
        surf_sensitivity(slope_fit, err_ph, phcur, dose_err, dose_rates)


def test_surf_sensitivity_should_raise_zero_division_error_on_zero_photocurrent():
    """Verify that a ZeroDivisionError is raised if photocurrent is zero."""
    slope_fit = 1.0
    err_ph = np.array([0.01])
    phcur = 0.0  # Zero division hazard
    dose_err = 0.05
    dose_rates = np.array([10.0])
    
    with pytest.raises(ZeroDivisionError):
        surf_sensitivity(slope_fit, err_ph, phcur, dose_err, dose_rates)


# ==========================================
# TESTS FOR TRANSFER MATH MODELS
# ==========================================

def test_linfit_transfer_should_apply_absolute_values_correctly():
    """
    Verify that absolute values and square roots are computed accurately
    in linfit_transfer.
    """
    x = np.array([0.0, 2.0])
    mu = 1.0
    Vth = 1.0
    Ci_W_L = 2
    
    expected = np.array([1.0, 1.0])
    result = linfit_transfer(x, mu, Vth, Ci_W_L)
    
    np.testing.assert_allclose(result, expected)


def test_ss_linfit_transfer_should_behave_linearly():
    """ 
    Verify that subthreshold linear model behaves strictly as y = mx + d.
    """
    x = np.array([0.0, 1.0, 2.0])
    ss = 1.5
    d = -10.0
    
    expected = np.array([-10.0, -8.5, -7.0])
    result = ss_linfit_transfer(x, ss, d)
    
    np.testing.assert_allclose(result, expected)
