### Initial file for drafts ###

### Coping some functions from previous Xray data analysis ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error


def load_It_data(file_path):
    """Load current-time data from a text/dat file with structural validation."""
    try:
        # Load data using pandas with automatic separator detection
        df = pd.read_csv(file_path, skiprows=10, sep=None, engine='python')
    except Exception as e:
        raise ValueError(f"Failed to parse the file due to malformed structure or separators: {e}")
        
    # Check if DataFrame is empty or doesn't have enough columns (at least 3 expected: time, _, ids)
    if df.empty or df.shape[1] < 3:
        raise ValueError("The loaded file does not contain the expected number of columns or is empty.")
        
    time = df.iloc[:, 0] * 1e-3  # Convert from ms to seconds 
    ids = df.iloc[:, 2]
    
    # Check if the number of loaded elements is strictly less than 1000
    if len(time) < 1000 or len(ids) < 1000:
        raise ValueError(f"Dataset too small: found {len(time)} elements, but at least 1000 are required.")
        
    return time, ids

def normalize_It_manual(time, It, min_points):
    t_segment = np.asarray(time)
    it_segment = np.asarray(It)
    print(f"Loaded {len(t_segment)} time points and {len(it_segment)} current points.")

    if len(it_segment) < 1000 or len(t_segment) < 1000: 
        # minium number of points for 3 peaks of correct shape, this part can be modified if needed
        raise ValueError("Input arrays must have at least 1000 data points.")

    indices_minimi = []
    for i in min_points:
        idx = 2 * i - 1  # Conversione basata sul campionamento (un punto ogni 0.5s)
        idx_int = [idx - 10, idx + 10]  # Intervallo intorno al punto
        
        # Check boundary limits to prevent out of bounds or empty slices
        start_idx = max(0, idx_int[0])
        end_idx = min(len(it_segment), idx_int[1])
        print(f"Processing min_point {i}: idx {idx}, start_idx {start_idx}, end_idx {end_idx}")


        if start_idx >= end_idx or start_idx >= len(it_segment):
            raise ValueError(f"Min point index {i} (mapped to idx {idx}) falls outside valid array boundaries.")
            
        segment_slice = it_segment[start_idx:end_idx]
        if len(segment_slice) == 0:
            raise ValueError(f"Attempt to get argmin of an empty sequence at min_point {i}.")
            
        check_min = np.argmin(segment_slice) + start_idx
        indices_minimi.append(check_min)
        
    time_minimi = t_segment[indices_minimi]
    it_minimi = it_segment[indices_minimi]

    # 3. Fitting
    coeffs = np.polyfit(time_minimi, it_minimi, 3)
    fit = np.polyval(coeffs, t_segment)
    par_fit = [coeffs] 

    # 4. Normalizzazione
    it_normalized = it_segment - fit
    it_normalized = it_normalized - np.min(it_normalized[indices_minimi])

    # 1. Calcolo del MAE
    mae = mean_absolute_error(it_minimi, np.polyval(coeffs, time_minimi))

    return it_normalized, par_fit, mae


def photocurrent(Ids, prec, mae):
    Ids = np.asarray(Ids)
    
    # Trova gli indici di massimi e minimi
    indices_min_locali, _ = find_peaks(-Ids, prominence=prec)
    indices_max_locali, _ = find_peaks(Ids, prominence=prec)
    
    if len(indices_min_locali) == 0 or len(indices_max_locali) == 0:
        raise ValueError("Nessun picco trovato. Controlla il valore di 'prec'.")

    photocurrent_values = []

    # Iteriamo sui MINIMI per cercare il MASSIMO successivo
    for min_idx in indices_min_locali:
        # Trova i massimi che avvengono DOPO questo minimo
        massimi_successivi = indices_max_locali[indices_max_locali > min_idx]
        
        if len(massimi_successivi) > 0:
            # Prendi il massimo immediatamente successivo
            next_max_idx = massimi_successivi[0]
            
            if Ids[min_idx] >= 0:
                delta = Ids[next_max_idx] - Ids[min_idx]
                photocurrent_values.append(delta)
            else:
                delta = Ids[next_max_idx]
                photocurrent_values.append(delta)


    photocurrent_values = np.array(photocurrent_values)
    n= len(photocurrent_values)

    if n < 2:
        raise ValueError("Troppi pochi cicli completi (Min -> Max) trovati.")

    # Parametri statistici
    mean_photocurrent = np.mean(photocurrent_values)    # Calcolo dei parametri statistici stabili
    
    # Calcolo della semi-dispersione reale sui cicli accoppiati
    semi_disp =[]
    for i in range(len(photocurrent_values)-1):
        for j in range(i+1, len(photocurrent_values)):
            semi_disp.append( (photocurrent_values[j] - photocurrent_values[i]) / 2 )
    
    semi_disp = np.max(np.abs(semi_disp)) / np.sqrt(n-1) # prendo il valore normalizzato per il numero di picchi   
    #semi_disp = (np.max(photocurrent_values) - np.min(photocurrent_values)) / 2
    
    err_sis= 0 # stima rumore strumentale 
    err = np.sqrt(semi_disp**2 + mae**2 + err_sis**2)

    return mean_photocurrent, err

def surf_sensitivity(slope_fit, err_ph, phcur, dose_err, dose_rates):
    """Calculate surface sensitivity and combined error."""
    err_ph = np.asarray(err_ph)
    dose_rates = np.asarray(dose_rates)
    
    # Input validation
    if len(err_ph) != len(dose_rates):
        raise ValueError("err_ph and dose_rates must have the same length.")
    if phcur == 0 or np.any(dose_rates == 0):
        raise ZeroDivisionError("Photocurrent or dose rates cannot contain zero values.")
    if phcur < 0:
        raise ValueError("Photocurrent must be positive.")

    area = 1e-2  # cm2 sample area
    muGy = 1e-6  # conversion from microGy to Gy
    A_to_muA = 1e6  # conversion from A to microA
    
    sens = (slope_fit * A_to_muA) / (area * muGy) 
    
    err_sens = []
    for i in range(len(err_ph)):
        err_sens_i = (err_ph[i] / phcur) + (dose_err / dose_rates[i])
        err_sens.append(err_sens_i)
        
    # Combined total error (Note: division by 16 reflects your scaling/averaging factor)
    err_sens = np.sqrt(np.sum(np.array(err_sens)**2)) * sens / 16 
    
    return sens, err_sens  # units: microC / (Gy * cm^2)


def import_ofet(path,skip, init, end):
    # Data Import
    data = pd.read_csv(path, skiprows= skip, sep=r'\s+' , engine='python')     # sep=None con engine='python' rileva automaticamente spazi o tabulazion
    Vg1 = data.iloc[init:end, 3]
    Ids1 = data.iloc[init:end, 2] # Usiamo subito il valore assoluto
    #print(Ids1)
    abs_Ids1 =np.abs(Ids1)
    sqrt_Ids1 = np.sqrt(np.abs(Ids1))
    log_Ids1 = (np.log10(abs_Ids1))    
    return Vg1, abs_Ids1, sqrt_Ids1 , log_Ids1    

def fit_ofet_lin(Vg, sqrtI, logI, init_mask, end_mask, init_ss, end_ss, Ci_W_L):
    """Linear and subthreshold fitting for OFET characteristics."""
    mask = (Vg > init_mask) & (Vg < end_mask)
    
    if np.sum(mask) < 2:
        raise ValueError("Insufficient data points within the specified mask range for linear fit.")
        
    try:
        popt, pcov = curve_fit(linfit_transfer, Vg[mask], sqrtI[mask], p0=[0.1, 0, Ci_W_L])
        sigma_mu = np.sqrt(pcov[0, 0])
        sigma_vth = np.sqrt(pcov[1, 1])
    except (RuntimeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Linear curve fitting failed to converge: {e}")
    
    ss_mask = (Vg > init_ss) & (Vg < end_ss)
    if np.sum(ss_mask) < 2:
        raise ValueError("Insufficient data points within the specified subthreshold range.")
        
    try:
        spopt, scov = curve_fit(ss_linfit_transfer, Vg[ss_mask], logI[ss_mask], p0=[0.1, 0])
    except (RuntimeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Subthreshold curve fitting failed to converge: {e}")
    
    # Safely try to find the variable name in globals, fallback to 'Unknown_Sample' if not found
    matching_vars = [var for var in globals() if globals()[var] is Vg]
    nome_sample = matching_vars[0] if matching_vars else "Unknown_Sample"
    
    print(f" {nome_sample}: mu={popt[0]:.5e} +/- {sigma_mu:.2e} cm^2/Vs, "
          f"Vth={popt[1]:.5e} + {sigma_vth:.2e} V, SS={spopt[0]:.5e} +/- {np.sqrt(scov[0, 0]):.2e} V/dec")
          
    return popt, sigma_mu, sigma_vth, spopt

# Definizione del modello di fitting

def linfit_transfer(x, mu, Vth, Ci_W_L):
    return np.sqrt(0.5 * mu * Ci_W_L ) * np.abs(x - Vth)

def ss_linfit_transfer(x, ss, d):
    return x *ss +d