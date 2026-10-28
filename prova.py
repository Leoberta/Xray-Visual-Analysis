### Initial file for drafts ###

### Coping some functions from previous Xray data analysis ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error


def load_It_data(file_path):
    # Carica i dati usando pandas
    df = pd.read_csv(file_path, skiprows=10, sep=None, engine='python')
    # Dato il formato dei file generato dalla strumentazione 
    # Time is the first column , Ids is the third
    time = df.iloc[:, 0] * 1e-3  # Convertiamo da ms a secondi 
    ids = df.iloc[:, 2]
    return time, ids



def normalize_It_manual(time, It, min_points):
    t_segment = np.asarray(time)
    it_segment = np.asarray(It)

    indices_minimi = []
    for i in min_points:
        idx = 2*i -1  # conversione effettuata poichè un punto ogni 0.5s
        idx_int = [idx-10, idx+10]  # Considera un intervallo intorno al punto specificato
        check_min = np.argmin(it_segment[idx_int[0]:idx_int[1]]) + idx_int[0]  # Controlla se è effettivamente un minimo
        indices_minimi.append(check_min)  # Trova l'indice del punto minimo specificato
        
    time_minimi = t_segment[indices_minimi]
    it_minimi = it_segment[indices_minimi]

    # 3. Fitting
    coeffs = np.polyfit(time_minimi, it_minimi, 3)  # fprse 3 meglio ?
    fit = np.polyval(coeffs, t_segment)
    par_fit = [coeffs] 

    # 4. Normalizzazione
    it_normalized = it_segment- fit  # Sottraiamo il fit per normalizzare 

    it_normalized = it_normalized - np.min(it_normalized[indices_minimi])    # Spostiamo il minimo a zero

    # 1. Calcolo del MAE
    mae = mean_absolute_error(it_minimi,  np.polyval(coeffs, time_minimi) )

    return it_normalized, par_fit , mae

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
    area = 1e-2 # in cm2 area campione
    muGy = 1e-6 # conversione da microGy a Gy
    A_to_muA = 1e6 # conversione da A a muA
    sens =(slope_fit * A_to_muA) / (area  * muGy) 
    err_sens = []
    for i in range(len(err_ph)):
        err_sens_i = ( (err_ph[i] / phcur) + (dose_err / dose_rates[i]) )
        err_sens.append(err_sens_i)
    err_sens = np.sqrt(np.sum(np.array(err_sens )**2)) *sens /16 # Errore totale combinato
    return sens, err_sens  # deve venire muC / (Gy x cm2)




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

def import_ofet_prob(path,skip, init, end):
    # Data Import
    data = pd.read_csv(path, skiprows= skip, sep=None , engine='python')     # sep=None con engine='python' rileva automaticamente spazi o tabulazion
    Vg1 = data.iloc[init:end, 0]
    Ids1 = data.iloc[init:end, 4] # Usiamo subito il valore assoluto
    #print(Ids1)
    abs_Ids1 =np.abs(Ids1)
    sqrt_Ids1 = np.sqrt(np.abs(Ids1))
    log_Ids1 = (np.log10(abs_Ids1))    
    return Vg1, abs_Ids1, sqrt_Ids1 , log_Ids1    

def fit_ofet_lin(Vg, sqrtI, logI, init_mask, end_mask, init_ss, end_ss):
    # Fitting 
    mask = (Vg > init_mask) & (Vg < end_mask)   
    popt, pcov = curve_fit(linfit_transfer, Vg[mask], sqrtI[mask], p0=[0.1, 0] )
    sigma_mu = np.sqrt(pcov[0,0])
    sigma_vth = np.sqrt(pcov[1,1])
    ss = (Vg > init_ss) & (Vg < end_ss) 
    spopt,scov= curve_fit(ss_linfit_transfer, Vg[ss], logI[ss], p0=[0.1, 0 ])
    # Ottieni il nome della variabile come stringa
    nome_sample = [var for var in globals() if globals()[var] is Vg][0]
    print ( f" {nome_sample}: mu={popt[0] :.5e} +/- {sigma_mu :.2e} cm^2/Vs, Vth={popt[1] :.5e} + {sigma_vth:.2e} V, SS={spopt[0] :.2e} +/- {np.sqrt(scov[0,0]) :.2e} V/dec")
    return popt,  sigma_mu, sigma_vth, spopt


    
# Definizione del modello di fitting
def sat_transfer(x, mu, Vth):
    return 0.5 * mu * Ci_W_L * (x - Vth)**2

def linfit_transfer(x, mu, Vth):
    return np.sqrt(0.5 * mu * Ci_W_L ) * np.abs(x - Vth)

def ss_fit_transfer(x, ss,mu, Vth):
    return 2*(x-Vth)*(Ci_W_L*mu) *ss

def ss_linfit_transfer(x, ss, d):
    return x *ss +d
