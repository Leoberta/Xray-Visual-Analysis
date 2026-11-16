import pandas as pd
import numpy as np

def export_it_series_to_csv(data_dict, filename, sep=';'):
    """
    Exports a dictionary of It series arrays to a CSV file.
    Automatically pads shorter arrays with NaNs so all columns have equal length,
    preventing shape mismatch errors .
    """
    if not data_dict:
        print(f"Warning: Dictionary for {filename} is empty. Skipping export.")
        return

    # Find the maximum length among all vectors in the dictionary
    max_len = max(len(v) for v in data_dict.values())
    
    # Create a padded copy of the dictionary
    padded_data = {}
    for key, values in data_dict.items():
        arr = np.asarray(values, dtype=float)
        pad_width = max_len - len(arr)
        if pad_width > 0:
            padded_data[key] = np.pad(arr, (0, pad_width), constant_values=np.nan)
        else:
            padded_data[key] = arr

    # Create DataFrame and save to CSV
    df = pd.DataFrame(padded_data)
    df.to_csv(filename, index=False, sep=sep)
    print(f"-> Saved successfully: '{filename}'")


def export_results_to_csv(data_dict, filename, sep='\t'):
    """
    Exports a dictionary of results/ parameters
    to a CSV file without padding.
    """
    if not data_dict:
        print(f"Warning: Dictionary for {filename} is empty. Skipping export.")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, index=False, sep=sep)
    print(f"-> Saved successfully: '{filename}'")