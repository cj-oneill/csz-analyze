import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress

# Helper function to compute and plot the PSD
def plot_psd(ax, data, title, color):
    """
    Computes and plots the Power Spectral Density (PSD).
    The X-axis is set to Frequency (cycles/year) and zoomed to focus on
    the month-to-10-year timescale.
    """
    # 1. Remove the mean and a linear trend (detrend)
    try:
        data = data.values # Ensure it's a NumPy array
    except:
        pass # Already a NumPy array

    # 2. Detrend the data
    days = np.arange(len(data))
    p = np.polyfit(days, data, 1)
    detrended_data = data - np.polyval(p, days)
    
    # 3. Compute PSD on a temporary axis to get Pxx and freqs without plotting
    # Fs=1.0 means frequencies are in cycles/day
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    Pxx, freqs = ax_temp.psd(detrended_data, NFFT=len(detrended_data), Fs=1.0, 
                        window=np.hanning(len(detrended_data)), scale_by_freq=True,
                        detrend='none', color=color, linewidth=1.5)
    plt.close(fig_temp) # Close temporary figure to avoid memory issues

    # Convert frequency from cycles/day to cycles/year
    freqs_cpy = freqs * 365.25

    # Plot manually on the target ax using log-log scale (standard for PSD)
    ax.loglog(freqs_cpy, Pxx, color=color, linewidth=1.5)
    
    ax.set_title(f'{title} PSD (Month - 10 Year Periods)', fontsize=12)
    ax.set_xlabel('Frequency (cycles/year)')
    # Power units are in distance^2 / frequency unit, so mm^2 / (cycles/year) = mm^2 * year
    ax.set_ylabel('Power ($\mathrm{mm}^2 \cdot \mathrm{year}$)')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- Set X-axis limits (in cycles/year) ---
    # 10 years period -> f_min = 0.1 cycles/year
    # 1 month period -> f_max = 365.25 / 30.4375 cycles/year (approx 12)
    f_min_cpy = 1 / 10.0      # 10-year frequency
    f_max_cpy = 365.25 / 30.4375 # 1-month frequency
    
    # Use slightly conservative limits for visualization
    ax.set_xlim(f_min_cpy * 0.5, f_max_cpy * 1.5)
    
    # Highlight the yearly (1.0) and semi-yearly (2.0) frequency in cycles/year
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.8, label='1-Year Period')
    ax.axvline(2.0, color='purple', linestyle='--', alpha=0.8, label='6-Month Period')
    ax.legend(fontsize=8, loc='lower left')


def plot_single_gnss_file(file_path, header_row=11):
    """
    Loads, cleans, plots, and performs time series and PSD analysis for a single GNSS file.
    
    Args:
        file_path (str): The path to the input CSV file.
        header_row (int): The zero-based index of the header row in the CSV.
    """
    print(f"--- Processing {file_path} ---")
    try:
        # --- 1. Load Data ---
        df = pd.read_csv(file_path, header=header_row)

        # --- 2. Clean and Prepare Data ---
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure we have clean numerical data, dropping NaNs in the component columns
        component_columns = ['North (mm)', 'East (mm)', 'Vertical (mm)']
        df_clean = df.dropna(subset=['Date'] + component_columns).copy()

        # Extract metadata from the filename
        parts = os.path.basename(file_path).split('.')
        station_id = parts[0].upper() if len(parts) > 0 else 'UNKNOWN'
        ref_frame = parts[2].upper() if len(parts) > 2 else 'UNKNOWN'
        
        # --- 3. Plotting (3 rows x 2 columns) ---
        # *** EDITED: Restored sharex='col'. This correctly links TS plots (Col 0) and PSD plots (Col 1) separately. ***
        fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex='col') 
        
        fig.suptitle(
            f"GNSS Station {station_id} Time Series and Spectral Analysis ({ref_frame} Frame)", 
            fontsize=18,
            y=1.00 # Adjusted position for larger title
        )
        
        plot_components = {
            0: {'column': 'North (mm)', 'color': 'red', 'title': 'North Component', 'ylabel': 'North Displacement (mm)'},
            1: {'column': 'East (mm)', 'color': 'green', 'title': 'East Component', 'ylabel': 'East Displacement (mm)'},
            2: {'column': 'Vertical (mm)', 'color': 'blue', 'title': 'Vertical Component', 'ylabel': 'Vertical Displacement (mm)'}
        }

        for i, (key, params) in enumerate(plot_components.items()):
            
            # --- Column 1: Time Series Plot ---
            ax_ts = axes[i, 0]
            ax_ts.plot(df_clean['Date'], df_clean[params['column']], color=params['color'], linewidth=1)
            ax_ts.set_ylabel(params['ylabel'])
            ax_ts.grid(True, linestyle='--', alpha=0.6)
            ax_ts.set_title(f"{params['title']} Time Series", fontsize=12)

            # --- Column 2: Power Spectral Density (PSD) Plot ---
            ax_psd = axes[i, 1]
            plot_psd(ax_psd, df_clean[params['column']], params['title'], params['color'])


        # Apply common settings to the Time Series column (Column 0)
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Rotate x-axis labels for better readability
        for i in range(3):
             plt.setp(axes[i, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Apply common settings to the PSD column (Column 1)
        # Note: Since sharex='col' is set, only the bottom-most PSD plot needs the x-label.
        axes[2, 1].set_xlabel('Frequency (cycles/year)')
        
        # --- Save the plot (changed to .svg for high-quality vector output) ---
        plot_filename = f"{station_id}_{ref_frame}_TS_PSD.svg"
        
        # Create 'figures' directory if it doesn't exist (ensuring the path works)
        output_dir = 'figures'
        os.makedirs(output_dir, exist_ok=True)
        full_plot_path = os.path.join(output_dir, plot_filename)
        
        # dpi=300 is removed as it's not strictly necessary for vector formats like SVG
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust for suptitle
        plt.savefig(full_plot_path)
        plt.close(fig) # Close the figure object

        print(f"Time series and PSD plot saved as: {full_plot_path}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.\n")
    except KeyError as e:
        print(f"Error: Required column missing or data is not clean in {file_path}. Details: {e}\n")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}\n")


def process_gnss_files(file_list):
    """
    Iterates through a list of file paths and plots the time series for each one.
    
    Args:
        file_list (list): A list of strings, where each string is a file path.
    """
    if not file_list:
        print("No files provided for processing.")
        return

    # Updated to prepend 'data/' path to each file name
    for file_name in file_list:
        file_path = os.path.join('data', file_name)
        plot_single_gnss_file(file_path)


if __name__ == "__main__":
    # --- Example of running the script with multiple files ---
    
    # NOTE: These file names are assumed to be located inside the 'data/' directory.
    
    files_to_process = [
        "P437.pbo.igs14.csv", 
        "P438.pbo.igs14.csv",
        "P439.pbo.igs14.csv", 
        "P439.pbo.nam14.csv",
    ]
    
    # Run the function with the list of files
    process_gnss_files(files_to_process)
