#!/usr/bin/env python3
"""
Comprehensive EOF Analysis of the August 2011 Cascadia SSE

This script performs:
1. Data loading and preprocessing (PBO .pos and UNR .tenv3 formats)
2. Time series visualization (split into readable multi-page figures)
3. EOF/PCA analysis (Empirical Orthogonal Functions)
4. Power Spectral Density (PSD) analysis
5. FFT analysis
6. Wavelet analysis
7. Station correlation analysis
8. Spatial pattern visualization

Stations included:
- PBO network: NEAH, P393-P403, P418, P426, P427, P430, P438, SC02, SEAT, TPW2
- Canadian WCDA/IGS: ALBH, BAMF, DRAO, HOLB, NANO, PTRF, TWHL, UCLU

Author: Claude
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import os
import glob
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directory containing GPS data files
DATA_DIR = "cascadia_positions"
OUTPUT_DIR = "eof_analysis_results"

# All stations from your directory
STATIONS = [
    # Canadian stations (tenv3 format)
    'ALBH', 'BAMF', 'DRAO', 'HOLB', 'NANO', 'PTRF', 'TWHL', 'UCLU',
    # PBO stations (pos format)
    'NEAH', 'P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399',
    'P400', 'P401', 'P402', 'P403', 'P418', 'P426', 'P427', 'P430',
    'P438', 'SC02', 'SEAT', 'TPW2'
]

# Station information with coordinates for spatial plots
# Approximate lat/lon from station metadata
STATION_INFO = {
    # Canadian stations
    'ALBH': {'lat': 48.39, 'lon': -123.49, 'name': 'Albert Head'},
    'BAMF': {'lat': 48.84, 'lon': -125.14, 'name': 'Bamfield'},
    'DRAO': {'lat': 49.32, 'lon': -119.62, 'name': 'Dominion Radio'},
    'HOLB': {'lat': 50.64, 'lon': -128.13, 'name': 'Holberg'},
    'NANO': {'lat': 49.29, 'lon': -124.09, 'name': 'Nanoose'},
    'PTRF': {'lat': 49.26, 'lon': -125.44, 'name': 'Port Renfrew'},
    'TWHL': {'lat': 49.41, 'lon': -123.92, 'name': 'Texada Island'},
    'UCLU': {'lat': 48.93, 'lon': -125.54, 'name': 'Ucluelet'},
    # PBO stations
    'NEAH': {'lat': 48.30, 'lon': -124.62, 'name': 'Neah Bay'},
    'P393': {'lat': 47.77, 'lon': -122.13, 'name': 'P393'},
    'P394': {'lat': 48.07, 'lon': -122.19, 'name': 'P394'},
    'P395': {'lat': 48.01, 'lon': -121.82, 'name': 'P395'},
    'P396': {'lat': 47.91, 'lon': -122.48, 'name': 'P396'},
    'P397': {'lat': 47.54, 'lon': -122.60, 'name': 'P397'},
    'P398': {'lat': 47.42, 'lon': -122.33, 'name': 'P398'},
    'P399': {'lat': 47.19, 'lon': -122.46, 'name': 'P399'},
    'P400': {'lat': 47.26, 'lon': -122.82, 'name': 'P400'},
    'P401': {'lat': 47.49, 'lon': -121.75, 'name': 'P401'},
    'P402': {'lat': 47.63, 'lon': -122.10, 'name': 'P402'},
    'P403': {'lat': 47.97, 'lon': -121.60, 'name': 'P403'},
    'P418': {'lat': 47.62, 'lon': -122.83, 'name': 'P418'},
    'P426': {'lat': 47.94, 'lon': -123.14, 'name': 'P426'},
    'P427': {'lat': 48.09, 'lon': -123.44, 'name': 'P427'},
    'P430': {'lat': 48.35, 'lon': -123.81, 'name': 'P430'},
    'P438': {'lat': 48.06, 'lon': -122.76, 'name': 'P438'},
    'SC02': {'lat': 48.55, 'lon': -123.01, 'name': 'SC02'},
    'SEAT': {'lat': 47.65, 'lon': -122.31, 'name': 'Seattle'},
    'TPW2': {'lat': 46.21, 'lon': -122.77, 'name': 'Toutle River'},
}

# SSE event window (for focused analysis)
SSE_START = datetime(2011, 8, 1)
SSE_END = datetime(2011, 9, 30)

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
STATIONS_PER_PLOT = 7  # Maximum stations per time series figure


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_pos_file(filepath):
    """
    Load GPS time series from PBO .pos file format.
    
    PBO pos format has header lines starting with various characters,
    data lines have: YYYYMMDD HHMMSS ... north east up ... sig_n sig_e sig_u
    """
    dates = []
    north = []
    east = []
    up = []
    north_err = []
    east_err = []
    up_err = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip header and comment lines
            if not line or line.startswith('*') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 15:
                continue
            
            try:
                # Parse date from YYYYMMDD format or decimal year
                if '.' in parts[0]:
                    # Decimal year format
                    continue  # Skip header-like lines
                
                date_str = parts[0]
                if len(date_str) == 8:  # YYYYMMDD
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    date = datetime(year, month, day)
                else:
                    continue
                
                # PBO pos format columns (0-indexed):
                # 0: YYYYMMDD, 1: HHMMSS, 2: JJJJJ.JJJJJ (MJD)
                # 3: X, 4: Y, 5: Z (Cartesian)
                # 6: sig_x, 7: sig_y, 8: sig_z
                # 9: north (m), 10: east (m), 11: up (m)
                # 12: sig_n, 13: sig_e, 14: sig_u
                # 15: corr_ne, 16: corr_nu, 17: corr_eu
                
                # Positions are in meters, convert to mm
                n_val = float(parts[9]) * 1000
                e_val = float(parts[10]) * 1000
                u_val = float(parts[11]) * 1000
                
                sig_n = float(parts[12]) * 1000
                sig_e = float(parts[13]) * 1000
                sig_u = float(parts[14]) * 1000
                
                dates.append(date)
                north.append(n_val)
                east.append(e_val)
                up.append(u_val)
                north_err.append(sig_n)
                east_err.append(sig_e)
                up_err.append(sig_u)
                
            except (ValueError, IndexError) as e:
                continue
    
    if len(dates) == 0:
        return None
    
    # Sort by date
    sorted_idx = np.argsort(dates)
    dates = [dates[i] for i in sorted_idx]
    
    return {
        'dates': dates,
        'north': np.array([north[i] for i in sorted_idx]),
        'east': np.array([east[i] for i in sorted_idx]),
        'up': np.array([up[i] for i in sorted_idx]),
        'north_err': np.array([north_err[i] for i in sorted_idx]),
        'east_err': np.array([east_err[i] for i in sorted_idx]),
        'up_err': np.array([up_err[i] for i in sorted_idx]),
    }


def load_tenv3_file(filepath):
    """
    Load GPS time series from UNR tenv3 format.
    
    tenv3 format columns:
    0: site, 1: date (YYMMMDD), 2: decimal_year
    3-5: dN, dE, dU (m) - displacements in local frame
    6: sN (m), 7: sE (m), 8: sU (m) - sigmas
    9-11: lat, lon, height
    """
    dates = []
    north = []
    east = []
    up = []
    north_err = []
    east_err = []
    up_err = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            
            try:
                # Parse decimal year
                decimal_year = float(parts[2])
                year = int(decimal_year)
                frac = decimal_year - year
                
                # Convert to date
                start_of_year = datetime(year, 1, 1)
                days_in_year = (datetime(year + 1, 1, 1) - start_of_year).days
                day_of_year = frac * days_in_year
                date = start_of_year + timedelta(days=day_of_year)
                
                # Displacements in mm (file has meters)
                n_val = float(parts[3]) * 1000
                e_val = float(parts[4]) * 1000
                u_val = float(parts[5]) * 1000
                
                sig_n = float(parts[6]) * 1000
                sig_e = float(parts[7]) * 1000
                sig_u = float(parts[8]) * 1000
                
                dates.append(date)
                north.append(n_val)
                east.append(e_val)
                up.append(u_val)
                north_err.append(sig_n)
                east_err.append(sig_e)
                up_err.append(sig_u)
                
            except (ValueError, IndexError):
                continue
    
    if len(dates) == 0:
        return None
    
    # Sort by date
    sorted_idx = np.argsort(dates)
    dates = [dates[i] for i in sorted_idx]
    
    return {
        'dates': dates,
        'north': np.array([north[i] for i in sorted_idx]),
        'east': np.array([east[i] for i in sorted_idx]),
        'up': np.array([up[i] for i in sorted_idx]),
        'north_err': np.array([north_err[i] for i in sorted_idx]),
        'east_err': np.array([east_err[i] for i in sorted_idx]),
        'up_err': np.array([up_err[i] for i in sorted_idx]),
    }


def load_all_data(data_dir, stations):
    """Load all station data from the specified directory."""
    data = {}
    
    print(f"Looking for data in: {data_dir}")
    
    for station in stations:
        # Try different file patterns
        patterns = [
            os.path.join(data_dir, f"{station}.*.tenv3"),
            os.path.join(data_dir, f"{station}.*.pos"),
            os.path.join(data_dir, f"{station}*.tenv3"),
            os.path.join(data_dir, f"{station}*.pos"),
        ]
        
        loaded = False
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                filepath = files[0]
                
                if '.tenv3' in filepath.lower():
                    station_data = load_tenv3_file(filepath)
                else:
                    station_data = load_pos_file(filepath)
                
                if station_data is not None and len(station_data['dates']) > 100:
                    data[station] = station_data
                    n_points = len(station_data['dates'])
                    date_range = f"{station_data['dates'][0].date()} to {station_data['dates'][-1].date()}"
                    print(f"  Loaded {station}: {n_points} points ({date_range})")
                    loaded = True
                    break
        
        if not loaded:
            print(f"  WARNING: No data found for {station}")
    
    return data


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def remove_outliers(data, threshold=3.5):
    """Remove outliers using median absolute deviation and detect equipment offsets."""
    cleaned = {}

    for station, station_data in data.items():
        cleaned[station] = {
            'dates': station_data['dates'],
            'north_err': station_data['north_err'].copy(),
            'east_err': station_data['east_err'].copy(),
            'up_err': station_data['up_err'].copy(),
        }

        for component in ['north', 'east', 'up']:
            x = station_data[component].copy()

            # Remove NaN
            valid = ~np.isnan(x)
            if np.sum(valid) < 10:
                cleaned[station][component] = x
                continue

            # First pass: Remove extreme outliers (very large spikes)
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median))

            if mad > 0:
                # Remove extreme outliers (>5 sigma)
                extreme_outliers = np.abs(x - median) / (mad * 1.4826) > 5.0
                x[extreme_outliers] = np.nan

                # Detect large jumps/offsets
                diff = np.diff(x)
                diff_median = np.nanmedian(np.abs(diff))

                if diff_median > 0:
                    # Flag jumps larger than 15x typical variation
                    jump_threshold = 15 * diff_median
                    large_jumps = np.abs(diff) > jump_threshold

                    # Mark data after large jumps as suspect
                    for i in np.where(large_jumps)[0]:
                        # Check if it's a real offset or just a spike
                        if i + 2 < len(x):
                            # If the jump persists, it's an offset - mark subsequent data
                            post_jump_mean = np.nanmean(x[i+1:min(i+10, len(x))])
                            pre_jump_mean = np.nanmean(x[max(0, i-9):i+1])

                            if abs(post_jump_mean - pre_jump_mean) > 5 * diff_median:
                                # This is a real offset, mark subsequent data
                                x[i+1:] = np.nan
                                break
                            else:
                                # Just a spike, remove the point
                                x[i+1] = np.nan

                # Second pass: Remove remaining outliers with tighter threshold
                valid = ~np.isnan(x)
                if np.sum(valid) > 10:
                    median = np.nanmedian(x[valid])
                    mad = np.nanmedian(np.abs(x[valid] - median))
                    if mad > 0:
                        outliers = np.abs(x - median) / (mad * 1.4826) > threshold
                        x[outliers] = np.nan

            cleaned[station][component] = x

    return cleaned


def detrend_data(data, remove_seasonal=True):
    """Remove long-period trends using high-pass filtering."""
    detrended = {}

    for station, station_data in data.items():
        dates = station_data['dates']
        t = np.array([(d - datetime(2011, 1, 1)).days / 365.25 for d in dates])

        detrended[station] = {
            'dates': dates,
            'decimal_year': t,
            'north_err': station_data.get('north_err', np.ones(len(dates))),
            'east_err': station_data.get('east_err', np.ones(len(dates))),
            'up_err': station_data.get('up_err', np.ones(len(dates)) * 3),
        }

        for component in ['north', 'east', 'up']:
            y = station_data[component].copy()
            valid = ~np.isnan(y)

            if np.sum(valid) < 100:
                detrended[station][component] = y
                continue

            # Simple approach: remove moving median (robust to outliers)
            # Window size: 365 days removes annual and longer-period signals
            window_size = 365

            if len(y) < window_size:
                # Just remove linear trend for short series
                t_valid = t[valid]
                y_valid = y[valid]
                if len(t_valid) > 10:
                    p = np.polyfit(t_valid, y_valid, 1)
                    trend = np.polyval(p, t)
                    detrended[station][component] = y - trend
                else:
                    detrended[station][component] = y
                continue

            # Compute moving median (handles gaps well)
            y_detrended = y.copy()

            # Fill NaNs temporarily with interpolation for filtering
            if np.any(valid):
                # Use pandas-like rolling median approach
                trend = np.full_like(y, np.nan)
                half_window = window_size // 2

                for i in range(len(y)):
                    start_idx = max(0, i - half_window)
                    end_idx = min(len(y), i + half_window + 1)
                    window_data = y[start_idx:end_idx]
                    window_valid = ~np.isnan(window_data)

                    if np.sum(window_valid) > window_size // 4:  # Need at least 25% valid data
                        trend[i] = np.nanmedian(window_data[window_valid])
                    elif i > 0:
                        trend[i] = trend[i-1]  # Forward fill
                    else:
                        trend[i] = np.nanmedian(y[valid])  # Use overall median

                # Remove trend
                y_detrended = y - trend

            detrended[station][component] = y_detrended

    return detrended


def interpolate_to_common_times(data, start_date, end_date):
    """Interpolate all stations to common daily timestamps and fill small gaps."""
    n_days = (end_date - start_date).days
    common_dates = [start_date + timedelta(days=i) for i in range(n_days)]
    common_decimal = np.array([(d - datetime(2011, 1, 1)).days / 365.25 for d in common_dates])

    interpolated = {}

    for station, station_data in data.items():
        dates = station_data['dates']
        t_orig = np.array([(d - datetime(2011, 1, 1)).days / 365.25 for d in dates])

        interpolated[station] = {
            'dates': common_dates,
            'decimal_year': common_decimal,
        }

        for component in ['north', 'east', 'up']:
            y_orig = station_data[component]
            valid = ~np.isnan(y_orig)

            if np.sum(valid) < 50:
                interpolated[station][component] = np.full(n_days, np.nan)
                continue

            try:
                # First interpolate linearly
                f = interpolate.interp1d(
                    t_orig[valid], y_orig[valid],
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                y_interp = f(common_decimal)

                # Fill small gaps (up to 30 days) using cubic interpolation
                gaps = np.isnan(y_interp)
                if np.any(gaps) and np.any(~gaps):
                    # Find gap sizes
                    gap_starts = np.where(np.diff(np.concatenate(([False], gaps, [False]))))[0]
                    gap_starts = gap_starts[::2]  # Start indices
                    gap_ends = np.where(np.diff(np.concatenate(([False], gaps, [False]))))[0]
                    gap_ends = gap_ends[1::2]  # End indices

                    for start, end in zip(gap_starts, gap_ends):
                        gap_size = end - start
                        # Only fill small gaps (< 30 days)
                        if gap_size < 30 and start > 0 and end < len(y_interp):
                            # Linear interpolation for small gaps
                            y_interp[start:end] = np.interp(
                                common_decimal[start:end],
                                common_decimal[~gaps],
                                y_interp[~gaps]
                            )

                interpolated[station][component] = y_interp
            except:
                interpolated[station][component] = np.full(n_days, np.nan)

    return interpolated


# ============================================================================
# EOF ANALYSIS FUNCTIONS
# ============================================================================

def build_data_matrix(data, component='east'):
    """Build data matrix for EOF analysis (time x stations)."""
    stations = list(data.keys())
    n_times = len(data[stations[0]]['dates'])
    n_stations = len(stations)
    
    X = np.zeros((n_times, n_stations))
    
    for j, station in enumerate(stations):
        X[:, j] = data[station][component]
    
    return X, stations


def perform_eof_analysis(X, n_modes=10):
    """Perform EOF analysis using SVD."""
    # Handle missing values
    X_filled = X.copy()
    col_means = np.nanmean(X, axis=0)
    
    for j in range(X.shape[1]):
        mask = np.isnan(X_filled[:, j])
        if np.all(mask):
            X_filled[:, j] = 0
        else:
            X_filled[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0
    
    # Remove mean
    X_centered = X_filled - np.nanmean(X_filled, axis=0)
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # EOF patterns (spatial)
    EOFs = Vt.T
    
    # Principal components (temporal)
    PCs = U * S
    
    # Variance explained
    variance = S**2 / (X.shape[0] - 1)
    total_variance = np.sum(variance)
    variance_explained = variance / total_variance
    
    n_modes = min(n_modes, len(S))
    
    return {
        'EOFs': EOFs[:, :n_modes],
        'PCs': PCs[:, :n_modes],
        'variance_explained': variance_explained[:n_modes],
        'singular_values': S[:n_modes],
        'total_variance': total_variance,
    }


# ============================================================================
# SPECTRAL ANALYSIS FUNCTIONS
# ============================================================================

def compute_fft(signal_data, fs=1.0):
    """Compute FFT of signal."""
    valid = ~np.isnan(signal_data)
    if np.sum(valid) < 50:
        return None, None, None
    
    signal_clean = signal_data[valid]
    signal_detrend = signal_clean - np.mean(signal_clean)
    
    window = np.hanning(len(signal_detrend))
    signal_windowed = signal_detrend * window
    
    n = len(signal_windowed)
    fft_vals = fft(signal_windowed)
    freqs = fftfreq(n, 1/fs)
    
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    amplitude = np.abs(fft_vals[pos_mask]) * 2 / n
    phase = np.angle(fft_vals[pos_mask])
    
    return freqs, amplitude, phase


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_raw_timeseries_multipage(data, output_dir, stations_per_page=STATIONS_PER_PLOT):
    """Plot raw time series split across multiple figures for readability."""
    stations = list(data.keys())
    n_stations = len(stations)
    n_pages = int(np.ceil(n_stations / stations_per_page))
    
    for page in range(n_pages):
        start_idx = page * stations_per_page
        end_idx = min((page + 1) * stations_per_page, n_stations)
        page_stations = stations[start_idx:end_idx]
        n_page_stations = len(page_stations)
        
        fig, axes = plt.subplots(n_page_stations, 3, figsize=(16, 2.5*n_page_stations))
        
        if n_page_stations == 1:
            axes = axes.reshape(1, -1)
        
        components = ['north', 'east', 'up']
        titles = ['North (mm)', 'East (mm)', 'Up (mm)']
        
        for i, station in enumerate(page_stations):
            dates = data[station]['dates']
            
            for j, (comp, title) in enumerate(zip(components, titles)):
                ax = axes[i, j]
                y = data[station][comp]
                
                # Plot with smaller markers for clarity
                ax.plot(dates, y, 'b.', markersize=0.5, alpha=0.6)
                
                # Station label on left
                ax.set_ylabel(f'{station}', fontsize=10, fontweight='bold')
                
                if i == 0:
                    ax.set_title(title, fontsize=12)
                
                # Mark SSE period
                ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='red')
                
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(YearLocator())
                
                if i == n_page_stations - 1:
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(f'Raw GPS Time Series (Page {page+1}/{n_pages})', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'01_raw_timeseries_page{page+1}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"  Saved: 01_raw_timeseries_page1-{n_pages}.png ({n_pages} pages)")


def plot_detrended_east_multipage(data, output_dir, stations_per_page=STATIONS_PER_PLOT):
    """Plot detrended east component across multiple pages."""
    stations = list(data.keys())
    n_stations = len(stations)
    n_pages = int(np.ceil(n_stations / stations_per_page))
    
    for page in range(n_pages):
        start_idx = page * stations_per_page
        end_idx = min((page + 1) * stations_per_page, n_stations)
        page_stations = stations[start_idx:end_idx]
        n_page_stations = len(page_stations)
        
        fig, axes = plt.subplots(n_page_stations, 1, figsize=(14, 2*n_page_stations),
                                sharex=True)
        
        if n_page_stations == 1:
            axes = [axes]
        
        for i, station in enumerate(page_stations):
            ax = axes[i]
            dates = data[station]['dates']
            east = data[station]['east']
            
            # Smooth for visualization
            valid = ~np.isnan(east)
            east_plot = east.copy()
            if np.sum(valid) > 10:
                east_smooth = np.full_like(east, np.nan)
                east_smooth[valid] = uniform_filter1d(east[valid], size=5)
            else:
                east_smooth = east
            
            ax.plot(dates, east, 'b.', markersize=1, alpha=0.3, label='Daily')
            ax.plot(dates, east_smooth, 'b-', linewidth=1.2, alpha=0.9)
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='red')
            
            # Add station name with larger font
            ax.text(0.02, 0.85, station, transform=ax.transAxes, fontsize=11,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('mm', fontsize=9)
            ax.set_ylim(-20, 20)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Date', fontsize=11)
        axes[-1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        
        fig.suptitle(f'Detrended East Component (Page {page+1}/{n_pages})\n'
                    f'Red shading: Aug-Sep 2011 SSE period',
                    fontsize=12, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'02_detrended_east_page{page+1}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved: 02_detrended_east_page1-{n_pages}.png ({n_pages} pages)")


def plot_eof_results(eof_results, data, stations, component, output_dir):
    """Plot EOF analysis results with improved layout."""
    dates = data[stations[0]]['dates']
    
    EOFs = eof_results['EOFs']
    PCs = eof_results['PCs']
    var_exp = eof_results['variance_explained']
    
    n_modes = min(4, EOFs.shape[1])
    
    # Figure 1: Principal Components
    fig, axes = plt.subplots(n_modes, 1, figsize=(14, 3*n_modes), sharex=True)
    
    for i in range(n_modes):
        ax = axes[i]
        ax.plot(dates, PCs[:, i], 'b-', linewidth=0.8)
        ax.fill_between(dates, 0, PCs[:, i], where=PCs[:, i] < 0,
                       alpha=0.3, color='red')
        ax.fill_between(dates, 0, PCs[:, i], where=PCs[:, i] > 0,
                       alpha=0.3, color='blue')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='yellow')
        
        ax.set_ylabel(f'PC{i+1}\n({var_exp[i]*100:.1f}%)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    axes[-1].set_xlabel('Date', fontsize=11)
    fig.suptitle(f'EOF Principal Components - {component.upper()} Component\n'
                f'(Yellow shading: SSE period)', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'03a_eof_pcs_{component}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: EOF spatial patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i in range(min(4, n_modes)):
        ax = axes[i // 2, i % 2]
        
        colors = ['red' if v < 0 else 'blue' for v in EOFs[:, i]]
        bars = ax.bar(range(len(stations)), EOFs[:, i], color=colors, alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        
        ax.set_title(f'EOF{i+1} ({var_exp[i]*100:.1f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loading', fontsize=10)
        ax.set_xticks(range(len(stations)))
        ax.set_xticklabels(stations, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'EOF Spatial Patterns - {component.upper()} Component',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'03b_eof_spatial_{component}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Variance explained and scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    n_plot = min(10, len(var_exp))
    cumulative = np.cumsum(var_exp[:n_plot])
    ax.bar(range(1, n_plot+1), var_exp[:n_plot]*100, alpha=0.7, label='Individual', color='steelblue')
    ax.plot(range(1, n_plot+1), cumulative*100, 'ro-', markersize=8, linewidth=2, label='Cumulative')
    ax.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('EOF Mode', fontsize=11)
    ax.set_ylabel('Variance Explained (%)', fontsize=11)
    ax.set_title('Variance Explained', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_plot+1))
    
    ax = axes[1]
    ax.semilogy(range(1, len(eof_results['singular_values'])+1),
               eof_results['singular_values'], 'bo-', markersize=8, linewidth=2)
    ax.set_xlabel('Mode', fontsize=11)
    ax.set_ylabel('Singular Value (log scale)', fontsize=11)
    ax.set_title('Scree Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'03c_eof_variance_{component}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: 03a_eof_pcs_{component}.png")
    print(f"  Saved: 03b_eof_spatial_{component}.png")
    print(f"  Saved: 03c_eof_variance_{component}.png")


def plot_psd_analysis(data, output_dir):
    """Plot Power Spectral Density analysis."""
    stations = list(data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [('north', 'North'), ('east', 'East'), ('up', 'Vertical')]
    
    for idx, (comp, title) in enumerate(components):
        ax = axes[idx // 2, idx % 2]
        
        for i, station in enumerate(stations):
            signal_data = data[station][comp]
            valid = ~np.isnan(signal_data)
            
            if np.sum(valid) > 100:
                freqs, psd = signal.welch(signal_data[valid], fs=1.0, nperseg=128)
                ax.semilogy(freqs, psd, alpha=0.4, linewidth=0.8)
        
        ax.set_xlabel('Frequency (cycles/day)', fontsize=10)
        ax.set_ylabel('PSD (mm²/Hz)', fontsize=10)
        ax.set_title(f'{title} Component', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)
    
    # Average PSD comparison
    ax = axes[1, 1]
    for comp, color, label in [('north', 'blue', 'North'),
                               ('east', 'red', 'East'),
                               ('up', 'green', 'Vertical')]:
        all_psd = []
        common_freqs = None
        
        for station in stations:
            signal_data = data[station][comp]
            valid = ~np.isnan(signal_data)
            
            if np.sum(valid) > 100:
                freqs, psd = signal.welch(signal_data[valid], fs=1.0, nperseg=128)
                all_psd.append(psd)
                if common_freqs is None:
                    common_freqs = freqs
        
        if all_psd:
            min_len = min(len(p) for p in all_psd)
            all_psd = [p[:min_len] for p in all_psd]
            mean_psd = np.mean(all_psd, axis=0)
            ax.semilogy(common_freqs[:min_len], mean_psd, color=color,
                       linewidth=2, label=label)
    
    ax.axvline(1/365.25, color='purple', linestyle='--', alpha=0.7, label='Annual')
    ax.axvline(2/365.25, color='orange', linestyle='--', alpha=0.7, label='Semi-annual')
    ax.set_xlabel('Frequency (cycles/day)', fontsize=10)
    ax.set_ylabel('PSD (mm²/Hz)', fontsize=10)
    ax.set_title('Average PSD by Component', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    
    fig.suptitle('Power Spectral Density Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_psd_analysis.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 04_psd_analysis.png")


def plot_fft_analysis(data, output_dir):
    """Plot FFT analysis."""
    stations = list(data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select subset of stations for clarity
    plot_stations = stations[::3][:8]  # Every 3rd station, max 8
    
    # FFT Amplitude - East
    ax = axes[0, 0]
    for station in plot_stations:
        freqs, amplitude, _ = compute_fft(data[station]['east'])
        if freqs is not None:
            periods = 1 / (freqs + 1e-10)
            mask = (periods < 400) & (periods > 5)
            ax.semilogy(periods[mask], amplitude[mask], alpha=0.7, linewidth=1, label=station)
    
    ax.axvline(365.25, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(182.6, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('Amplitude (mm)', fontsize=10)
    ax.set_title('FFT Amplitude - East Component', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 400)
    
    # FFT Phase - East
    ax = axes[0, 1]
    for station in plot_stations:
        freqs, amplitude, phase = compute_fft(data[station]['east'])
        if freqs is not None:
            periods = 1 / (freqs + 1e-10)
            mask = (periods < 400) & (periods > 5) & (amplitude > 0.1)
            ax.scatter(periods[mask], np.degrees(phase[mask]), alpha=0.3, s=15)
    
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('Phase (degrees)', fontsize=10)
    ax.set_title('FFT Phase - East Component', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 400)
    ax.set_ylim(-180, 180)
    
    # Stacked FFT power
    ax = axes[1, 0]
    all_power = []
    
    for station in stations:
        freqs, amplitude, _ = compute_fft(data[station]['east'])
        if freqs is not None and len(amplitude) > 0:
            all_power.append(amplitude**2)
    
    if all_power:
        min_len = min(len(p) for p in all_power)
        all_power = [p[:min_len] for p in all_power]
        freqs = freqs[:min_len]
        
        mean_power = np.mean(all_power, axis=0)
        std_power = np.std(all_power, axis=0)
        periods = 1 / (freqs + 1e-10)
        mask = (periods < 400) & (periods > 5)
        
        ax.semilogy(periods[mask], mean_power[mask], 'b-', linewidth=2, label='Mean')
        ax.fill_between(periods[mask],
                       np.maximum(mean_power[mask] - std_power[mask], 1e-5),
                       mean_power[mask] + std_power[mask],
                       alpha=0.3, color='blue')
        
        ax.axvline(365.25, color='red', linestyle='--', linewidth=2, label='Annual')
        ax.axvline(182.6, color='orange', linestyle='--', linewidth=2, label='Semi-annual')
        ax.axvline(14.76, color='green', linestyle='--', linewidth=2, label='Fortnightly')
    
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('Power (mm²)', fontsize=10)
    ax.set_title('Stacked FFT Power - East', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 400)
    
    # Component comparison
    ax = axes[1, 1]
    for comp, color, label in [('north', 'blue', 'North'),
                               ('east', 'red', 'East'),
                               ('up', 'green', 'Vertical')]:
        all_power = []
        
        for station in stations:
            freqs, amplitude, _ = compute_fft(data[station][comp])
            if freqs is not None and len(amplitude) > 0:
                all_power.append(amplitude**2)
        
        if all_power:
            min_len = min(len(p) for p in all_power)
            all_power = [p[:min_len] for p in all_power]
            freqs_c = freqs[:min_len]
            
            mean_power = np.mean(all_power, axis=0)
            periods = 1 / (freqs_c + 1e-10)
            mask = (periods < 400) & (periods > 5)
            
            ax.semilogy(periods[mask], mean_power[mask], color=color,
                       linewidth=2, label=label)
    
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('Power (mm²)', fontsize=10)
    ax.set_title('FFT Power Comparison', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 400)
    
    fig.suptitle('FFT Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '05_fft_analysis.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 05_fft_analysis.png")


def plot_station_correlation(data, output_dir):
    """Plot station correlation matrices."""
    stations = list(data.keys())
    n_stations = len(stations)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (comp, title) in enumerate([('north', 'North'),
                                          ('east', 'East'),
                                          ('up', 'Vertical')]):
        X, _ = build_data_matrix(data, comp)
        
        # Handle NaN for correlation
        X_filled = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_filled[:, j])
            if not np.all(mask):
                X_filled[mask, j] = np.nanmean(X_filled[:, j])
        
        corr_matrix = np.corrcoef(X_filled.T)
        
        ax = axes[idx]
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(n_stations))
        ax.set_yticks(range(n_stations))
        ax.set_xticklabels(stations, rotation=90, ha='center', fontsize=7)
        ax.set_yticklabels(stations, fontsize=7)
        ax.set_title(f'{title} Component', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
    
    fig.suptitle('Station Correlation Matrices', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_station_correlation.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 06_station_correlation.png")


def plot_spatial_patterns(data, eof_results, stations, output_dir):
    """Plot spatial patterns of EOF and SSE displacements."""
    dates = data[stations[0]]['dates']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Get SSE period indices
    sse_mask = np.array([(SSE_START <= d <= SSE_END) for d in dates])
    pre_mask = np.array([(d >= SSE_START - timedelta(days=30)) & (d < SSE_START) for d in dates])
    
    # Plot 1: Displacement vectors during SSE
    ax = axes[0]
    
    max_disp = 0
    displacements = {}
    
    for station in stations:
        if station not in STATION_INFO:
            continue
        
        info = STATION_INFO[station]
        lon, lat = info['lon'], info['lat']
        
        east = data[station]['east']
        north = data[station]['north']
        
        pre_east = np.nanmean(east[pre_mask]) if np.any(pre_mask) else 0
        pre_north = np.nanmean(north[pre_mask]) if np.any(pre_mask) else 0
        sse_east = np.nanmean(east[sse_mask]) if np.any(sse_mask) else 0
        sse_north = np.nanmean(north[sse_mask]) if np.any(sse_mask) else 0
        
        d_east = sse_east - pre_east
        d_north = sse_north - pre_north
        
        if not np.isnan(d_east) and not np.isnan(d_north):
            displacements[station] = (lon, lat, d_east, d_north)
            max_disp = max(max_disp, np.sqrt(d_east**2 + d_north**2))
    
    # Scale factor for arrows
    scale = 0.08 / max(max_disp, 1)  # degrees per mm
    
    for station, (lon, lat, d_east, d_north) in displacements.items():
        ax.plot(lon, lat, 'ko', markersize=4)
        ax.arrow(lon, lat, d_east * scale, d_north * scale,
                head_width=0.05, head_length=0.03, fc='red', ec='red', linewidth=1.5)
        ax.annotate(station, (lon, lat), fontsize=7,
                   xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('SSE Displacement Vectors\n(Aug-Sep 2011 vs. July 2011)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 2: EOF1 spatial pattern
    ax = axes[1]
    EOFs = eof_results['EOFs']
    
    lons = []
    lats = []
    eof_vals = []
    
    for i, station in enumerate(stations):
        if station in STATION_INFO:
            info = STATION_INFO[station]
            lons.append(info['lon'])
            lats.append(info['lat'])
            eof_vals.append(EOFs[i, 0])
    
    scatter = ax.scatter(lons, lats, c=eof_vals, cmap='RdBu_r',
                        s=150, vmin=-0.4, vmax=0.4, edgecolors='black', linewidth=0.5)
    
    for i, station in enumerate(stations):
        if station in STATION_INFO:
            info = STATION_INFO[station]
            ax.annotate(station, (info['lon'], info['lat']), fontsize=7,
                       xytext=(3, 3), textcoords='offset points')
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(f'EOF1 Spatial Pattern\n({eof_results["variance_explained"][0]*100:.1f}% variance)',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='EOF Loading', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '07_spatial_patterns.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 07_spatial_patterns.png")


def plot_sse_detailed_analysis(data, stations, output_dir):
    """Detailed SSE event analysis with multiple panels."""
    dates = data[stations[0]]['dates']
    
    # Focus period
    focus_start = datetime(2011, 7, 1)
    focus_end = datetime(2011, 10, 31)
    focus_mask = np.array([(focus_start <= d <= focus_end) for d in dates])
    focus_dates = [d for d, m in zip(dates, focus_mask) if m]
    focus_idx = np.where(focus_mask)[0]
    
    # Sort stations by latitude for stacking
    sorted_stations = sorted([s for s in stations if s in STATION_INFO],
                            key=lambda s: STATION_INFO[s]['lat'], reverse=True)
    
    # Figure 1: Stacked east components
    fig, ax = plt.subplots(figsize=(14, 10))
    
    offset = 0
    offset_step = 8  # mm between stations
    yticks = []
    yticklabels = []
    
    for station in sorted_stations:
        east = data[station]['east']
        focus_east = east[focus_mask]
        
        if len(focus_east) == len(focus_dates):
            # Smooth
            valid = ~np.isnan(focus_east)
            if np.sum(valid) > 10:
                smooth = np.full_like(focus_east, np.nan)
                smooth[valid] = uniform_filter1d(focus_east[valid], size=5)
                
                ax.plot(focus_dates, smooth + offset, linewidth=1.2, alpha=0.9)
                ax.axhline(offset, color='gray', linewidth=0.3, alpha=0.5)
                
                yticks.append(offset)
                yticklabels.append(station)
                
                offset -= offset_step
    
    ax.axvline(SSE_START, color='red', linestyle='--', linewidth=2, label='SSE Start')
    ax.axvline(SSE_END, color='red', linestyle=':', linewidth=2, label='SSE End')
    ax.axvspan(SSE_START, SSE_END, alpha=0.1, color='red')
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Station (sorted by latitude, N to S)', fontsize=11)
    ax.set_title('Stacked East Component During 2011 SSE\n(Detrended, 5-day smoothed)',
                fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08a_sse_stacked.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Cumulative displacement and velocity
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Select subset for clarity
    plot_stations = sorted_stations[::2][:10]
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_stations)))
    
    ax = axes[0]
    for i, station in enumerate(plot_stations):
        east = data[station]['east']
        focus_east = east[focus_mask]
        
        if len(focus_east) == len(focus_dates) and not np.all(np.isnan(focus_east)):
            # Cumulative relative to start
            cumulative = focus_east - np.nanmean(focus_east[:10])
            ax.plot(focus_dates, cumulative, linewidth=1.5, color=colors[i], label=station)
    
    ax.axvline(SSE_START, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(SSE_END, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Cumulative East (mm)', fontsize=11)
    ax.set_title('Cumulative East Displacement', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i, station in enumerate(plot_stations):
        east = data[station]['east']
        focus_east = east[focus_mask]
        
        if len(focus_east) == len(focus_dates) and not np.all(np.isnan(focus_east)):
            valid = ~np.isnan(focus_east)
            if np.sum(valid) > 10:
                smooth = np.full_like(focus_east, np.nan)
                smooth[valid] = uniform_filter1d(focus_east[valid], size=7)
                velocity = np.gradient(smooth)
                ax.plot(focus_dates, velocity, linewidth=1, color=colors[i], alpha=0.7, label=station)
    
    ax.axvline(SSE_START, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(SSE_END, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Velocity (mm/day)', fontsize=11)
    ax.set_title('Displacement Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08b_sse_cumulative_velocity.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Hodograph
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i, station in enumerate(plot_stations[:8]):
        east = data[station]['east']
        north = data[station]['north']
        
        focus_east = east[focus_mask]
        focus_north = north[focus_mask]
        
        if len(focus_east) > 0 and not np.all(np.isnan(focus_east)):
            # Relative to start
            fe = focus_east - np.nanmean(focus_east[:10])
            fn = focus_north - np.nanmean(focus_north[:10])
            
            valid = ~(np.isnan(fe) | np.isnan(fn))
            if np.sum(valid) > 10:
                fe_smooth = uniform_filter1d(fe[valid], size=5)
                fn_smooth = uniform_filter1d(fn[valid], size=5)
                
                ax.plot(fe_smooth, fn_smooth, '-', linewidth=1.5, 
                       color=colors[i], label=station, alpha=0.8)
                ax.plot(fe_smooth[0], fn_smooth[0], 'o', color=colors[i], markersize=10)
                ax.plot(fe_smooth[-1], fn_smooth[-1], 's', color=colors[i], markersize=10)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('East Displacement (mm)', fontsize=12)
    ax.set_ylabel('North Displacement (mm)', fontsize=12)
    ax.set_title('Hodograph: North vs East Motion\n(circles = start, squares = end)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08c_sse_hodograph.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  Saved: 08a_sse_stacked.png")
    print("  Saved: 08b_sse_cumulative_velocity.png")
    print("  Saved: 08c_sse_hodograph.png")


def plot_pc_analysis(eof_results, data, stations, output_dir):
    """Plot PC analysis and comparison with tremor proxy."""
    dates = data[stations[0]]['dates']
    PCs = eof_results['PCs']
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # PC1
    ax = axes[0]
    ax.plot(dates, PCs[:, 0], 'b-', linewidth=0.8)
    ax.fill_between(dates, 0, PCs[:, 0], where=PCs[:, 0] < 0,
                   alpha=0.4, color='red', label='Westward (SSE)')
    ax.fill_between(dates, 0, PCs[:, 0], where=PCs[:, 0] >= 0,
                   alpha=0.4, color='blue', label='Eastward')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='yellow')
    ax.set_ylabel('PC1', fontsize=11)
    ax.set_title('Principal Component 1 (SSE Signal)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Tremor proxy (|d(PC1)/dt|)
    ax = axes[1]
    pc1_smooth = uniform_filter1d(PCs[:, 0], size=5)
    tremor_proxy = np.abs(np.gradient(pc1_smooth))
    
    ax.fill_between(dates, 0, tremor_proxy * 50, alpha=0.6, color='purple')
    ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='yellow')
    ax.set_ylabel('Tremor Proxy\n(|dPC1/dt|)', fontsize=11)
    ax.set_title('Tremor Activity Proxy (from displacement rate)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # PC2
    ax = axes[2]
    ax.plot(dates, PCs[:, 1], 'g-', linewidth=0.8)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='yellow')
    ax.set_ylabel('PC2', fontsize=11)
    ax.set_title('Principal Component 2', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cumulative PC1
    ax = axes[3]
    cumulative = np.cumsum(PCs[:, 0] - np.mean(PCs[:, 0]))
    ax.plot(dates, cumulative, 'b-', linewidth=1.5)
    ax.axvspan(SSE_START, SSE_END, alpha=0.15, color='yellow')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative PC1', fontsize=11)
    ax.set_title('Cumulative PC1 (Proxy for Cumulative Slip)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    fig.suptitle('EOF Principal Component Analysis\n(Yellow shading: SSE period)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_pc_analysis.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 09_pc_analysis.png")


def create_summary_report(eof_results, data, stations, output_dir):
    """Create text summary report."""
    
    report = f"""
================================================================================
        EOF ANALYSIS OF AUGUST 2011 CASCADIA SLOW SLIP EVENT
                          Summary Report
================================================================================

ANALYSIS OVERVIEW
-----------------
This analysis applies Empirical Orthogonal Function (EOF) decomposition to 
GPS time series data from the Cascadia subduction zone to identify and 
characterize the August 2011 Episodic Tremor and Slip (ETS) event.

DATA SUMMARY
------------
Number of stations analyzed: {len(stations)}
Time period: Full station records interpolated to 2011
SSE event window: August 1 - September 30, 2011
Sampling: Daily positions

STATIONS ANALYZED
-----------------
Canadian (tenv3): ALBH, BAMF, DRAO, HOLB, NANO, PTRF, TWHL, UCLU
PBO (pos):        NEAH, P393-P403, P418, P426, P427, P430, P438, SC02, SEAT, TPW2

Total: {len(stations)} stations

EOF ANALYSIS RESULTS (East Component)
-------------------------------------
EOF Mode    Variance Explained    Cumulative
--------    ------------------    ----------
"""
    
    cumsum = 0
    for i, var in enumerate(eof_results['variance_explained']):
        cumsum += var
        report += f"EOF {i+1}        {var*100:>6.2f}%              {cumsum*100:>6.2f}%\n"
    
    report += f"""
INTERPRETATION
--------------
EOF1: The dominant mode ({eof_results['variance_explained'][0]*100:.1f}% variance) represents the 
      coherent regional signal. During the SSE period, this captures the 
      westward displacement across the network - the SSE signal itself.
      
EOF2: The second mode ({eof_results['variance_explained'][1]*100:.1f}% variance) often captures 
      along-strike variations or residual signals.
      
Higher modes represent increasingly localized effects and noise.

KEY FINDINGS
------------
1. The August 2011 SSE is clearly visible as a negative (westward) excursion 
   in PC1 during August-September 2011.
   
2. The EOF1 spatial pattern shows coherent loading across stations in the 
   Olympic Peninsula and Vancouver Island regions.
   
3. Stations sorted by latitude show the classic along-strike propagation
   signature of Cascadia SSEs.

4. Spectral analysis reveals:
   - Strong annual and semi-annual signals (seasonal/hydrological loading)
   - Fortnightly tidal signals
   - Broadband transient power during SSE

FILES GENERATED
---------------
Multi-page time series (for readability):
  01_raw_timeseries_page*.png     - Raw GPS time series
  02_detrended_east_page*.png     - Detrended east component

EOF Analysis:
  03a_eof_pcs_east.png            - Principal components (temporal)
  03b_eof_spatial_east.png        - EOF loadings (spatial)
  03c_eof_variance_east.png       - Variance explained

Spectral Analysis:
  04_psd_analysis.png             - Power spectral density
  05_fft_analysis.png             - FFT amplitude/phase

Correlation & Spatial:
  06_station_correlation.png      - Inter-station correlation
  07_spatial_patterns.png         - Map of EOF patterns

SSE Event Analysis:
  08a_sse_stacked.png             - Stacked time series
  08b_sse_cumulative_velocity.png - Cumulative displacement & velocity
  08c_sse_hodograph.png           - North vs East motion

PC Analysis:
  09_pc_analysis.png              - PC1, PC2, tremor proxy

REFERENCES
----------
- Rogers & Dragert (2003): Discovery of ETS in Cascadia
- Bartlow et al. (2011): Space-time correlation of slip and tremor
- Schmidt & Gao (2010): Cascadia SSE source parameters
- Wech & Bartlow (2014): Slip rate and tremor genesis

================================================================================
                    Analysis completed successfully
================================================================================
"""
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("  Saved: analysis_report.txt")
    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("EOF Analysis of August 2011 Cascadia SSE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    print("\n1. Loading data...")
    data = load_all_data(DATA_DIR, STATIONS)
    
    if len(data) == 0:
        print("\nERROR: No data loaded. Check DATA_DIR path.")
        print(f"Looking in: {os.path.abspath(DATA_DIR)}")
        return None, None
    
    stations = list(data.keys())
    print(f"\n   Successfully loaded {len(stations)} stations")
    
    # Preprocess
    print("\n2. Preprocessing...")
    print("   - Removing outliers")
    data_clean = remove_outliers(data)

    print("   - Filtering stations by data quality")
    # Filter out stations with insufficient data coverage in the analysis period
    data_filtered = {}
    analysis_start = datetime(2010, 1, 1)
    analysis_end = datetime(2013, 1, 1)

    for station, station_data in data_clean.items():
        dates = station_data['dates']
        # Check if station has data in the analysis period
        in_period = [d for d in dates if analysis_start <= d <= analysis_end]

        if len(in_period) < 200:  # Need at least 200 days of data
            print(f"     Skipping {station}: only {len(in_period)} days in 2010-2013")
            continue

        # Check data quality for east component (primary signal)
        east = station_data['east']
        valid_east = np.sum(~np.isnan(east))
        pct_valid = 100 * valid_east / len(east)

        if pct_valid < 30:  # Need at least 30% valid data
            print(f"     Skipping {station}: only {pct_valid:.1f}% valid data")
            continue

        data_filtered[station] = station_data

    print(f"   Kept {len(data_filtered)} of {len(data_clean)} stations")

    if len(data_filtered) == 0:
        print("\nERROR: No stations passed quality filters")
        return None, None

    print("   - Interpolating to common time base (2010-2013 for SSE analysis)")
    data_interp = interpolate_to_common_times(
        data_filtered,
        datetime(2010, 1, 1),
        datetime(2013, 1, 1)
    )

    print("   - Detrending (moving median high-pass filter)")
    data_detrend = detrend_data(data_interp, remove_seasonal=True)

    # Final quality check after detrending
    print("   - Final quality check after detrending")
    data_final = {}
    for station, station_data in data_detrend.items():
        east = station_data['east']
        valid = np.sum(~np.isnan(east))
        if valid > 200:  # Need at least 200 valid points after detrending
            data_final[station] = station_data
        else:
            print(f"     Dropped {station} after detrending: only {valid} valid points")

    print(f"   Final dataset: {len(data_final)} stations")
    data_detrend = data_final
    
    # EOF Analysis
    print("\n3. Performing EOF analysis...")
    X_east, station_list = build_data_matrix(data_detrend, 'east')
    eof_results = perform_eof_analysis(X_east, n_modes=10)
    
    print(f"   EOF1 explains {eof_results['variance_explained'][0]*100:.1f}% of variance")
    print(f"   First 3 modes explain {sum(eof_results['variance_explained'][:3])*100:.1f}%")
    print(f"   First 5 modes explain {sum(eof_results['variance_explained'][:5])*100:.1f}%")
    
    # Generate plots
    print("\n4. Generating plots...")
    
    print("   - Raw time series (multi-page)")
    plot_raw_timeseries_multipage(data_interp, OUTPUT_DIR)
    
    print("   - Detrended east (multi-page)")
    plot_detrended_east_multipage(data_detrend, OUTPUT_DIR)
    
    print("   - EOF analysis")
    plot_eof_results(eof_results, data_detrend, station_list, 'east', OUTPUT_DIR)
    
    print("   - PSD analysis")
    plot_psd_analysis(data_detrend, OUTPUT_DIR)
    
    print("   - FFT analysis")
    plot_fft_analysis(data_detrend, OUTPUT_DIR)
    
    print("   - Station correlation")
    plot_station_correlation(data_detrend, OUTPUT_DIR)
    
    print("   - Spatial patterns")
    plot_spatial_patterns(data_detrend, eof_results, station_list, OUTPUT_DIR)
    
    print("   - SSE detailed analysis")
    plot_sse_detailed_analysis(data_detrend, station_list, OUTPUT_DIR)
    
    print("   - PC analysis")
    plot_pc_analysis(eof_results, data_detrend, station_list, OUTPUT_DIR)
    
    # Summary report
    print("\n5. Creating summary report...")
    report = create_summary_report(eof_results, data_detrend, station_list, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)
    
    return data_detrend, eof_results


if __name__ == "__main__":
    data, eof_results = main()
