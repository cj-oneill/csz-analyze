import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def load_pbo_pos_file(filepath):
    """Load PBO .pos format files (US stations)"""
    data = []
    in_data = False
    
    with open(filepath, 'r') as f:
        for line in f:
            # Data starts after "End Field Description"
            if "End Field Description" in line:
                in_data = True
                continue
            
            if not in_data:
                continue
                
            # Skip comment/header lines
            if line.startswith('*') or len(line.strip()) == 0:
                continue
                
            parts = line.split()
            if len(parts) >= 24:  # Make sure we have all columns
                try:
                    # Column format from header:
                    # [0]YYYYMMDD [1]HHMMSS [2]MJD [3]X [4]Y [5]Z [6]Sx [7]Sy [8]Sz 
                    # [9]Rxy [10]Rxz [11]Ryz [12]NLat [13]Elong [14]Height 
                    # [15]dN [16]dE [17]dU [18]Sn [19]Se [20]Su [21]Rne [22]Rnu [23]Reu [24]Soln
                    
                    date_str = parts[0]  # YYYYMMDD
                    year = int(date_str[0:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    
                    # dN, dE, dU are differences from reference (columns 15, 16, 17)
                    # Already in meters, convert to mm
                    north = float(parts[15]) * 1000
                    east = float(parts[16]) * 1000
                    up = float(parts[17]) * 1000
                    
                    # Sigmas (columns 18, 19, 20)
                    sig_n = float(parts[18]) * 1000
                    sig_e = float(parts[19]) * 1000
                    sig_u = float(parts[20]) * 1000
                    
                    date = datetime(year, month, day)
                    data.append([date, east, north, up, sig_e, sig_n, sig_u])
                except (ValueError, IndexError) as e:
                    continue
    
    if len(data) == 0:
        return None
    
    df = pd.DataFrame(data, columns=['date', 'east', 'north', 'up', 'sig_e', 'sig_n', 'sig_u'])
    return df

def load_tenv3_file(filepath):
    """Load NGL .tenv3 format files (Canadian stations)"""
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Skip header line
            if line.startswith('site') or len(line.strip()) == 0:
                continue
                
            parts = line.split()
            if len(parts) >= 12:
                try:
                    # Format: site YYMMMDD yyyy.yyyy MJD week d reflon e0 east n0 north u0 up ant sigE sigN sigU ...
                    date_str = parts[1]  # Format: YYMMMDD like "94JAN01"
                    
                    # Parse year
                    year = int(date_str[0:2])
                    year += 2000 if year < 80 else 1900
                    
                    # Parse month
                    month_str = date_str[2:5]
                    months = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6,
                             'JUL':7, 'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
                    month = months[month_str]
                    
                    # Parse day
                    day = int(date_str[5:7])
                    
                    # Position columns (already in meters, convert to mm)
                    east = float(parts[8]) * 1000
                    north = float(parts[10]) * 1000
                    up = float(parts[12]) * 1000
                    sig_e = float(parts[14]) * 1000
                    sig_n = float(parts[15]) * 1000
                    sig_u = float(parts[16]) * 1000
                    
                    date = datetime(year, month, day)
                    data.append([date, east, north, up, sig_e, sig_n, sig_u])
                except (ValueError, IndexError, KeyError) as e:
                    continue
    
    if len(data) == 0:
        return None
    
    df = pd.DataFrame(data, columns=['date', 'east', 'north', 'up', 'sig_e', 'sig_n', 'sig_u'])
    return df

def load_all_stations(data_dir, start_date, end_date):
    """Load all GPS stations and filter by date range"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist!")
        return {}
    
    stations = {}
    
    print("Loading GPS data...")
    print("=" * 70)
    
    # Load PBO stations (.pos files)
    pos_files = list(data_dir.glob("*.pos"))
    print(f"Found {len(pos_files)} .pos files")
    
    for pos_file in sorted(pos_files):
        station = pos_file.stem.split('.')[0]
        print(f"Loading {station}...", end=" ")
        try:
            df = load_pbo_pos_file(pos_file)
            if df is not None and len(df) > 0:
                # Filter date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if len(df) > 0:
                    stations[station] = df
                    print(f"✓ ({len(df)} days)")
                else:
                    print("✗ (no data in date range)")
            else:
                print("✗ (no data loaded)")
        except Exception as e:
            print(f"✗ (error: {e})")
    
    # Load Canadian stations (.tenv3 files)
    tenv_files = list(data_dir.glob("*.tenv3"))
    print(f"\nFound {len(tenv_files)} .tenv3 files")
    
    for tenv_file in sorted(tenv_files):
        station = tenv_file.stem.split('.')[0]
        print(f"Loading {station}...", end=" ")
        try:
            df = load_tenv3_file(tenv_file)
            if df is not None and len(df) > 0:
                # Filter date range
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                if len(df) > 0:
                    stations[station] = df
                    print(f"✓ ({len(df)} days)")
                else:
                    print("✗ (no data in date range)")
            else:
                print("✗ (no data loaded)")
        except Exception as e:
            print(f"✗ (error: {e})")
    
    print("=" * 70)
    print(f"Loaded {len(stations)} stations")
    
    return stations

# Test the loading
if __name__ == "__main__":
    # Define date range for Winter 2022-23 ETS event
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2023, 4, 30)
    
    # Load all stations
    stations = load_all_stations("cascadia_positions", start_date, end_date)
    
    # Print summary
    if len(stations) > 0:
        print("\nStation Summary:")
        for station, df in sorted(stations.items()):
            print(f"  {station}: {len(df)} observations, "
                  f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        # Quick visualization
        if 'P395' in stations:
            df = stations['P395']
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            
            axes[0].plot(df['date'], df['east'], 'b-', linewidth=0.5)
            axes[0].set_ylabel('East (mm)')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('P395 - Vancouver Island (Oct 2022 - Apr 2023)')
            
            axes[1].plot(df['date'], df['north'], 'g-', linewidth=0.5)
            axes[1].set_ylabel('North (mm)')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(df['date'], df['up'], 'r-', linewidth=0.5)
            axes[2].set_ylabel('Up (mm)')
            axes[2].set_xlabel('Date')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('p395_timeseries.png', dpi=150)
            print("\n✓ Saved plot to p395_timeseries.png")
            plt.close()
    else:
        print("\nNo stations loaded!")