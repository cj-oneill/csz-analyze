import requests
import os
from pathlib import Path

# Configuration
stations = ['P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 
            'P400', 'P401', 'P402', 'P403', 'P418', 'P426', 'P427', 
            'P430', 'P438', 'NEAH', 'SEAT', 'SC02', 'TPW2']

output_dir = Path("cascadia_positions")
output_dir.mkdir(exist_ok=True)

# Nevada Geodetic Lab provides multiple reference frames
# Try NAM14 first, then IGS14 as backup
base_urls = {
    'NAM14': "http://geodesy.unr.edu/gps_timeseries/tenv3/NAM14",
    'IGS14': "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14"
}

print("Downloading Cascadia GPS stations from Nevada Geodetic Laboratory...")
print("=" * 70)

success = []
failed = []

for station in stations:
    print(f"Downloading {station}...", end=" ")
    
    downloaded = False
    
    # Try NAM14 first
    for frame, base_url in base_urls.items():
        if downloaded:
            break
            
        url = f"{base_url}/{station}.{frame}.tenv3"
        output_file = output_dir / f"{station}.{frame}.tenv3"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                file_size = len(response.content) / 1024  # KB
                print(f"✓ ({frame}, {file_size:.1f} KB)")
                success.append(f"{station} ({frame})")
                downloaded = True
            elif frame == 'IGS14':  # Only print error after trying both
                print(f"✗ (Not found in either frame)")
                failed.append(station)
        except Exception as e:
            if frame == 'IGS14':
                print(f"✗ (Error: {e})")
                failed.append(station)

print("=" * 70)
print(f"\nResults:")
print(f"  Successfully downloaded: {len(success)}/{len(stations)} stations")
if failed:
    print(f"  Failed: {', '.join(failed)}")
    print(f"\n  These stations may not exist or have different names.")

print(f"\nData saved to: {output_dir.absolute()}")

# Print file format info
print("\n" + "=" * 70)
print("File Format: TENV3 (Time, East, North, Vertical)")
print("Columns: Site, Date, DecYear, MJD, East(m), North(m), Up(m), sigE, sigN, sigU, ...")
print("Reference frames: NAM14 (preferred) or IGS14")