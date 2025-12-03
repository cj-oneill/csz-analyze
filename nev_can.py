import requests
from pathlib import Path

# Canadian stations
canadian_stations = ['ALBH', 'UCLU', 'PTRF', 'NANO', 'DRAO', 'HOLB', 'BAMF', 'TWHL']

output_dir = Path("cascadia_positions")
output_dir.mkdir(exist_ok=True)

print("Downloading Canadian stations from Nevada Geodetic Lab...")
print("=" * 70)

# NGL now uses IGS20 (as of August 2024) but IGS14 is still available
# URL format: https://geodesy.unr.edu/gps_timeseries/tenv3/{FRAME}/{STATION}.tenv3
# Or for plate frames: https://geodesy.unr.edu/gps_timeseries/tenv3/plates/{PLATE}/{STATION}.{PLATE}.tenv3

success = []
failed = []

for station in canadian_stations:
    print(f"Downloading {station}...", end=" ")
    downloaded = False
    
    # Try different reference frames and formats
    urls_to_try = [
        # IGS20 (current)
        f"https://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/{station}.tenv3",
        # IGS14 (legacy, but more data)
        f"https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3",
        # North America plate frame
        f"https://geodesy.unr.edu/gps_timeseries/tenv3/plates/NA/{station}.NA.tenv3",
    ]
    
    for url in urls_to_try:
        if downloaded:
            break
            
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:  # Make sure we got real data
                # Determine frame from URL
                if '/IGS20/' in url:
                    frame = 'IGS20'
                elif '/IGS14/' in url:
                    frame = 'IGS14'
                else:
                    frame = 'NA'
                    
                output_file = output_dir / f"{station}.{frame}.tenv3"
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                size = len(response.content) / 1024
                print(f"✓ ({frame}, {size:.1f} KB)")
                success.append(f"{station} ({frame})")
                downloaded = True
        except Exception as e:
            continue
    
    if not downloaded:
        print(f"✗ (Not found)")
        failed.append(station)

print("=" * 70)
print(f"\nResults:")
print(f"  Successfully downloaded: {len(success)}/{len(canadian_stations)} stations")

if failed:
    print(f"\n  Failed stations: {', '.join(failed)}")
    print(f"  These stations may need manual download from NRCan")
else:
    print(f"\n  ✓ All Canadian stations downloaded successfully!")

print(f"\nData saved to: {output_dir.absolute()}")