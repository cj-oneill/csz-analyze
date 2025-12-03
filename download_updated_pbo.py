import requests
from pathlib import Path

# All the P-stations you need
p_stations = ['P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 
              'P400', 'P401', 'P402', 'P403', 'P418', 'P426', 'P427', 
              'P430', 'P438']

# Also the named stations
other_stations = ['NEAH', 'SEAT', 'SC02', 'TPW2']

all_stations = p_stations + other_stations

output_dir = Path("cascadia_positions")

print("Downloading P-stations from Nevada Geodetic Lab...")
print("Testing if these stations exist in NGL's database...")
print("=" * 70)

success = []
failed = []

for station in all_stations:
    print(f"Trying {station}...", end=" ")
    downloaded = False
    
    # Try multiple reference frames
    frames_to_try = [
        ('IGS20', f"http://geodesy.unr.edu/gps_timeseries/tenv3/IGS20/{station}.tenv3"),
        ('IGS14', f"http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{station}.tenv3"),
        ('IGS08', f"http://geodesy.unr.edu/gps_timeseries/tenv3/IGS08/{station}.tenv3"),
        ('NA', f"http://geodesy.unr.edu/gps_timeseries/tenv3/plates/NA/{station}.NA.tenv3"),
    ]
    
    for frame, url in frames_to_try:
        if downloaded:
            break
        
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                # Check if it's actual data (not an error page)
                content = response.text
                if len(content) > 1000 and not content.startswith('<!DOCTYPE'):
                    output_file = output_dir / f"{station}.NGL.{frame}.tenv3"
                    with open(output_file, 'w') as f:
                        f.write(content)
                    size = len(content) / 1024
                    print(f"✓ ({frame}, {size:.1f} KB)")
                    success.append(station)
                    downloaded = True
        except Exception as e:
            continue
    
    if not downloaded:
        print(f"✗ (Not in NGL database)")
        failed.append(station)

print("=" * 70)
print(f"\nResults:")
print(f"  Found in NGL: {len(success)}/{len(all_stations)} stations")

if success:
    print(f"\n  Successfully downloaded:")
    for s in success:
        print(f"    - {s}")

if failed:
    print(f"\n  Not available in NGL database:")
    for s in failed[:10]:
        print(f"    - {s}")
    if len(failed) > 10:
        print(f"    ... and {len(failed)-10} more")

# Count total stations we now have
all_tenv3 = list(output_dir.glob("*.tenv3"))
print(f"\n{'='*70}")
print(f"TOTAL STATIONS AVAILABLE: {len(all_tenv3)}")
print(f"{'='*70}")

if len(all_tenv3) >= 15:
    print(f"✓ Excellent! {len(all_tenv3)} stations is great for EEOF!")
elif len(all_tenv3) >= 8:
    print(f"✓ Good! {len(all_tenv3)} stations is sufficient for EEOF")
else:
    print(f"⚠ Only {len(all_tenv3)} stations available")

print(f"\nNext steps:")
print(f"  1. python load_gps_data.py  (load all .tenv3 files)")
print(f"  2. python eeof_analysis.py  (run EEOF analysis)")