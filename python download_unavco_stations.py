import requests
import os
from pathlib import Path

# Configuration
stations = ['P393', 'P394', 'P395', 'P396', 'P397', 'P398', 'P399', 
            'P400', 'P401', 'P402', 'P403', 'P418', 'P426', 'P427', 
            'P430', 'P438', 'NEAH', 'SEAT', 'SC02', 'TPW2']

output_dir = Path("cascadia_positions")
output_dir.mkdir(exist_ok=True)

base_url = "https://data.unavco.org/archive/gnss/products/position"

print("Downloading Cascadia GPS stations from UNAVCO...")
print("=" * 60)

success = []
failed = []

for station in stations:
    url = f"{base_url}/{station}/{station}.nam14.pos"
    output_file = output_dir / f"{station}.nam14.pos"
    
    print(f"Downloading {station}...", end=" ")
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            file_size = len(response.content) / 1024  # KB
            print(f"✓ ({file_size:.1f} KB)")
            success.append(station)
        else:
            print(f"✗ (HTTP {response.status_code})")
            failed.append(station)
    except Exception as e:
        print(f"✗ (Error: {e})")
        failed.append(station)

print("=" * 60)
print(f"\nResults:")
print(f"  Successfully downloaded: {len(success)}/{len(stations)} stations")
if failed:
    print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Note: Failed stations may not have NAM14 products available.")
    print(f"  Try IGS14 or check Nevada Geodetic Lab for these stations.")

print(f"\nData saved to: {output_dir.absolute()}")