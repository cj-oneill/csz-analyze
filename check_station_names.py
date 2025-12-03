from pathlib import Path

data_dir = Path("cascadia_positions")

print("Checking PBO station names...")
print("=" * 70)

for pos_file in sorted(data_dir.glob("*.pbo.igs14.pos")):
    with open(pos_file, 'r') as f:
        for line in f:
            if "Station name" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    station_name = parts[1].strip()
                    print(f"{pos_file.stem.split('.')[0]:6s} -> {station_name}")
                break