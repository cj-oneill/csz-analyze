from pathlib import Path

pbo_file = Path("cascadia_positions/P395.pbo.igs14.pos")

print(f"Reading {pbo_file.name}")
print("=" * 70)

with open(pbo_file, 'r') as f:
    lines = f.readlines()
    
# Find where data starts (after "End Field Description" or similar)
for i, line in enumerate(lines):
    if i > 30:  # Start showing from line 30 onwards
        print(f"{i:3d}: {line.rstrip()}")
    if i > 45:  # Show ~15 data lines
        break