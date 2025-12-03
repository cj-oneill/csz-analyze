from pathlib import Path

# Read a PBO file to see its actual structure
pbo_file = Path("cascadia_positions/P395.pbo.igs14.pos")

print(f"Reading {pbo_file.name}")
print("=" * 70)

with open(pbo_file, 'r') as f:
    for i, line in enumerate(f):
        print(f"{i:3d}: {line.rstrip()}")
        if i > 30:  # Show first 30 lines
            print("...")
            break