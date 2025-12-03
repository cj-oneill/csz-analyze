from pathlib import Path

data_dir = Path("cascadia_positions")

# Check what files we have
pos_files = list(data_dir.glob("*.pos"))
tenv_files = list(data_dir.glob("*.tenv3"))

print("PBO .pos files found:")
for f in pos_files[:3]:  # Show first 3
    print(f"  {f.name}")
    print(f"  First few lines:")
    with open(f, 'r') as file:
        for i, line in enumerate(file):
            if i < 10:
                print(f"    {line.rstrip()}")
            else:
                break
    print()

print("\nTENV3 files found:")
for f in tenv_files[:3]:  # Show first 3
    print(f"  {f.name}")
    print(f"  First few lines:")
    with open(f, 'r') as file:
        for i, line in enumerate(file):
            if i < 10:
                print(f"    {line.rstrip()}")
            else:
                break
    print()