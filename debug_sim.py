import sys
import traceback
try:
    from src.simulator import generate_dataset
    print("Import successful.")
    data = generate_dataset(10, 10, 0.4, 0.6)
    print(f"Generated {len(data)} items.")
    print(f"Item 0: {data[0]}")
except Exception:
    traceback.print_exc()
