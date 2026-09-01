import os
import sys
import subprocess

print("==================================================")
print("Diagnostic Tool for macOS Apple Silicon Deadlock")
print("==================================================")

# 1. Check NumPy Config
try:
    import numpy as np
    print("\n[1] NumPy Configuration:")
    # Use show_config to see BLAS library
    try:
        np.show_config()
    except Exception as e:
        print(f"Error printing config: {e}")
except ImportError:
    print("\n[1] NumPy is not installed in this environment.")

# 2. Inspect AREA runner.py
runner_path = "/Users/ashlynsloane/Developer/area-workspace/AREA/src/area/runner.py"
print(f"\n[2] Looking for AREA runner.py at: {runner_path}")

if os.path.exists(runner_path):
    print("Found runner.py! Inspecting threading implementation...")
    try:
        with open(runner_path, "r") as f:
            lines = f.readlines()
        
        print("\n--- runner.py contents ---")
        for i, line in enumerate(lines):
            # Print lines around ThreadPoolExecutor or thread usage
            print(f"{i+1:3d}: {line}", end="")
        print("\n--------------------------")
    except Exception as e:
        print(f"Error reading runner.py: {e}")
else:
    print(f"runner.py not found at {runner_path}. Let's look in the current directory...")
    # Try relative path
    alt_path = "src/area/runner.py"
    if os.path.exists(alt_path):
        print(f"Found runner.py at relative path: {alt_path}")
        try:
            with open(alt_path, "r") as f:
                lines = f.readlines()
            print("\n--- runner.py contents ---")
            for i, line in enumerate(lines):
                print(f"{i+1:3d}: {line}", end="")
            print("\n--------------------------")
        except Exception as e:
            print(f"Error reading runner.py: {e}")
    else:
        print("Could not locate runner.py. Please verify your folder structure.")

print("\n==================================================")
