#!/usr/bin/env bash
# run_progressive_area.sh
# Safely runs the progressive clinical and pathological attributes sequentially
# on macOS to avoid Apple Silicon thread deadlocks and GIL contention.

set -e

echo "======================================================================"
echo "Starting Progressive ROSMAP AREA Run (Sequential macOS Workaround)"
echo "======================================================================"

# 1. Create the keep_bools file to restrict the run to only our new attributes
# This reduces the attributes from 11 to 5, cutting the runtime in half!
KEEP_FILE="results/keep_bools.txt"
mkdir -p results
cat << 'EOF' > "$KEEP_FILE"
NCI_vs_Rest
MCI_vs_Rest
AD_vs_Rest
Braak_III_VI_vs_0_II
Sex_Female
EOF

echo "--> Created $KEEP_FILE with targeted progressive attributes."

# 2. Set strict thread locks to prevent Apple Accelerate (vecLib) and OpenBLAS
# from spawning nested threads that clash with Python.
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "--> Applied macOS Apple Silicon thread locks."

# 3. Run AREA sequentially (-t 1)
# This is 100% stable, bypasses GIL locks, and will finish in ~15 minutes.
echo "--> STEP 3: Running AREA sequentially. Please wait..."
python3 /Users/ashlynsloane/Developer/area-workspace/AREA/run_area.py \
  -bf results/rosmap_area_bools.csv \
  -rf results/rosmap_area_ranks.csv \
  -jc sample_id \
  -od results/ \
  -t 1 \
  --keep-bool-columns "$KEEP_FILE"

echo "======================================================================"
echo "Progressive AREA Run Completed Successfully!"
echo "Results are saved under results/area_results/"
echo "======================================================================"
