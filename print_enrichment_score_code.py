import sys
import os

# Add the AREA source directory to Python's search path
sys.path.append("/Users/ashlynsloane/Developer/area-workspace/AREA")

try:
    import inspect
    import src.area.enrichment as enrichment
    
    print("=== Source code of compute_enrichment_score ===")
    print(inspect.getsource(enrichment.compute_enrichment_score))
    
    print("\n=== Source code of compute_nes_pvalue ===")
    print(inspect.getsource(enrichment.compute_nes_pvalue))
    
except Exception as e:
    print(f"Error: {e}")
