import sys
import inspect

sys.path.append("/Users/ashlynsloane/Developer/area-workspace/AREA")

try:
    import src.area.enrichment as enrichment
    print("=== Source code of permute_enrichment_scores ===")
    print(inspect.getsource(enrichment.permute_enrichment_scores))
except Exception as e:
    print(f"Error: {e}")
