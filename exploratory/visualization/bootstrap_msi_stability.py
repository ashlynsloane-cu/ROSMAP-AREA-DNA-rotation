#!/usr/bin/env python3
"""
bootstrap_msi_stability.py
==============================================================================
Bootstrap stability of corrected AMS-AREA MSI for selected real genes.
==============================================================================

Uses corrected full-null Gaussian p-values in every bootstrap replicate.
"""

from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).resolve().parent
AREA_SCRIPT_DIR = VALIDATION_DIR.parent
if str(AREA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_SCRIPT_DIR))

from run_ams_area import (
    calculate_msi,
    compute_weighted_enrichment_score,
    configure_area_import,
    encode_trait_vector,
    full_null_gaussian_pvalue,
    load_expression,
    load_metadata,
    permute_weighted_enrichment_scores,
)

def get_vectors(trait, meta, order):
    source = meta.loc[order, trait["source_column"]]
    reg = encode_trait_vector(source, trait["regular"], trait["name"], "regular")
    if trait["type"].lower()=="binary_only":
        raise ValueError("Bootstrap MSI requires an AMS trait.")
    wgt = encode_trait_vector(source, trait["weighted"], trait["name"], "weighted")
    if not reg.notna().equals(wgt.notna()):
        raise ValueError("Regular/Weighted masks differ.")
    valid = reg.notna()
    ids = reg.index[valid].tolist()
    return ids, reg.loc[ids].to_numpy(float), wgt.loc[ids].to_numpy(float)

def choose_genes(results, genes_arg, gene_file, top_n):
    if genes_arg:
        return [x.strip() for x in genes_arg.split(",") if x.strip()]
    if gene_file:
        path = Path(gene_file).expanduser()
        if path.suffix.lower()==".csv":
            df = pd.read_csv(path)
            col = "gene_id" if "gene_id" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()
        return [x.strip() for x in path.read_text().splitlines() if x.strip()]

    sig = results[
        (results["Regular_FDR"]<0.05) | (results["Weighted_FDR"]<0.05)
    ].copy()
    sig["abs_MSI"] = sig["MSI"].abs()
    return sig.sort_values("abs_MSI",ascending=False)["gene_id"].head(top_n).astype(str).tolist()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-e","--expression",required=True)
    p.add_argument("-m","--metadata",required=True)
    p.add_argument("-c","--config",required=True)
    p.add_argument("--trait",required=True)
    p.add_argument("--results",required=True)
    p.add_argument("-o","--outdir",default="results/ams_area_validation/bootstrap_corrected")
    p.add_argument("--area-root",default=None)
    p.add_argument("--genes",default=None)
    p.add_argument("--gene-file",default=None)
    p.add_argument("--top-n",type=int,default=100)
    p.add_argument("--bootstraps",type=int,default=200)
    p.add_argument("--permutations",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--candidate-msi-threshold",type=float,default=None)
    p.add_argument("--msi-p-floor",type=float,default=1e-300)
    return p.parse_args()

def main():
    args = parse_args()
    configure_area_import(args.area_root)
    from src.area.enrichment import compute_enrichment_score, permute_enrichment_scores

    expr = load_expression(Path(args.expression).expanduser())
    meta = load_metadata(Path(args.metadata).expanduser())
    results = pd.read_csv(args.results)

    with open(args.config) as handle:
        config = json.load(handle)
    matches = [t for t in config["traits"] if t["name"]==args.trait and t.get("enabled",True)]
    if len(matches)!=1:
        raise ValueError("Trait not found uniquely.")
    trait=matches[0]

    common = sorted(set(expr.index).intersection(meta.index))
    random.Random(args.seed).shuffle(common)
    ids, reg, wgt = get_vectors(trait, meta, common)
    expr = expr.loc[ids]

    genes = choose_genes(results,args.genes,args.gene_file,args.top_n)
    missing=[g for g in genes if g not in expr.columns]
    if missing:
        raise ValueError(f"Genes missing from expression: {missing[:10]}")

    lookup=results.set_index("gene_id")
    matrix=expr[genes].to_numpy(float)
    ctrl=np.where(reg==0)[0]
    case=np.where(reg==1)[0]
    rng=np.random.default_rng(args.seed+700001)
    rows=[]

    for b in range(args.bootstraps):
        idx=np.concatenate([
            rng.choice(ctrl,size=len(ctrl),replace=True),
            rng.choice(case,size=len(case),replace=True),
        ])
        rng.shuffle(idx)
        rb,wb,xb=reg[idx],wgt[idx],matrix[idx]

        reg_null=permute_enrichment_scores(
            rb,n_permutations=args.permutations,seed=args.seed+b,xp=np,verbose=False
        )
        wgt_null=permute_weighted_enrichment_scores(
            wb,n_permutations=args.permutations,seed=args.seed+b
        )

        for j,gene in enumerate(genes):
            order=np.argsort(xb[:,j],kind="stable")
            re,*_=compute_enrichment_score(rb[order],xp=np,verbose=False)
            we=compute_weighted_enrichment_score(wb[order])
            rp,_=full_null_gaussian_pvalue(float(re),reg_null)
            wp,_=full_null_gaussian_pvalue(float(we),wgt_null)
            msi=float(calculate_msi(np.array([rp]),np.array([wp]),args.msi_p_floor)[0])
            rows.append({"bootstrap":b+1,"gene_id":gene,"Regular_P":rp,"Weighted_P":wp,"MSI":msi})

        if (b+1)%max(1,args.bootstraps//10)==0:
            print(f"Completed bootstrap {b+1}/{args.bootstraps}")

    boot=pd.DataFrame(rows)
    outdir=Path(args.outdir).expanduser()/args.trait
    outdir.mkdir(parents=True,exist_ok=True)
    boot.to_csv(outdir/"MSI_bootstrap_replicates.csv",index=False)

    summaries=[]
    for gene in genes:
        x=boot.loc[boot.gene_id==gene,"MSI"].dropna().to_numpy()
        q025=float(np.quantile(x,.025))
        q975=float(np.quantile(x,.975))
        row={
            "gene_id":gene,
            "original_MSI":float(lookup.loc[gene,"MSI"]),
            "original_Regular_FDR":float(lookup.loc[gene,"Regular_FDR"]),
            "original_Weighted_FDR":float(lookup.loc[gene,"Weighted_FDR"]),
            "bootstrap_n":len(x),
            "bootstrap_median_MSI":float(np.median(x)),
            "bootstrap_q025_MSI":q025,
            "bootstrap_q975_MSI":q975,
            "frac_MSI_lt_0":float(np.mean(x<0)),
            "frac_MSI_gt_0":float(np.mean(x>0)),
            "ci_excludes_zero":bool(q975<0 or q025>0),
        }
        if args.candidate_msi_threshold is not None:
            t=args.candidate_msi_threshold
            row["frac_MSI_lt_minus_threshold"]=float(np.mean(x < -t))
            row["frac_MSI_gt_plus_threshold"]=float(np.mean(x > t))
        summaries.append(row)

    pd.DataFrame(summaries).to_csv(outdir/"MSI_bootstrap_stability_summary.csv",index=False)
    print(f"Results written to {outdir}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
