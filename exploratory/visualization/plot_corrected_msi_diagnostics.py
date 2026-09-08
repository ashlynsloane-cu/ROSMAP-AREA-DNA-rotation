#!/usr/bin/env python3
"""
plot_corrected_msi_diagnostics.py
==============================================================================
Descriptive diagnostics for corrected AMS-AREA output.
==============================================================================

Does not impose a final MSI threshold.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True)
    p.add_argument("-o","--outdir",required=True)
    p.add_argument("--thresholds",default="0.25,0.5,0.75,1,1.25,1.5,2,2.5,3")
    return p.parse_args()

def main():
    args=parse_args()
    df=pd.read_csv(args.input)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    thresholds=[float(x) for x in args.thresholds.split(",") if x.strip()]
    sig=(df.Regular_FDR<.05)|(df.Weighted_FDR<.05)

    rows=[]
    for t in thresholds:
        rows.append({
            "abs_MSI_threshold":t,
            "sig_weighted_favored":int(np.sum(sig & (df.MSI < -t))),
            "sig_regular_favored":int(np.sum(sig & (df.MSI > t))),
            "sig_no_strong_preference":int(np.sum(sig & (df.MSI.abs() <= t))),
        })
    pd.DataFrame(rows).to_csv(out/"MSI_threshold_sensitivity_counts.csv",index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite=df.MSI.replace([np.inf,-np.inf],np.nan).dropna()
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.hist(finite,bins=80)
    ax.axvline(0,linestyle="--")
    ax.set_xlabel("Corrected MSI = log10(Pweighted / Pregular)")
    ax.set_ylabel("Genes")
    ax.set_title("Corrected MSI distribution")
    fig.tight_layout(); fig.savefig(out/"MSI_histogram.png",dpi=300); plt.close(fig)

    r=-np.log10(np.clip(df.Regular_P.to_numpy(float),1e-300,1))
    w=-np.log10(np.clip(df.Weighted_P.to_numpy(float),1e-300,1))
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(r,w,s=5,alpha=.3)
    lim=max(np.nanpercentile(r,99.5),np.nanpercentile(w,99.5),1)
    ax.plot([0,lim],[0,lim],linestyle="--")
    ax.set_xlim(0,lim); ax.set_ylim(0,lim)
    ax.set_xlabel("Regular -log10(corrected p)")
    ax.set_ylabel("Weighted -log10(corrected p)")
    fig.tight_layout(); fig.savefig(out/"Regular_vs_Weighted_neglog10P.png",dpi=300); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(np.maximum(r,w),df.MSI,s=5,alpha=.3)
    ax.axhline(0,linestyle="--")
    ax.set_xlabel("Overall association strength")
    ax.set_ylabel("Corrected MSI")
    fig.tight_layout(); fig.savefig(out/"MSI_vs_association_strength.png",dpi=300); plt.close(fig)

    print(f"Diagnostics written to {out}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
