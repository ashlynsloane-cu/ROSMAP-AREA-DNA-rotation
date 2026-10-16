#!/usr/bin/env python3
"""
Generalized covariate-adjusted binary Regular AREA.

Core inference matches the validated ROSMAP implementation:
- average ranks for ties
- FWL residualization of feature ranks and binary phenotype
- within-batch/block exchangeability
- analytic blockwise permutation variance
- two-sided Gaussian p-values
- BH FDR

Input expression may be genes/proteins x samples OR samples x features.
No imputation is performed. All selected values must be finite.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfc

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--expression",required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--outdir",required=True)
    p.add_argument("--contrast-name",required=True)
    p.add_argument("--group-a",required=True)
    p.add_argument("--group-b",required=True)
    p.add_argument("--sample-col",default="sample_id")
    p.add_argument("--group-col",default="analysis_group")
    p.add_argument("--orientation",choices=["features_by_samples","samples_by_features"],required=True)
    p.add_argument("--feature-id-col",default=None)
    p.add_argument("--continuous",default="")
    p.add_argument("--categorical",default="")
    p.add_argument(
        "--block",
        default=None,
        help="Optional blocking variable for analytic permutation variance."
    )
    p.add_argument("--fdr",type=float,default=0.05)
    p.add_argument("--global-complete-features", action="store_true",
                   help="For features-by-samples matrices, retain only features complete across ALL matrix samples before subsetting the analysis cohort.")
    return p.parse_args()

def read_any(path):
    return pd.read_csv(path,sep="\t" if str(path).lower().endswith(".tsv") else ",",low_memory=False)

def bh(p):
    p=np.asarray(p,float); out=np.full(len(p),np.nan)
    ok=np.isfinite(p); x=p[ok]
    o=np.argsort(x); q=x[o]*len(x)/np.arange(1,len(x)+1)
    q=np.minimum.accumulate(q[::-1])[::-1]; q=np.minimum(q,1)
    z=np.empty_like(q); z[o]=q; out[ok]=z
    return out

def residualize_matrix(Y,C):
    # Y features x samples, C samples x covariates
    Q,_=np.linalg.qr(C,mode="reduced")
    return Y - (Y @ Q) @ Q.T

def residualize_vector(y,C):
    Q,_=np.linalg.qr(C,mode="reduced")
    return y - Q @ (Q.T @ y)

def build_design(m,continuous,categorical):
    parts=[pd.DataFrame({"Intercept":np.ones(len(m))})]
    names=["Intercept"]
    for v in continuous:
        x=pd.to_numeric(m[v],errors="coerce").to_numpy(float)
        if not np.isfinite(x).all(): raise ValueError(f"Invalid continuous covariate {v}")
        sd=x.std(ddof=0)
        if sd==0: raise ValueError(f"No variation in continuous covariate {v}")
        z=(x-x.mean())/sd
        parts.append(pd.DataFrame({v+"__z":z})); names.append(v+"__z")
    for v in categorical:
        if m[v].isna().any(): raise ValueError(f"Missing categorical covariate {v}")
        dm=pd.get_dummies(m[v].astype("category"),prefix=v,drop_first=True,dtype=float).reset_index(drop=True)
        if dm.shape[1]==0: raise ValueError(f"Categorical covariate {v} has <2 levels")
        parts.append(dm); names.extend(dm.columns.tolist())
    C=pd.concat(parts,axis=1)
    A=C.to_numpy(float)
    if np.linalg.matrix_rank(A)!=A.shape[1]:
        raise ValueError("Reduced covariate design is not full rank.")
    return C

def load_expression(path,orientation,feature_id_col,samples,global_complete_features=False):
    d=read_any(path)
    if orientation=="features_by_samples":
        fid=feature_id_col or d.columns[0]
        if fid not in d:
            raise ValueError(f"Feature ID column missing: {fid}")

        all_sample_cols=[c for c in d.columns if c != fid]
        missing=set(samples)-set(all_sample_cols)
        if missing:
            raise ValueError(
                f"{len(missing)} manifest samples missing from expression: "
                f"{list(missing)[:10]}"
            )

        features=d[fid].astype(str).to_numpy()
        if pd.Series(features).duplicated().any():
            raise ValueError("Duplicate feature IDs")

        numeric_all=d[all_sample_cols].apply(pd.to_numeric,errors="coerce")

        if global_complete_features:
            keep=np.isfinite(numeric_all.to_numpy(float)).all(axis=1)
            features=features[keep]
            numeric_all=numeric_all.loc[keep].reset_index(drop=True)

        X=numeric_all[list(samples)].to_numpy(float)

    else:
        sid=d.columns[0]
        ids=d[sid].astype(str).str.strip()
        if ids.duplicated().any():
            raise ValueError("Duplicate expression sample IDs")
        idx=pd.Index(ids)
        missing=set(samples)-set(idx)
        if missing:
            raise ValueError(
                f"{len(missing)} manifest samples missing from expression: "
                f"{list(missing)[:10]}"
            )
        sub=d.set_index(sid).loc[list(samples)]
        features=np.array(sub.columns,dtype=str)
        X=sub.apply(pd.to_numeric,errors="coerce").to_numpy(float).T

    if not np.isfinite(X).all():
        raise ValueError(
            "Selected expression matrix contains missing/non-finite values; "
            "no imputation is allowed."
        )
    return features,X

def main():
    a=parse_args()
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    m=read_any(a.manifest)
    continuous=[x for x in a.continuous.split(",") if x]
    categorical=[x for x in a.categorical.split(",") if x]
    need=[a.sample_col,a.group_col]+continuous+categorical
    if a.block:
        need.append(a.block)
    miss=[x for x in need if x not in m]
    if miss: raise ValueError(f"Manifest missing: {miss}")
    if m[a.sample_col].duplicated().any(): raise ValueError("Duplicate samples in manifest")
    groups=set(m[a.group_col].astype(str))
    if groups!={a.group_a,a.group_b}: raise ValueError(f"Expected exactly groups A/B; observed {sorted(groups)}")
    samples=m[a.sample_col].astype(str).tolist()
    features,X=load_expression(
        a.expression,a.orientation,a.feature_id_col,samples,
        global_complete_features=a.global_complete_features
    )

    # Average expression ranks across participants for each feature.
    R=np.empty_like(X,dtype=float)
    for i in range(X.shape[0]):
        R[i]=stats.rankdata(X[i],method="average")

    C=build_design(m,continuous,categorical).to_numpy(float)

    # Covariate-adjusted residual degrees of freedom.
    design_rank=np.linalg.matrix_rank(C)
    residual_df=len(m)-design_rank

    if residual_df <= 0:
        raise ValueError(
            f"Non-positive residual degrees of freedom: "
            f"n={len(m)}, rank(C)={design_rank}"
        )
    E=residualize_matrix(R,C)
    y=(m[a.group_col].astype(str).to_numpy()==a.group_b).astype(float)
    u=residualize_vector(y,C)

    # Partial rank statistic.
    T=E @ u

    # Analytic permutation variance.
    # If --block is supplied, estimate variance within blocks.
    # Otherwise, treat the full residualized cohort as one block.
    if a.block:
        blocks=m[a.block].astype(str).to_numpy()
    else:
        blocks=np.repeat("__ALL__", len(m))

    var=np.zeros(X.shape[0],float)
    block_rows=[]
    for b in pd.unique(blocks):
        ix=np.where(blocks==b)[0]
        nb=len(ix)
        if nb<2:
            raise ValueError(f"Block {b!r} has n={nb}; cannot use blockwise variance.")
        Eb=E[:,ix]
        ub=u[ix]
        sse=((Eb-Eb.mean(axis=1,keepdims=True))**2).sum(axis=1)
        ssu=((ub-ub.mean())**2).sum()
        if a.block:
            variance_df=nb-1
        else:
            variance_df=residual_df

        var += sse*ssu/variance_df
        block_rows.append({"block":b,"n":nb,"u_ss":ssu})
    z=np.divide(T,np.sqrt(var),out=np.full_like(T,np.nan),where=var>0)
    p=erfc(np.abs(z)/np.sqrt(2.0))
    q=bh(p)

    # Raw rank-biserial-like effect for intuitive direction.
    gA=m[a.group_col].astype(str).to_numpy()==a.group_a
    gB=~gA
    mean_rank_A=R[:,gA].mean(axis=1)
    mean_rank_B=R[:,gB].mean(axis=1)
    raw_rank_diff=mean_rank_B-mean_rank_A

    res=pd.DataFrame({
        "feature_id":features,
        "contrast":a.contrast_name,
        "n":len(m),"n_A":gA.sum(),"n_B":gB.sum(),
        "raw_mean_rank_B_minus_A":raw_rank_diff,
        "partial_rank_statistic":T,
        "adjusted_Regular_AREA_Z":z,
        "pvalue":p,"padj":q,
        "fdr_significant":q<a.fdr,
    })
    res["direction"]=np.where(res["adjusted_Regular_AREA_Z"]>0,
                              f"{a.group_b}_associated_higher_rank",
                              f"{a.group_b}_associated_lower_rank")
    res=res.sort_values("pvalue",na_position="last")
    res.to_csv(out/f"{a.contrast_name}_Regular_AREA_all_results.csv",index=False)
    res[res.fdr_significant].to_csv(out/f"{a.contrast_name}_Regular_AREA_FDR05.csv",index=False)
    pd.DataFrame(block_rows).to_csv(out/"block_diagnostics.csv",index=False)

    manifest={
        "contrast":a.contrast_name,"group_A":a.group_a,"group_B":a.group_b,
        "n":len(m),"n_A":int(gA.sum()),"n_B":int(gB.sum()),
        "n_features":int(len(features)),"continuous_covariates":continuous,
        "categorical_covariates":categorical,"block":a.block,
        "design_rank":int(design_rank),
        "residual_df":int(residual_df),
        "variance_df":int(
            residual_df if not a.block else -1
        ),
        "orientation":a.orientation,"fdr":a.fdr,
        "n_FDR05":int((q<a.fdr).sum()),
        "inference":"FWL partial rank + residual-df-corrected analytic permutation variance + Gaussian two-sided p + BH"
    }
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("GENERALIZED REGULAR AREA")
    print("="*72)
    print(f"Contrast: {a.contrast_name}")
    print(f"n={len(m)} | {a.group_a}={gA.sum()} | {a.group_b}={gB.sum()}")
    print(f"Features={len(features)} | FDR<0.05={(q<a.fdr).sum()}")
    if a.block:
        print(f"Block={a.block} | blocks={m[a.block].nunique()}")
    else:
        print("Block=None | analytic variance uses full residualized cohort")
    print(f"Outdir: {out}")

if __name__=="__main__":
    main()
