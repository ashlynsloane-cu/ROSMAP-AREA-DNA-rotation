#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
CONTRASTS={'cogdx_extreme':'AD vs NCI','cogdx_early':'MCI vs NCI','cogdx_late':'AD vs MCI','braak_anchor':'Braak 5–6\nvs 0–2','braak_early':'Braak 3–4\nvs 0–2','braak_late':'Braak 5–6\nvs 3–4','cerad':'CERAD\n1–2 vs 3–4'}
# Externally motivated shortlist, not a comprehensive causal reference set.
EVIDENCE={
'APP':('Familial AD causality','https://pubmed.ncbi.nlm.nih.gov/1671712/','APP feature does not specifically quantify amyloid-beta'),
'PSEN1':('Familial AD causality','https://pubmed.ncbi.nlm.nih.gov/7596406/','Protein abundance need not change for a causal variant'),
'PSEN2':('Familial AD causality','https://pubmed.ncbi.nlm.nih.gov/7651536/','Excluded from recovery denominator if not measured'),
'APOE':('Human genetic modifier','https://www.nature.com/articles/s41591-019-0611-3','Total APOE feature does not establish APOE genotype'),
'SORL1':('Strong rare-variant evidence','https://www.nature.com/articles/s41588-022-01208-7','Variant-specific risk; no abundance-causality inference'),
'TREM2':('Protein-altering risk variants','https://www.nejm.org/doi/full/10.1056/NEJMoa1211851','Excluded if not measured'),
'ABCA7':('Loss-of-function risk variants','https://www.nature.com/articles/ng.3246','Excluded if not measured'),
'PLCG2':('Protective coding variant','https://www.nature.com/articles/ng.3916','Excluded if not measured'),
'BACE1':('Biochemical mechanism','https://pubmed.ncbi.nlm.nih.gov/10591214/','BACE inhibitor trials did not establish clinical benefit; https://www.nejm.org/doi/full/10.1056/NEJMoa1812840'),
'MAPT':('Mouse perturbation evidence','https://pubmed.ncbi.nlm.nih.gov/17478722/','Annotated tau features are not phospho-tau or aggregation measurements')}
def main():
 root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=root/'all_tested_protein_evidence.csv');p.add_argument('--output-root',type=Path,default=root);a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.input);audit=[];long=[]
 for gene,(evidence,url,limit) in EVIDENCE.items():
  sub=d[d.gene==gene];ar=sub[[f'AREA_q__{c}' for c in CONTRASTS]].lt(.05).any().any();lm=sub[[f'limma_q__{c}' for c in CONTRASTS]].lt(.05).any().any()
  audit.append(dict(gene=gene,eligible=not sub.empty,n_tested_features=len(sub),AREA_recovered=bool(ar) if len(sub) else None,limma_recovered=bool(lm) if len(sub) else None,evidence=evidence,source=url,interpretation_limit=limit))
  for _,r in sub.iterrows():
   for method in ['AREA','limma']:
    for c in CONTRASTS:long.append(dict(gene=gene,protein_id=r.protein_id,method=method,contrast=c,q=r[f'{method}_q__{c}'],significant=r[f'{method}_q__{c}']<.05,evidence=evidence,source=url))
 audit=pd.DataFrame(audit);v=pd.DataFrame(long);audit.to_csv(a.output_root/'protein_eligibility_and_evidence_audit.csv',index=False);v.to_csv(a.output_root/'all_eligible_protein_feature_results.csv',index=False)
 # Feature-level selection: any significant AREA contrast, then show both methods for that same feature.
 selected=v[(v.method=='AREA')&v.significant].protein_id.drop_duplicates().tolist()
 assert selected==['APP|P05067','APOE|P02649','BACE1|P56817','MAPT|P10636-6']
 shown=v[v.protein_id.isin(selected)];shown.to_csv(a.output_root/'displayed_protein_data.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42});fig=plt.figure(figsize=(170/25.4,140/25.4));ax=fig.add_axes([.24,.28,.72,.45]);colors={'AREA':'#596F86','limma':'#9BAFC1'}
 labels=[]
 for i,feature in enumerate(selected):
  gene=feature.split('|')[0]
  for j,method in enumerate(['AREA','limma']):
   y=2*i+j;labels.append(f'{gene if gene!="MAPT" else "Tau · P10636-6"} | {method}')
   for x,c in enumerate(CONTRASTS):
    r=shown[(shown.protein_id==feature)&(shown.method==method)&(shown.contrast==c)].iloc[0];q=r.q;ax.add_patch(Rectangle((x-.47,y-.43),.94,.86,facecolor=colors[method] if q<.05 else '#D8DEE3',edgecolor='white',lw=.6));ax.text(x,y,f'{q:.1e}' if q<.001 else f'{q:.3f}',ha='center',va='center',fontsize=6,color='white' if q<.05 and method=='AREA' else '#253444')
 ax.set_xlim(-.55,6.55);ax.set_ylim(7.6,-.6);ax.set_xticks(range(7),list(CONTRASTS.values()),fontsize=6.5);ax.set_yticks(range(8),labels,fontsize=7);ax.tick_params(length=0,pad=6);[s.set_visible(False) for s in ax.spines.values()]
 fig.text(.5,.955,'AREA recovers measured AD-relevant proteins',ha='center',fontsize=11,weight='bold');fig.text(.5,.912,'Recovered examples with distinct genetic and mechanistic evidence',ha='center',fontsize=8)
 fig.legend(handles=[Patch(facecolor=colors['AREA'],label='AREA: q < 0.05'),Patch(facecolor=colors['limma'],label='limma: q < 0.05'),Patch(facecolor='#D8DEE3',label='Not significant')],loc='upper center',bbox_to_anchor=(.5,.88),ncol=3,frameon=False,fontsize=7)
 fig.text(.24,.79,'APP: familial AD causality · APOE: human genetic modifier\nBACE1: biochemical mechanism · Tau: mouse perturbation evidence',fontsize=7,color='#445D74',linespacing=1.6)
 fig.text(.08,.16,'Cells: original proteome-wide BH-adjusted q values. Only AREA-recovered features shown.\nUnmeasured proteins excluded from eligibility; full measured-feature audit supplied.\nTau row is the P10636-6 annotated feature, not a phospho-tau measurement.\nThis is a selected-example display, not an enrichment or method-superiority test.',fontsize=7,va='top',linespacing=1.7)
 for ext in ['png','pdf']:fig.savefig(a.output_root/f'AREA_measured_AD_protein_recovery.{ext}',dpi=400)
 plt.close(fig);print(audit[['gene','eligible','n_tested_features','AREA_recovered','limma_recovered']].to_string(index=False))
if __name__=='__main__':main()
