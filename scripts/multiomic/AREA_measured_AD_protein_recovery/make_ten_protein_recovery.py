#!/usr/bin/env python3
"""Ten measured AD-related proteins; paired method cells with binary significance."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from make_measured_protein_recovery import CONTRASTS,EVIDENCE
GENES=['APP','PSEN1','APOE','SORL1','ADAM10','BACE1','MAPT','CLU','BIN1','PICALM']
ADDED={'ADAM10':('Rare variants + experiments','https://pubmed.ncbi.nlm.nih.gov/24055016/','Experimental variant evidence; not equivalent to established familial AD genes'), 'CLU':('GWAS locus','https://www.nature.com/articles/ng.439','Locus association alone does not establish protein causality'),'BIN1':('GWAS locus','https://pubmed.ncbi.nlm.nih.gov/20460622/','Locus association alone does not establish protein causality'),'PICALM':('GWAS locus','https://www.nature.com/articles/ng.440','Locus association alone does not establish protein causality')}
SHORT={'APP':'Familial causality','PSEN1':'Familial causality','APOE':'Genetic modifier','SORL1':'Rare variants','ADAM10':'Variants + experiments','BACE1':'Biochemical mechanism','MAPT':'Mouse perturbation','CLU':'GWAS locus','BIN1':'GWAS locus','PICALM':'GWAS locus'}
def main():
 root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=root/'all_tested_protein_evidence.csv');p.add_argument('--output-root',type=Path,default=root);a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.input);ev={**EVIDENCE,**ADDED};rows=[];audit=[];cells=[]
 for gene in GENES:
  sub=d[d.gene==gene];assert len(sub)>0,gene
  e,url,limit=ev[gene];audit.append(dict(gene=gene,evidence=e,source=url,interpretation_limit=limit,n_features=len(sub),feature_ids=';'.join(sub.protein_id)))
  for method in ['AREA','limma']:
   for c in CONTRASTS:
    q=sub[f'{method}_q__{c}'];sig=sub.loc[q<.05,'protein_id'].tolist();cells.append(dict(gene=gene,method=method,contrast=c,significant=bool(sig),significant_feature_ids=';'.join(sig)))
    for _,r in sub.iterrows():rows.append(dict(gene=gene,protein_id=r.protein_id,method=method,contrast=c,q=r[f'{method}_q__{c}'],significant=r[f'{method}_q__{c}']<.05))
 pd.DataFrame(rows).to_csv(a.output_root/'ten_protein_feature_results.csv',index=False);pd.DataFrame(audit).to_csv(a.output_root/'ten_protein_sources.csv',index=False);cells=pd.DataFrame(cells);cells.to_csv(a.output_root/'ten_protein_displayed_cells.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42});fig=plt.figure(figsize=(170/25.4,140/25.4));ax=fig.add_axes([.38,.24,.59,.55]);colors={'AREA':'#596F86','limma':'#9BAFC1'}
 for y,g in enumerate(GENES):
  for x,c in enumerate(CONTRASTS):
   for j,m in enumerate(['AREA','limma']):
    xpos=x+(-.22 if j==0 else .22);r=cells[(cells.gene==g)&(cells.contrast==c)&(cells.method==m)].iloc[0]
    ax.add_patch(Rectangle((xpos-.19,y-.39),.38,.78,facecolor=colors[m] if r.significant else '#E4E8EC',edgecolor='white',lw=.4))
  ax.text(-.065,y,'Tau / MAPT' if g=='MAPT' else g,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=8,weight='bold',clip_on=False)
  ax.text(-.30,y,SHORT[g],transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.2,color='#596F86',clip_on=False)
 ax.set_xlim(-.5,6.5);ax.set_ylim(9.6,-.6);ax.set_xticks(range(7),list(CONTRASTS.values()),fontsize=6);ax.set_yticks([]);ax.tick_params(length=0,pad=6);[s.set_visible(False) for s in ax.spines.values()]
 fig.text(.5,.955,'Recovery of ten measured AD-related proteins',ha='center',fontsize=11,weight='bold');fig.text(.5,.914,'Literature-selected examples spanning genetic and mechanistic evidence',ha='center',fontsize=8)
 fig.legend(handles=[Patch(facecolor=colors['AREA'],label='AREA (left)'),Patch(facecolor=colors['limma'],label='limma (right)')],loc='upper center',bbox_to_anchor=(.5,.89),ncol=2,frameon=False,fontsize=8)
 fig.text(.5,.825,'Blue: q < 0.05     Light gray: not significant',ha='center',fontsize=7.5)
 fig.text(.06,.135,'All ten proteins occur in the tested TMT set; absent proteins are excluded.\nEach cell: at least one annotated feature significant at original proteome-wide BH q < 0.05.\nTau and BIN1 have multiple features; exact feature results and literature sources are supplied.\nEvidence types differ: these are not ten equally established causal proteins.',fontsize=7,va='top',linespacing=1.65)
 for ext in ['png','pdf']:fig.savefig(a.output_root/f'AREA_measured_AD_protein_recovery.{ext}',dpi=400)
 plt.close(fig)
if __name__=='__main__':main()
