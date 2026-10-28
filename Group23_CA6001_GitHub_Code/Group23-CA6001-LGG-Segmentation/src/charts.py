"""Read saved results and build figures. No training calls or raw MRI access."""
from pathlib import Path
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#176B87','#E27C45','#699957','#8566A9']

def analyze(root, evaluation_split="test", planned=None):
    if evaluation_split not in ["test", "val"]:raise ValueError("Expected test or val")
    marker="EVALUATION_COMPLETE.json" if evaluation_split=="test" else "VALIDATION_CHECK_COMPLETE.json"
    prediction_folder='predictions' if evaluation_split=='test' else 'validation_predictions'
    root=Path(root);manifest=pd.read_csv(root/'data_split.csv');runs={}
    selection=root/'analysis/selected_models.json'
    allowed={e['name'] if isinstance(e,dict) else e for e in planned} if planned is not None else (set(json.loads(selection.read_text())) if selection.exists() and evaluation_split=='test' else None)
    for p in sorted((root/'experiments').glob('*')):
        if (p/marker).exists() and (allowed is None or p.name in allowed):
            runs[p.name]=dict(path=p,slices=pd.read_csv(p/f'{evaluation_split}_slices.csv'),patients=pd.read_csv(p/f'{evaluation_split}_patients.csv'),
                history=pd.read_csv(p/'history.csv'),meta=json.loads((p/'experiment.json').read_text()),metrics=json.loads((p/f'{evaluation_split}_metrics.json').read_text()))
    if not runs:raise FileNotFoundError('No completed evaluation. This command does not train.')
    if allowed is not None and set(runs)!=allowed:raise ValueError('A selected model has no completed evaluation.')
    expected=manifest[manifest.split==evaluation_split]
    reference=next(iter(runs.values()))['meta']['spec']
    keys=['seed','image_size','batch_size','epochs','learning_rate','threshold','split_hash','core_hash']
    for name,run in runs.items():
        spec=run['meta']['spec']
        if any(spec.get(k)!=reference.get(k) for k in keys):raise ValueError('Incompatible experiment settings: '+name)
        slices=run['slices'];patients=run['patients']
        if len(slices)!=len(expected) or set(zip(slices.patient,slices.slice_id))!=set(zip(expected.patient,expected.slice_id)):raise ValueError('Evaluation slice identifiers differ from manifest: '+name)
        if patients.patient.duplicated().any() or set(patients.patient)!=set(expected.patient):raise ValueError('Evaluation patient identifiers differ from manifest: '+name)
        if evaluation_split=='test':
            from result_guards import test_cache_complete
            if not test_cache_complete(spec,manifest,spec['experiment'],root,signature_value=run['meta']['signature']):raise ValueError('Incomplete or stale test evidence: '+name)
    out=root/('analysis' if evaluation_split=='test' else 'validation_preview');tables=out/'tables';figdir=out/'figures';tables.mkdir(parents=True,exist_ok=True);figdir.mkdir(exist_ok=True);figures=[]
    if evaluation_split=='test':selection.write_text(json.dumps(list(runs),indent=2),encoding='utf-8')
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig);figures.append(name+'.png')
    summary=manifest.groupby('split').agg(patients=('patient','nunique'),slices=('image','size'),positive=('positive','sum'))
    summary['empty']=summary.slices-summary.positive;summary.to_csv(tables/'dataset_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));summary[['positive','empty']].plot.bar(stacked=True,ax=axes[0],color=COLORS[:2]);summary.patients.plot.bar(ax=axes[1],color=COLORS[0])
    axes[0].set(ylabel='Slices',title='Available images');axes[1].set(ylabel='Patients',title='Patient-disjoint split');save(fig,'01_dataset_split')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].hist(manifest.loc[manifest.positive,'mask_fraction']*100,bins=25,color=COLORS[0]);axes[1].hist(manifest.groupby('patient').size(),bins=15,color=COLORS[1])
    axes[0].set(xlabel='Annotated area (% of image)',ylabel='Positive slices');axes[1].set(xlabel='Slices per patient',ylabel='Patients');save(fig,'02_data_distributions')
    q=manifest[(manifest.split=='train')&manifest.positive].mask_fraction.quantile([1/3,2/3]);q1,q2=q.tolist()
    rows=[];all_slices=[];all_patients=[];size_rows=[];empty_rows=[];ci_rows=[];settings=[];confusion=[]
    refspec=next(iter(runs.values()))['meta']['spec']
    for name,run in runs.items():
        slices=run['slices'].copy();patients=run['patients'].copy();slices['model']=name;patients['model']=name
        slices['area_fraction']=slices.true_pixels/slices.total_pixels
        slices['area_group']=np.select([slices.true_pixels==0,slices.area_fraction<=q1,slices.area_fraction<=q2],['empty','small','medium'],default='large')
        slices['false_positive_pixels']=slices.pred_pixels-slices.intersection;slices['false_negative_pixels']=slices.true_pixels-slices.intersection
        slices['area_error']=slices.pred_pixels-slices.true_pixels;all_slices.append(slices);all_patients.append(patients)
        m=run['metrics'];best=json.loads((run['path']/'best_epoch.json').read_text());spec=run['meta']['spec']
        rows.append(dict(model=name,loss=spec['experiment']['loss'],use_skips=spec['experiment']['use_skips'],patient_dice=m['dice'],patient_iou=m['iou'],best_epoch=best['epoch'],best_val_dice=best['val_dice'],parameters=run['meta']['parameters'],training_and_validation_minutes=run['history'].seconds.sum()/60,evaluation_seconds=m['evaluation_seconds'],patients=len(patients),slices=len(slices)))
        settings.append(dict(model=name,differences_from_first=';'.join(k for k in refspec if k!='experiment' and refspec[k]!=spec.get(k))))
        for group,part in slices.groupby('area_group'):size_rows.append(dict(model=name,group=group,slices=len(part),mean_slice_dice=part.dice.mean(),mean_slice_iou=part.iou.mean()))
        empty=slices[slices.true_pixels==0];empty_rows.append(dict(model=name,empty_slices=len(empty),false_positive_slices=int((empty.pred_pixels>0).sum()),false_positive_slice_rate=float((empty.pred_pixels>0).mean()) if len(empty) else np.nan))
        vals=patients.dice.to_numpy();draws=np.random.default_rng(42).choice(vals,size=(2000,len(vals)),replace=True).mean(axis=1);low,high=np.quantile(draws,[.025,.975])
        ci_rows.append(dict(model=name,patients=len(vals),mean_dice=vals.mean(),bootstrap_low=low,bootstrap_high=high))
        tp=int(slices.intersection.sum());fp=int(slices.false_positive_pixels.sum());fn=int(slices.false_negative_pixels.sum());tn=int(slices.total_pixels.sum()-tp-fp-fn)
        confusion.append(dict(model=name,TP=tp,FP=fp,FN=fn,TN=tn))
    comparison=pd.DataFrame(rows);comparison.to_csv(tables/'model_comparison.csv',index=False)
    pd.concat(all_slices).to_csv(tables/'all_slice_metrics.csv',index=False);pd.concat(all_patients).to_csv(tables/'all_patient_metrics.csv',index=False)
    for name,records in [('comparison_settings',settings),('lesion_size',size_rows),('empty_mask_errors',empty_rows),('patient_bootstrap',ci_rows),('pixel_confusion',confusion)]:pd.DataFrame(records).to_csv(tables/(name+'.csv'),index=False)
    (tables/'size_thresholds.json').write_text(json.dumps(dict(source='training positive mask fractions only',small_max=q1,medium_max=q2),indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    for i,(name,run) in enumerate(runs.items()):
        h=run['history']
        for ax,col in zip(axes,['train_loss','val_loss','val_dice']):ax.plot(h.epoch,h[col],marker='o' if len(h)<2 else None,label=name,color=COLORS[i%4]);ax.set(xlabel='Epoch',ylabel=col);ax.legend(fontsize=7)
    axes[0].set(ylabel='Mean training loss',title='Training')
    axes[1].set(ylabel='Mean validation loss',title='Validation')
    axes[2].set(ylabel='Patient-macro Dice',title='Validation checkpoint selection',ylim=(0,1))
    epochs=sorted({int(epoch) for run in runs.values() for epoch in run['history'].epoch})
    if len(epochs)<=10:
        for ax in axes:ax.set_xticks(epochs)
    save(fig,'03_learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4));comparison.set_index('model')[['patient_dice','patient_iou']].plot.bar(ax=axes[0],color=COLORS[:2]);axes[0].set(ylim=(0,1),title=evaluation_split.title()+' patient macro averages')
    axes[1].bar(comparison.model,comparison.parameters/1e6,color=COLORS[0]);axes[1].set(ylabel='Parameters (millions)');[ax.tick_params(axis='x',rotation=15) for ax in axes];save(fig,'04_model_comparison')
    names=list(runs);fig,ax=plt.subplots(figsize=(8,4));values=[r['patients'].dice for r in runs.values()];ax.boxplot(values,**({'tick_labels':names} if tuple(int(v) for v in matplotlib.__version__.split('.')[:2])>=(3,9) else {'labels':names}),showfliers=False)
    for i,v in enumerate(values):ax.scatter(i+1+np.random.default_rng(42).uniform(-.06,.06,len(v)),v,s=18,color=COLORS[i%4])
    ax.set(ylim=(0,1),ylabel='Patient Dice',title='Each point is one '+evaluation_split+' patient');save(fig,'05_patient_scores')
    fig,ax=plt.subplots(figsize=(8,4))
    for i,s in enumerate(all_slices):ax.hist(s.loc[s.true_pixels>0,'dice'],bins=np.linspace(0,1,11),alpha=.45,label=names[i],color=COLORS[i%4])
    ax.set(xlabel='Slice Dice',ylabel='Positive slices');ax.legend();save(fig,'06_slice_scores')
    st=pd.DataFrame(size_rows);fig,ax=plt.subplots(figsize=(8,4));st[st.group!='empty'].pivot(index='group',columns='model',values='mean_slice_dice').reindex(['small','medium','large']).plot.bar(ax=ax,color=COLORS)
    ax.set(ylim=(0,1),ylabel='Mean slice Dice',title='Area groups defined on training masks');save(fig,'07_lesion_size')
    e=pd.DataFrame(empty_rows);fig,ax=plt.subplots(figsize=(8,4));ax.bar(e.model,e.false_positive_slice_rate,color=COLORS[1]);ax.set(ylim=(0,1),ylabel='Fraction with any predicted foreground',title='False positives on empty slices');save(fig,'08_empty_masks')
    fig,ax=plt.subplots(figsize=(7,5));largest=1
    for i,s in enumerate(all_slices):
        ax.scatter(s.true_pixels,s.pred_pixels,s=12,alpha=.5,label=names[i],color=COLORS[i%4]);largest=max(largest,s.true_pixels.max(),s.pred_pixels.max())
    ax.plot([0,largest],[0,largest],'k--');ax.set(xlabel='True foreground pixels',ylabel='Predicted foreground pixels');ax.legend();save(fig,'09_area_agreement')
    if len(names)>1:
        pair=runs[names[0]]['patients'][['patient','dice']].merge(runs[names[1]]['patients'][['patient','dice']],on='patient',suffixes=('_a','_b'),validate='one_to_one');pair['difference_b_minus_a']=pair.dice_b-pair.dice_a;pair.to_csv(tables/'paired_patients.csv',index=False)
        fig,ax=plt.subplots(figsize=(5,5));ax.scatter(pair.dice_a,pair.dice_b);ax.plot([0,1],[0,1],'k--');ax.set(xlabel=names[0]+' Dice',ylabel=names[1]+' Dice',xlim=(0,1),ylim=(0,1));save(fig,'10_paired_patients')
    ref='unet' if 'unet' in runs else names[0];s=runs[ref]['slices'];pos=s[s.true_pixels>0].sort_values(['dice','patient','slice_id']);selected=[]
    for category,part in [('low',pos.head(2)),('middle',pos.iloc[max(0,len(pos)//2-1):len(pos)//2+1]),('high',pos.tail(2)),('empty_high_fp',s[s.true_pixels==0].sort_values('pred_pixels',ascending=False).head(2))]:
        for _,row in part.iterrows():selected.append(dict(patient=row.patient,slice_id=int(row.slice_id),category=category,reference_model=ref))
    cases=pd.DataFrame(selected).drop_duplicates(['patient','slice_id']);cases.to_csv(tables/'case_selection.csv',index=False)
    for page,start in enumerate(range(0,len(cases),4),1):
        part=cases.iloc[start:start+4];fig,axes=plt.subplots(len(part),2+len(names),figsize=(3*(2+len(names)),3*len(part)),squeeze=False)
        for i,(_,case) in enumerate(part.iterrows()):
            stem=f'{case.patient}_{case.slice_id}';base=runs[ref]['path']/prediction_folder;panels=[(base/(stem+'_image.png'),'MRI channel 1'),(base/(stem+'_true.png'),'Ground truth')]
            for name in names:
                df=runs[name]['slices'];row=df[(df.patient==case.patient)&(df.slice_id==case.slice_id)].iloc[0]
                panels.append((runs[name]['path']/prediction_folder/(stem+'_pred.png'),f'{name}\nDice {row.dice:.3f}'))
            for j,(path,title) in enumerate(panels):axes[i,j].imshow(Image.open(path),cmap='gray',vmin=0,vmax=255);axes[i,j].set_title(title,fontsize=9);axes[i,j].axis('off')
            axes[i,0].set_title(f'{case.category}: {case.patient}\nslice {case.slice_id}',fontsize=7)
        save(fig,f'11_cases_{page}')
    part=cases.head(4);fig,axes=plt.subplots(1,len(part),figsize=(4*len(part),4),squeeze=False)
    for ax,(_,case) in zip(axes[0],part.iterrows()):
        stem=f'{case.patient}_{case.slice_id}';base=runs[ref]['path']/prediction_folder;im=np.asarray(Image.open(base/(stem+'_image.png')))/255;truth=np.asarray(Image.open(base/(stem+'_true.png')))>0;pred=np.asarray(Image.open(base/(stem+'_pred.png')))>0
        rgb=np.repeat(im[...,None],3,axis=2);rgb[truth]=.55*rgb[truth]+.45*np.array([0,1,0]);rgb[pred]=.55*rgb[pred]+.45*np.array([1,0,1]);ax.imshow(rgb);ax.axis('off');ax.set_title(f'{case.category}, slice {case.slice_id}',fontsize=9)
    fig.suptitle(ref+': ground truth green; prediction magenta');save(fig,'12_overlays')
    notes='''Saved experiment analysis. No training performed. Main metrics aggregate pixels within each patient, then average patients. Both-empty masks score 1. The fixed probability threshold is read from the signed experiment settings below. Slice scores from one patient are correlated. Lesion-size group thresholds use only training masks. Any predicted foreground counts as an empty-slice false positive. Bootstrap intervals resample patients; small test sets give unreliable intervals. Comparison settings must match for controlled comparisons. Parameter counts differ between skip/no-skip architectures. Evaluation time includes image exports, not pure model latency. Cases include low/middle/high positive-slice scores and empty slices with false positives, selected using the reference model. Loss magnitudes from different objectives (BCE-only versus BCE+Dice) are not directly comparable; compare Dice/IoU. No clinical performance or significant improvement is assumed. Any new tuning after inspecting test results should be labelled exploratory.'''
    notes += '\nProbability threshold: '+str(refspec['threshold'])+'\nTraining_and_validation_minutes includes validation.\n'
    (out/'RESULTS_NOTES.txt').write_text(notes,encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(n[:-4])}</h2><img src="figures/{n}" loading="lazy"></section>' for n in figures)
    banner='<p><b>Validation-only short-run check. Not final test results.</b></p>' if evaluation_split=='val' else ''
    doc='<!doctype html><meta charset="utf-8"><title>LGG saved results</title><style>body{font:16px system-ui;background:#f4f7fa;color:#203040;max-width:1200px;margin:30px auto;padding:0 20px}section{background:white;border-radius:12px;padding:24px;margin:24px 0}img{max-width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>LGG MRI segmentation — saved results</h1><p>Actual saved predictions. Read RESULTS_NOTES.txt for metric definitions and limits.</p><section style="overflow:auto">'+comparison.round(4).to_html(index=False,border=0)+'</section>'+cards
    doc=doc.replace('<h1>',banner+'<h1>',1)
    (out/'OPEN_RESULTS.html').write_text(doc,encoding='utf-8');print('Analysis ready:',out/'OPEN_RESULTS.html',flush=True)
    return comparison
