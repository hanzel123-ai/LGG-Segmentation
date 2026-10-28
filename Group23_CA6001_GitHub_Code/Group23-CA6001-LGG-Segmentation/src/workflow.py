"""Small helpers used by the notebooks; no training happens on import."""
from pathlib import Path
import json,sys,zipfile,urllib.request,shutil,time,platform
from core import *
from result_guards import (test_cache_complete, training_complete,
                           migrate_validation_predictions, isolate_prediction_cache)
PROJECT=Path(__file__).resolve().parent.parent

def settings():
    return dict(seed=42,image_size=256,base_channels=32,batch_size=4,epochs=5,learning_rate=.0001,threshold=.5,
        device='auto',cpu_threads=4,cpu_bfloat16=False,gpu_amp=True)

def experiments(include_ablation=False):
    result=[dict(name='unet',use_skips=True,loss='bce_dice'),dict(name='no_skip',use_skips=False,loss='bce_dice')]
    if include_ablation:result.append(dict(name='unet_bce_ablation',use_skips=True,loss='bce'))
    return result

def resolve_data(data_root='',download=False):
    if data_root:
        path=Path(data_root).expanduser()
        if path.is_dir():return path.resolve()
        raise FileNotFoundError('DATA_ROOT does not exist. Correct the path or leave it blank to select a folder.')
    remembered=PROJECT/'local_paths.json'
    if remembered.exists():
        path=Path(json.loads(remembered.read_text())['data_root'])
        if path.is_dir():return path
    if download:
        folder=PROJECT/'data';folder.mkdir(exist_ok=True);archive=folder/'lgg.zip'
        if not archive.exists():
            print('Downloading public MRI archive (~714 MB).',flush=True)
            request=urllib.request.Request('https://www.kaggle.com/api/v1/datasets/download/mateuszbuda/lgg-mri-segmentation',headers={'User-Agent':'LGG-course-project'})
            part=folder/'lgg.zip.part'
            with urllib.request.urlopen(request,timeout=120) as response,part.open('wb') as f:shutil.copyfileobj(response,f)
            if not zipfile.is_zipfile(part):raise RuntimeError('Download was not a ZIP. Download manually from Kaggle and set DATA_ROOT.')
            part.replace(archive)
        path=(folder/'extracted').resolve()
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                dest=(path/item.filename).resolve()
                if path not in dest.parents and dest!=path:raise ValueError('Unsafe ZIP entry.')
            z.extractall(path)
    else:
        value=''
        try:
            import tkinter as tk
            from tkinter import filedialog
            window=tk.Tk();window.withdraw();value=filedialog.askdirectory(title='Choose extracted LGG MRI dataset');window.destroy()
        except Exception:pass
        if not value:value=input('Path to extracted MRI folder: ').strip().strip('"')
        if not value:raise RuntimeError('No folder selected. Set DATA_ROOT and run this cell again.')
        path=Path(value).expanduser().resolve()
    atomic_json(remembered,dict(data_root=str(path)));return path

def prepare(data_root,config,out):
    masks=list(Path(data_root).rglob('*_mask.tif'));roots={}
    if not masks:raise FileNotFoundError('No *_mask.tif found. Extract the dataset first.')
    for p in masks:roots[p.parent.parent]=roots.get(p.parent.parent,0)+1
    chosen=max(roots,key=lambda p:(roots[p],len(p.parts)));print('Using one dataset copy:',chosen,flush=True)
    frame=build_manifest(chosen);areas=[];ih=[];mh=[]
    for row in frame.itertuples():
        y=tifffile.imread(row.mask);areas.append(float((y>0).mean()))
        ih.append(hashlib.sha256(Path(row.image).read_bytes()).hexdigest());mh.append(hashlib.sha256(Path(row.mask).read_bytes()).hexdigest())
    frame['mask_fraction']=areas;frame['image_sha256']=ih;frame['mask_sha256']=mh
    frame=assign_splits(frame,config['seed']).sort_values(['patient','slice_id']).reset_index(drop=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);path=out/'data_split.csv'
    if path.exists():
        old=pd.read_csv(path);cols=['patient','slice_id','split','image_sha256','mask_sha256']
        if not old[cols].equals(frame[cols]):raise ValueError('Saved split/data content differs. Do not overwrite it; use a new output directory.')
    frame.to_csv(path,index=False)
    atomic_json(out/'settings.json',config)
    for run in (out/'experiments').glob('*'):
        if run.is_dir():migrate_validation_predictions(frame,run)
    print(frame.groupby('split').agg(patients=('patient','nunique'),slices=('image','size'),positive_slices=('positive','sum')),flush=True)
    return frame

def status(config,planned,out):
    out=Path(out);rows=[]
    for e in planned:
        run=out/'experiments'/e['name'];h=run/'history.csv'
        completed=int(pd.read_csv(h).epoch.max()) if h.exists() else 0
        rows.append(dict(model=e['name'],saved_epochs=completed,target_epochs=config['epochs'],training_complete=(run/'TRAINING_COMPLETE.json').exists(),evaluation_complete=(run/'EVALUATION_COMPLETE.json').exists()))
    return pd.DataFrame(rows)

def all_evaluated(config,planned,out):
    out=Path(out)
    if not (out/'data_split.csv').exists():return False
    frame=pd.read_csv(out/'data_split.csv')
    for e in planned:
        run=out/'experiments'/e['name'];_,sig=signature(config,e,frame)
        if not test_cache_complete(config,frame,e,out,signature_value=sig):return False
    return True

def local_check(config,frame,out):
    """One real training batch, isolated from experiment weights. Runs before the long training."""
    device=set_device(config);x,y,_,_=next(iter(loader(frame,config,'train')));trials=[]
    choices=[False,True] if device.type=='cpu' else [False]
    for amp in choices:
        test_config=dict(config);test_config['cpu_bfloat16']=amp
        model=None
        try:
            seed_everything(config['seed']);model=UNet(config['base_channels'],True).to(device).to(memory_format=torch.channels_last)
            optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
            scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['gpu_amp'])
            start=time.time()
            xx=x.to(device,memory_format=torch.channels_last);yy=y.to(device)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with autocast(test_config,device):logits=model(xx).float()
                if logits.shape!=yy.shape:raise RuntimeError('Output/mask shape mismatch.')
                loss=criterion(logits,yy,'bce_dice')
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss in local check.')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite gradients in local check.')
                scaler.step(optimizer);scaler.update()
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=(time.time()-start)/2;trials.append(dict(cpu_bfloat16=amp,seconds_per_batch=elapsed,passed=True));print('Local check passed:',str(device),'CPU bfloat16:',amp,'seconds/batch:',round(elapsed,2),flush=True)
        except Exception as exc:
            trials.append(dict(cpu_bfloat16=amp,passed=False,error=str(exc)));print('Check failed for this compute mode:',exc,flush=True)
        finally:
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
    valid=[t for t in trials if t['passed']]
    if not valid:
        if device.type=='cuda':
            print('GPU check failed; retrying on CPU without changing model/batch/split.',flush=True);config['device']='cpu';return local_check(config,frame,out)
        raise RuntimeError('CPU check failed. Stop here and send the error; no experiment checkpoint was modified.')
    best=min(valid,key=lambda t:t['seconds_per_batch']);config['cpu_bfloat16']=best['cpu_bfloat16'] if device.type=='cpu' else False
    record=dict(device=str(device),torch=str(torch.__version__),python=platform.python_version(),trials=trials,selected_cpu_bfloat16=config['cpu_bfloat16'])
    atomic_json(Path(out)/'local_check.json',record);atomic_json(Path(out)/'settings.json',config)
    return record

def finish(config,frame,planned,out):
    # Only call after all scheduled training has completed.
    for e in planned:
        run=Path(out)/'experiments'/e['name'];_,sig=signature(config,e,frame)
        if not training_complete(run,sig,config['epochs']):raise RuntimeError('An experiment is unfinished or its saved settings/history do not match. Resume training before final test evaluation.')
    for e in planned:
        isolate_prediction_cache(frame,Path(out)/'experiments'/e['name'])
        test_saved(config,frame,e,out)
        _,sig=signature(config,e,frame)
        if not test_cache_complete(config,frame,e,out,signature_value=sig):raise RuntimeError('Test export is incomplete or inconsistent; no final handoff was generated.')
    from charts import analyze
    table=analyze(out,planned=planned);make_handoff(out);return table

def make_handoff(out):
    out=Path(out);target=out/'HANDOFF.zip'
    if not (out/'analysis/OPEN_RESULTS.html').exists():raise FileNotFoundError('Final analysis has not been generated.')
    with zipfile.ZipFile(target.with_suffix('.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['data_split.csv','settings.json','local_check.json']:
            if (out/name).exists():z.write(out/name,'results/'+name)
        for folder in ['experiments','analysis','custom_analysis','checks']:
            for p in (out/folder).rglob('*'):
                if not p.is_file() or p.name=='last.pt' or p.name.endswith('.tmp'):continue
                if any(part in ['validation_predictions','cache_quarantine'] for part in p.parts):continue
                if p.name.startswith(('val_','VALIDATION_CHECK')):continue
                z.write(p,'results/'+p.relative_to(out).as_posix())
        for name in ['build_handoff_html.py','OPEN_HANDOFF.html','build_handoff_assets.py','Main.ipynb','Analysis.ipynb','使用说明.txt','requirements.txt','README.md','run_project.py','REPORT_HANDOFF.md','REVIEW_CHANGES.md','requirements-tested.txt','build_report_handoff.py','verify_outputs.py']:
            if (PROJECT/name).exists():z.write(PROJECT/name,name)
        for p in (PROJECT/'src').glob('*.py'):z.write(p,'src/'+p.name)
        for p in (PROJECT/'tests').glob('*.py'):z.write(p,'tests/'+p.name)
        for item in (PROJECT/'docs').rglob('*'):
            if item.is_file():z.write(item,'docs/'+item.relative_to(PROJECT/'docs').as_posix())
        if (out/'5_EPOCHS.html').exists():z.write(out/'5_EPOCHS.html','results/5_EPOCHS.html')
    target.with_suffix('.tmp').replace(target);print('Share this file:',target,flush=True)
