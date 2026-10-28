"""Readable shared data and model code. Architecture references are in README.md."""
from pathlib import Path
import random, json, time, hashlib, os
import numpy as np
import pandas as pd
import tifffile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_manifest(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError('Set DATA_ROOT to the extracted kaggle_3m folder.')
    rows = []
    for mask in sorted(root.rglob('*_mask.tif')):
        image = mask.with_name(mask.name.replace('_mask.tif', '.tif'))
        if not image.is_file():
            raise FileNotFoundError(f'Missing image: {image}')
        # Dataset filename: patient identifier + numeric slice index.
        patient, slice_id = image.stem.rsplit('_', 1)
        if not slice_id.isdigit():
            raise ValueError(f'Unexpected filename: {image.name}')
        x, y = tifffile.imread(image), tifffile.imread(mask)
        if x.ndim != 3 or x.shape[-1] != 3 or y.ndim != 2 or x.shape[:2] != y.shape:
            raise ValueError(f'Unexpected image/mask shape: {image.name}: {x.shape}, {y.shape}')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Nonfinite pixels: {image.name}')
        values = set(np.unique(y).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError(f'Unexpected mask values: {mask.name}: {values}')
        rows.append(dict(patient=patient, slice_id=int(slice_id), image=str(image.resolve()),
                         mask=str(mask.resolve()), positive=bool((y > 0).any())))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No *_mask.tif files found. Extract the ZIP first.')
    if df.duplicated(['patient', 'slice_id']).any():
        raise ValueError('Duplicate slices found. Point DATA_ROOT to only one dataset copy.')
    return df.sort_values(['patient', 'slice_id']).reset_index(drop=True)

def assign_splits(df, seed=42):
    patients = np.array(sorted(df.patient.unique()))
    if len(patients) < 7:
        raise ValueError('At least 7 patients are needed for this split.')
    np.random.default_rng(seed).shuffle(patients)
    n_val = max(1, round(len(patients) * .15))
    n_test = max(1, round(len(patients) * .15))
    mapping = {p: ('test' if i < n_test else 'val' if i < n_test+n_val else 'train')
               for i, p in enumerate(patients)}
    result = df.copy()
    result['split'] = result.patient.map(mapping)
    assert result.groupby('patient').split.nunique().max() == 1
    return result

class MRIDataset(Dataset):
    def __init__(self, frame, size=256):
        self.frame = frame.reset_index(drop=True)
        self.size = size
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        x = tifffile.imread(row.image).astype(np.float32)
        lo, hi = np.percentile(x, [1, 99], axis=(0, 1), keepdims=True)
        x = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)
        y = (tifffile.imread(row['mask']) > 0).astype(np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy(y[None].copy())
        x = F.interpolate(x[None], (self.size, self.size), mode='bilinear', align_corners=False)[0]
        y = F.interpolate(y[None], (self.size, self.size), mode='nearest')[0]
        return x, y, row.patient, int(row.slice_id)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class UNet(nn.Module):
    def __init__(self, base=32, use_skips=True):
        super().__init__()
        self.use_skips = use_skips
        widths = [base * 2**i for i in range(5)]
        self.encoders = nn.ModuleList([conv_block(a, b) for a, b in zip([3]+widths[:3], widths[:4])])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(widths[3], widths[4])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[i+1], widths[i], 2, stride=2) for i in reversed(range(4))])
        self.decoders = nn.ModuleList([conv_block(widths[i]*(2 if use_skips else 1), widths[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat([x, skip], dim=1) if self.use_skips else x)
        return self.head(x)

def score_counts(frame):
    frame = frame.copy()
    denom = frame.pred_pixels + frame.true_pixels
    union = denom - frame.intersection
    frame['dice'] = np.divide(2*frame.intersection, denom, out=np.ones(len(frame)), where=denom>0)
    frame['iou'] = np.divide(frame.intersection, union, out=np.ones(len(frame)), where=union>0)
    return frame

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(temp,path)

def atomic_save(value,path):
    temp=Path(str(path)+'.tmp');torch.save(value,temp);os.replace(temp,path)

def set_device(config):
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(min(config['cpu_threads'],os.cpu_count() or 1))
    return torch.device('cuda' if torch.cuda.is_available() and config['device']!='cpu' else 'cpu')

def autocast(config,device):
    from contextlib import nullcontext
    if device.type=='cuda' and config['gpu_amp']:return torch.autocast('cuda',dtype=torch.float16)
    if device.type=='cpu' and config['cpu_bfloat16']:return torch.autocast('cpu',dtype=torch.bfloat16)
    return nullcontext()

def criterion(logits,target,kind):
    bce=F.binary_cross_entropy_with_logits(logits,target)
    if kind=='bce':return bce
    if kind!='bce_dice':raise ValueError('Unknown loss: '+kind)
    prob=logits.sigmoid();dims=(1,2,3)
    dice=(2*(prob*target).sum(dims)+1)/(prob.sum(dims)+target.sum(dims)+1)
    return .5*bce+.5*(1-dice.mean())

def loader(frame,config,split,epoch=0):
    return DataLoader(MRIDataset(frame[frame.split==split],config['image_size']),batch_size=config['batch_size'],
        shuffle=split=='train',num_workers=0,generator=torch.Generator().manual_seed(config['seed']+epoch))

def signature(config,experiment,frame):
    keys=['seed','image_size','base_channels','batch_size','epochs','learning_rate','threshold']
    spec={k:config[k] for k in keys};spec['experiment']={k:experiment[k] for k in ['name','use_skips','loss']}
    # Canonical LF keeps a patient/content split portable between Windows and Linux.
    spec['split_hash']=hashlib.sha256(frame[['patient','slice_id','split','image_sha256','mask_sha256']].to_csv(index=False,lineterminator='\n').encode()).hexdigest()
    spec['core_hash']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return spec,hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()

@torch.no_grad()
def evaluate(model,frame,config,experiment,device,split,export=None):
    model.eval();rows=[];total=0.;n=0;start=time.time()
    if export:export=Path(export);export.mkdir(parents=True,exist_ok=True)
    for x,y,patients,ids in loader(frame,config,split):
        x=x.to(device,memory_format=torch.channels_last);y=y.to(device)
        with autocast(config,device):logits=model(x).float()
        loss=criterion(logits,y,experiment['loss'])
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite predictions/loss during evaluation.')
        total+=loss.item()*len(x);n+=len(x)
        pred=logits.sigmoid()>=config['threshold'];truth=y>.5
        for i,patient in enumerate(patients):
            rows.append(dict(patient=patient,slice_id=int(ids[i]),intersection=int((pred[i]&truth[i]).sum()),
                pred_pixels=int(pred[i].sum()),true_pixels=int(truth[i].sum()),total_pixels=truth[i].numel()))
            if export:
                stem=f'{patient}_{int(ids[i])}'
                for suffix,a in [('image',x[i,1].cpu().numpy()),('true',truth[i,0].cpu().numpy()),('pred',pred[i,0].cpu().numpy())]:
                    Image.fromarray((a*255).astype('uint8')).save(export/f'{stem}_{suffix}.png')
    slices=score_counts(pd.DataFrame(rows));patients=score_counts(slices.groupby('patient')[['intersection','pred_pixels','true_pixels','total_pixels']].sum().reset_index())
    metrics=dict(loss=total/n,dice=float(patients.dice.mean()),iou=float(patients.iou.mean()),patients=len(patients),slices=n,evaluation_seconds=time.time()-start)
    return metrics,slices,patients

def train(config,frame,experiment,out,epoch_budget=None):
    run=Path(out)/'experiments'/experiment['name'];run.mkdir(parents=True,exist_ok=True)
    spec,sig=signature(config,experiment,frame);meta=run/'experiment.json'
    if meta.exists() and json.loads(meta.read_text())['signature']!=sig:
        raise ValueError('Settings changed for '+experiment['name']+'. Use a NEW output_dir or experiment name.')
    if (run/'TRAINING_COMPLETE.json').exists():
        from result_guards import training_complete
        if not training_complete(run,sig,config['epochs']):raise RuntimeError('Training completion marker does not match history/signature.')
        print(experiment['name'],'training already complete: SKIPPED',flush=True);return True
    device=set_device(config);seed_everything(config['seed'])
    model=UNet(config['base_channels'],experiment['use_skips']).to(device).to(memory_format=torch.channels_last)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and config['gpu_amp'])
    sessions=json.loads(meta.read_text()).get('sessions',[]) if meta.exists() else []
    runtime={k:config[k] for k in ['device','cpu_threads','cpu_bfloat16','gpu_amp']}
    sessions.append(dict(started=time.strftime('%Y-%m-%d %H:%M:%S'),device=str(device),torch=str(torch.__version__),runtime=runtime))
    atomic_json(meta,dict(signature=sig,spec=spec,runtime=runtime,sessions=sessions,device=str(device),torch=str(torch.__version__),parameters=sum(p.numel() for p in model.parameters())))
    (run/'architecture.txt').write_text(str(model),encoding='utf-8')
    history=[];best=-1.;best_epoch=0;start_epoch=1
    if (run/'last.pt').exists():
        # Trusted checkpoint written by this project only; do not substitute unknown downloaded files.
        ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        if ck['signature']!=sig:raise ValueError('Checkpoint signature mismatch.')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        if scaler.is_enabled() and ck['scaler']:scaler.load_state_dict(ck['scaler'])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v) and k!='step':state[k]=v.to(device)
        history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
        if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
        if best_epoch==ck['epoch']:atomic_save(ck['model'],run/'best.pt')
        print('RESUME',experiment['name'],'epoch',start_epoch,flush=True)
    end=config['epochs'] if epoch_budget is None else min(config['epochs'],start_epoch+epoch_budget-1)
    for epoch in range(start_epoch,end+1):
        start=time.time();model.train();total=0.;n=0;train_loader=loader(frame,config,'train',epoch)
        for step,(x,y,_,_) in enumerate(train_loader,1):
            x=x.to(device,memory_format=torch.channels_last);y=y.to(device);optimizer.zero_grad(set_to_none=True)
            with autocast(config,device):logits=model(x).float()
            loss=criterion(logits,y,experiment['loss'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss.')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total+=loss.item()*len(x);n+=len(x)
            if step==1 and epoch==1:print('First-batch rough training estimate (minutes):',round((time.time()-start)*len(train_loader)*config['epochs']/60,1),flush=True)
            if step%25==0:print(experiment['name'],f'epoch {epoch}, batch {step}/{len(train_loader)}',flush=True)
        val,_,_=evaluate(model,frame,config,experiment,device,'val')
        row=dict(epoch=epoch,train_loss=total/n,val_loss=val['loss'],val_dice=val['dice'],val_iou=val['iou'],seconds=time.time()-start)
        history.append(row);improved=val['dice']>best
        if improved:best=val['dice'];best_epoch=epoch
        ck=dict(signature=sig,epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            history=history,best=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        atomic_save(ck,run/'last.pt')
        if improved:atomic_save(model.state_dict(),run/'best.pt')
        pd.DataFrame(history).to_csv(run/'history.csv',index=False);atomic_json(run/'best_epoch.json',dict(epoch=best_epoch,val_dice=best))
        print(experiment['name'],f'{epoch}/{config["epochs"]}: loss {row["train_loss"]:.4f}, val Dice {val["dice"]:.4f}, {row["seconds"]:.0f}s',flush=True)
    if end==config['epochs']:
        pd.DataFrame(history).to_csv(run/'history.csv',index=False);atomic_json(run/'best_epoch.json',dict(epoch=best_epoch,val_dice=best))
        atomic_json(run/'TRAINING_COMPLETE.json',dict(signature=sig,epochs=config['epochs']));return True
    return False

def test_saved(config,frame,experiment,out):
    from result_guards import test_cache_complete, isolate_prediction_cache, training_complete
    run=Path(out)/'experiments'/experiment['name'];_,sig=signature(config,experiment,frame)
    if not training_complete(run,sig,config['epochs']):raise RuntimeError('Train the model first; checkpoint/epoch evidence is incomplete.')
    if json.loads((run/'experiment.json').read_text())['signature']!=sig:raise ValueError('Evaluation configuration changed.')
    if test_cache_complete(config,frame,experiment,out,signature_value=sig):
        print('Saved evaluation reused:',experiment['name'],flush=True);return
    isolate_prediction_cache(frame,run)
    device=set_device(config);model=UNet(config['base_channels'],experiment['use_skips']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True))
    metrics,slices,patients=evaluate(model,frame,config,experiment,device,'test',run/'predictions')
    slices.to_csv(run/'test_slices.csv',index=False);patients.to_csv(run/'test_patients.csv',index=False)
    atomic_json(run/'test_metrics.json',metrics);atomic_json(run/'EVALUATION_COMPLETE.json',dict(signature=sig,best_weights_sha256=hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()));print('TEST',experiment['name'],metrics,flush=True)

@torch.no_grad()
def predict_new(run_dir,image_path,destination):
    run=Path(run_dir);meta=json.loads((run/'experiment.json').read_text());spec=dict(meta['spec']);spec.update(meta.get('runtime',{}))
    # Preserve the recorded precision policy; only device discovery is portable.
    spec.setdefault('cpu_threads',4);spec.setdefault('cpu_bfloat16',False);spec.setdefault('gpu_amp',True)
    spec['device']='auto';experiment=spec['experiment']
    device=set_device(spec);model=UNet(spec['base_channels'],experiment['use_skips']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(run/'best.pt',map_location=device,weights_only=True));model.eval()
    x=tifffile.imread(image_path).astype(np.float32)
    if x.ndim!=3 or x.shape[-1]!=3:raise ValueError('Expected three-channel TIFF matching training data.')
    lo,hi=np.percentile(x,[1,99],axis=(0,1),keepdims=True)
    x=np.clip((x-lo)/np.maximum(hi-lo,1e-6),0,1).astype(np.float32)
    x=torch.from_numpy(x.transpose(2,0,1).copy())[None]
    x=F.interpolate(x,(spec['image_size'],spec['image_size']),mode='bilinear',align_corners=False).to(device,memory_format=torch.channels_last)
    with autocast(spec,device):prob=model(x).float().sigmoid()[0,0].cpu().numpy()
    mask=(prob>=spec['threshold']).astype('uint8')*255;Path(destination).parent.mkdir(parents=True,exist_ok=True);Image.fromarray(mask).save(destination)
    return prob,mask
