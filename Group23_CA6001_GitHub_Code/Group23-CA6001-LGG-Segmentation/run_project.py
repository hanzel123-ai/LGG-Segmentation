"""Start an explicit new experiment; the published results folder stays read-only."""
from pathlib import Path
import argparse,contextlib,ctypes,json,re,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams:stream.flush()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,help='Extracted Kaggle dataset, exactly one kaggle_3m copy')
    parser.add_argument('--run-name',default=time.strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--resume',action='store_true',help='Resume this explicitly named new run, never the published archive')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_name):parser.error('Use a simple run name with letters, digits, _, - or .')
    if args.epochs<1:parser.error('--epochs must be positive')
    data=Path(args.data_root).expanduser().resolve()
    if not data.is_dir():parser.error('--data-root must be an extracted dataset directory')
    out=(ROOT/'runs'/args.run_name).resolve()
    if not out.is_relative_to(ROOT/'runs'):parser.error('Run directory must stay under runs/')
    if out.exists() and not args.resume:parser.error('Run exists. Use a new --run-name or explicitly --resume it.')
    if args.resume and not (out/'settings.json').exists():parser.error('No saved settings found for the requested resume run')
    try:
        import torch
        from core import train,test_saved,atomic_json
        from workflow import settings,experiments,prepare,local_check
        from charts import analyze
    except ImportError as exc:parser.error('Install requirements.txt in your environment first: '+str(exc))
    config=settings();config.update(epochs=args.epochs,device=args.device)
    if args.resume:
        old=json.loads((out/'settings.json').read_text())
        for key in ['seed','image_size','base_channels','batch_size','epochs','learning_rate','threshold']:
            if old[key]!=config[key]:parser.error('Resume setting differs: '+key)
    planned=experiments();out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        print('New experiment directory:',out,flush=True)
        print('Published results are separate; existing report numbers will not be rewritten.',flush=True)
        print('Python:',sys.version,'Torch:',torch.__version__,'CUDA:',torch.cuda.is_available(),flush=True)
        try:
            if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            frame=prepare(data,config,out);local_check(config,frame,out)
            if args.check_only:return
            for experiment in planned:train(config,frame,experiment,out)
            for experiment in planned:test_saved(config,frame,experiment,out)
            # Do not use workflow.finish/make_handoff: archived documents belong to the published run.
            table=analyze(out,planned=planned)
            atomic_json(out/'RUN_COMPLETE.json',{'models':[e['name'] for e in planned],'epochs':config['epochs'],'scientific_settings':config,'published_results_modified':False})
            print(table.to_string(index=False),flush=True);print('New results page:',out/'analysis/OPEN_RESULTS.html',flush=True)
        finally:
            if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
