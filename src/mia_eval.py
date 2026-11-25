"""Loss-based membership inference on actual client members vs held-out Testing images."""
import argparse,json
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,roc_curve
from common import build_model,list_class_folder,load_global,load_json,make_eval_loader,pick_device

def losses(model,loader,device):
    model.eval(); ls=[]
    with torch.no_grad():
        for x,y in loader: ls.extend(F.cross_entropy(model(x.to(device)),y.to(device),reduction='none').cpu().tolist())
    return np.asarray(ls)
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data_root',required=True); ap.add_argument('--run_dir',required=True); ap.add_argument('--round',type=int,required=True); ap.add_argument('--out',required=True); ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--partition_dir',default='partitions'); ap.add_argument('--clients',nargs='+',default=['A','B','C','D']); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    dev=pick_device(a.device); root=Path(a.data_root); non=list_class_folder(root/'Testing')
    # Membership is the union of images actually assigned to clients. The 400 images
    # left over after validation/client partitioning must not be labeled as members.
    mem_by_path={}
    for cid in a.clients:
        manifest=Path(a.partition_dir)/f'client_{cid}.json'
        if not manifest.is_file(): raise SystemExit(f'Client manifest not found: {manifest}')
        for path,label in load_json(manifest)['samples']:
            key=str(Path(path).resolve())
            if key in mem_by_path and mem_by_path[key] != int(label):
                raise SystemExit(f'Conflicting labels for client member {key}')
            mem_by_path[key]=int(label)
    mem=[(p,l) for p,l in mem_by_path.items()]
    if not mem: raise SystemExit('No client member samples found')
    test_paths={str(Path(p).resolve()) for p,_ in non}
    overlap=set(mem_by_path)&test_paths
    if overlap: raise SystemExit(f'Data leakage: {len(overlap)} member images also occur in Testing')
    print(f'[MIA] actual client members={len(mem)} non-member Testing={len(non)}')
    model=build_model(pretrained=False).to(dev); state,_=load_global(Path(a.run_dir)/'global'/f'global_round_{a.round:02d}.pt'); model.load_state_dict(state)
    lm=losses(model,make_eval_loader(mem,a.batch_size),dev); ln=losses(model,make_eval_loader(non,a.batch_size),dev)
    scores=np.r_[-lm,-ln]; labels=np.r_[np.ones(len(lm)),np.zeros(len(ln))]; auc=float(roc_auc_score(labels,scores)); fpr,tpr,_=roc_curve(labels,scores); idx=int(np.argmax(tpr-fpr))
    result={'run_dir':a.run_dir,'round':a.round,'n_member':len(lm),'n_nonmember':len(ln),'loss_member_mean':float(lm.mean()),'loss_nonmember_mean':float(ln.mean()),'loss_gap':float(ln.mean()-lm.mean()),'mia_auc':auc,'mia_attack_acc_best':float(.5*(tpr[idx]+1-fpr[idx])),'tpr_at_fpr_1pct':float(np.interp(.01,fpr,tpr)),'tpr_at_fpr_01pct':float(np.interp(.001,fpr,tpr))}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
