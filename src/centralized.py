"""Centralized training baseline using all Training images."""
import argparse,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from common import CachedImageDataset,NUM_CLASSES,atomic_save,build_model,list_class_folder,pick_device,prepare_for_local_training,set_seed,train_transform

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data_root',required=True); ap.add_argument('--run_dir',required=True)
    ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--batch_size',type=int,default=32); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--weight_decay',type=float,default=1e-2); ap.add_argument('--train_mode',default='last_blocks'); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    set_seed(a.seed); dev=pick_device(a.device); ds=CachedImageDataset(list_class_folder(Path(a.data_root)/'Training'),train_transform())
    loader=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=0,pin_memory=dev.type=='cuda')
    model=build_model(pretrained=True).to(dev); start,_=prepare_for_local_training(model,a.train_mode)
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=a.weight_decay); t=time.time()
    for ep in range(a.epochs):
        losses=[]
        for x,y in loader:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(x,start=start),y); loss.backward(); opt.step(); losses.append(loss.item())
        if ep==0 or (ep+1)%5==0: print(f'Epoch {ep+1}/{a.epochs} loss={sum(losses)/max(1,len(losses)):.4f} elapsed={time.time()-t:.0f}s')
    out=Path(a.run_dir)/'global'/'global_round_00.pt'; atomic_save({'round':0,'num_classes':NUM_CLASSES,'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},out); print(f'Saved {out}')
if __name__=='__main__': main()
