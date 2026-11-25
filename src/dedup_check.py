"""Find exact or near duplicate MRI images across Training and Testing."""
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image
from common import find_duplicates, hamming_hex, list_class_folder, perceptual_hash

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_root', required=True); ap.add_argument('--hash_size',type=int,default=8)
    ap.add_argument('--threshold',type=int,default=0); ap.add_argument('--remove',action='store_true')
    a=ap.parse_args(); root=Path(a.data_root)
    train=list_class_folder(root/'Training'); test=list_class_folder(root/'Testing')
    def build(samples):
        m=defaultdict(list)
        for p,l in samples:
            with Image.open(p) as im: m[perceptual_hash(im,a.hash_size)].append((str(p),l))
        return m
    tr,te=build(train),build(test); pairs=[]
    for th,titems in te.items():
        for rh,ritems in tr.items():
            if hamming_hex(th,rh)<=a.threshold:
                pairs.extend((tp,rp) for tp,_ in titems for rp,_ in ritems)
    print(f'Training={len(train)} Testing={len(test)} cross-split pairs={len(pairs)} threshold={a.threshold}')
    for tp,rp in pairs[:20]: print(f' {Path(tp).name} ~ {Path(rp).name}')
    for name,m in [('Training',tr),('Testing',te)]:
        print(f'{name} duplicate pairs (threshold={a.threshold}): {len(find_duplicates(m,a.threshold))}')
    if a.remove:
        paths={rp for _,rp in pairs}
        for p in paths: Path(p).unlink(missing_ok=True)
        print(f'Removed {len(paths)} Training files; repartition the data.')
    elif pairs: print('Review matches, then rerun with --remove if appropriate.')
if __name__=='__main__': main()
