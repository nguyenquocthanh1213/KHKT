"""
partition.py - Chia tap Training thanh cac "benh vien" theo kich ban Non-IID CUC DOAN.

Kich ban goc cua ban: moi benh vien 90% mot lop + 5% + 5% cho hai lop con lai (3 lop).
Bo du lieu that co 4 lop nen dieu chinh nhe: 90% lop chu dao + 10% chia deu cho 3 lop con lai
(~3.3% moi lop). Co the doi bang --dominant_frac / --clients.

  - Cac client KHONG chung anh nao (tap roi nhau).
  - Giu lai `val_per_class` anh/lop tu Training lam tap validation toan cuc (de chon sieu tham
    so ma khong cham vao tap Testing). Tap nay khong thuoc client nao.
  - Chi sinh file manifest (danh sach duong dan + nhan), khong sao chep anh.

Vi du:
  python partition.py --data_root ./brain-tumor-mri-dataset --out_dir ./partitions
  python partition.py --data_root ... --clients A:glioma B:meningioma C:notumor D:pituitary
"""
import argparse
import random
from pathlib import Path

from common import CLASSES, CLASS_TO_IDX, list_class_folder, save_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Thu muc chua Training/ va Testing/")
    ap.add_argument("--out_dir", default="./partitions")
    ap.add_argument("--clients", nargs="+", default=[
        "A:glioma", "B:meningioma", "C:notumor", "D:pituitary"],
                    help="Danh sach <id>:<lop_chu_dao>")
    ap.add_argument("--samples_per_client", type=int, default=1200)
    ap.add_argument("--dominant_frac", type=float, default=0.9)
    ap.add_argument("--val_per_class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    a = parse_args()
    rng = random.Random(a.seed)
    train_root = Path(a.data_root).resolve() / "Training"
    all_samples = list_class_folder(train_root)

    pools = {c: [] for c in range(len(CLASSES))}
    for p, l in all_samples:
        pools[l].append(p)
    for l in pools:
        rng.shuffle(pools[l])

    # Tap validation toan cuc (tach truoc, khong thuoc client nao)
    val = []
    for l in pools:
        for _ in range(a.val_per_class):
            val.append((pools[l].pop(), l))
    save_json({"samples": val}, Path(a.out_dir) / "global_val.json")

    n_cls = len(CLASSES)
    summary = []
    for spec in a.clients:
        cid, dom_name = spec.split(":")
        dom = CLASS_TO_IDX[dom_name]
        n = a.samples_per_client
        n_dom = round(n * a.dominant_frac)
        n_other = (n - n_dom) // (n_cls - 1)
        quota = {c: (n_dom if c == dom else n_other) for c in range(n_cls)}
        samples = []
        for c, q in quota.items():
            if len(pools[c]) < q:
                raise SystemExit(
                    f"Client {cid}: lop {CLASSES[c]} con {len(pools[c])} anh < can {q}. "
                    f"Giam --samples_per_client hoac --val_per_class.")
            for _ in range(q):
                samples.append((pools[c].pop(), c))
        rng.shuffle(samples)
        counts = [sum(1 for _, l in samples if l == c) for c in range(n_cls)]
        save_json({"client_id": cid, "dominant_class": dom_name,
                   "class_counts": counts, "samples": samples},
                  Path(a.out_dir) / f"client_{cid}.json")
        summary.append((cid, dom_name, counts))

    print(f"Da tao manifest trong {a.out_dir}")
    print(f"{'client':8s}{'chu dao':14s}" + "".join(f"{c:>12s}" for c in CLASSES))
    for cid, dom, counts in summary:
        tot = sum(counts)
        print(f"{cid:8s}{dom:14s}" + "".join(f"{n:>6d} ({100*n/tot:4.1f}%)" for n in counts))
    print(f"Validation toan cuc: {len(val)} anh ({a.val_per_class}/lop)")


if __name__ == "__main__":
    main()
