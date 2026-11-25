"""
evaluate.py - Danh gia cac mo hinh tren Testing (hoac validation toan cuc).

Danh gia:
  * Server round 0           : mo hinh TRUOC khi huan luyen lien ket
  * Server round cuoi        : mo hinh SAU khi huan luyen xong (them --all_rounds de ve duong cong theo round)
  * Client X sau train cuc bo: mo hinh cuc bo (= global_{r-1} + delta cua client) o cac round chon,
                               TRUOC khi hop nhat. "Truoc train" cua client = server round r-1.

Vi du:
  python evaluate.py --run_dir ./runs/exp1 --data_root ./brain-tumor-mri-dataset --all_rounds
  python evaluate.py --run_dir ./runs/exp1 --split val --partition_dir ./partitions   # tune sieu tham so
Ket qua: <run_dir>/eval/eval_<split>.json / .csv / *.png
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np

from common import (CLASSES, build_model, compute_metrics, list_class_folder, load_global,
                    load_json, make_eval_loader, pick_device, predict, safe_load, save_json)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--data_root", help="Thu muc chua Testing/ (cho --split test)")
    ap.add_argument("--partition_dir", default="./partitions", help="Noi co global_val.json (cho --split val)")
    ap.add_argument("--all_rounds", action="store_true", help="Danh gia server o MOI round")
    ap.add_argument("--server_rounds", help="Chi danh gia server o cac round nay, vd 0,8")
    ap.add_argument("--client_rounds", default="first,last", help="'first,last' | 'all' | danh sach vd 1,5,10")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_plots", action="store_true")
    return ap.parse_args()


def round_of(path):
    return int(re.search(r"(\d+)", path.stem).group(1))


def main():
    a = parse_args()
    run_dir = Path(a.run_dir)
    device = pick_device(a.device)

    if a.split == "test":
        if not a.data_root:
            raise SystemExit("--split test can --data_root")
        samples = list_class_folder(Path(a.data_root) / "Testing")
    else:
        samples = [(p, l) for p, l in load_json(Path(a.partition_dir) / "global_val.json")["samples"]]
    print(f"[Eval] {a.split}: {len(samples)} anh | thiet bi {device}")
    loader = make_eval_loader(samples, a.batch_size)

    gfiles = sorted((run_dir / "global").glob("global_round_*.pt"), key=round_of)
    if not gfiles:
        raise SystemExit("Khong co file global nao trong run_dir/global")
    last = round_of(gfiles[-1])
    gpath = lambda r: run_dir / "global" / f"global_round_{r:02d}.pt"

    # Avoid dict.get(..., parsed_value): Python evaluates parsed_value eagerly,
    # which previously tried int("first") even when the key was "first,last".
    if a.client_rounds == "first,last":
        sel = {1, last}
    elif a.client_rounds == "all":
        sel = set(range(1, last + 1))
    else:
        try:
            sel = {int(x.strip()) for x in a.client_rounds.split(",") if x.strip()}
        except ValueError as exc:
            raise SystemExit("--client_rounds phai la 'first,last', 'all', hoac danh sach so nhu 1,5,10") from exc
        invalid = sorted(r for r in sel if r < 1 or r > last)
        if invalid:
            raise SystemExit(f"--client_rounds ngoai pham vi 1..{last}: {invalid}")
    if a.server_rounds:
        try:
            server_rounds = {int(x.strip()) for x in a.server_rounds.split(",") if x.strip()}
        except ValueError as exc:
            raise SystemExit("--server_rounds phai la danh sach so, vi du 0,8") from exc
        invalid = sorted(r for r in server_rounds if r < 0 or r > last)
        if invalid:
            raise SystemExit(f"--server_rounds ngoai pham vi 0..{last}: {invalid}")
    elif a.all_rounds:
        server_rounds = set(range(0, last + 1))
    else:
        server_rounds = {0, last}

    model = build_model(pretrained=False).to(device)
    results = []

    def run(label, kind, rnd, who, state):
        model.load_state_dict(state)
        y, p = predict(model, loader, device)
        m = compute_metrics(y, p)
        m.update(label=label, kind=kind, round=rnd, owner=who)
        results.append(m)

    # Thu tu bao cao: Server R0 -> (client sau train round chon xen ke) -> Server R cuoi
    for r in range(0, last + 1):
        if r in server_rounds:
            tag = " (TRUOC huan luyen)" if r == 0 else (" (SAU huan luyen xong)" if r == last else "")
            state, _ = load_global(gpath(r))
            run(f"Server | R{r}{tag}", "server", r, "server", state)
        if r + 1 in sel:
            prev, _ = load_global(gpath(r))
            for f in sorted((run_dir / "updates" / f"round_{r+1:02d}").glob("client_*.pt")):
                u = safe_load(f)
                st = {k: v.clone() for k, v in prev.items()}
                for k, d in u["delta"].items():
                    st[k] = st[k] + d
                run(f"Client {u['client_id']} | R{r+1} sau train cuc bo", "client", r + 1, u["client_id"], st)

    # ---- In bang -------------------------------------------------------------------------
    short = [c[:5] for c in CLASSES]
    print("\n" + f"{'Mo hinh':44s}{'Acc':>7s}{'BalAcc':>8s}{'F1':>7s} | " + " ".join(f"{s:>6s}" for s in short) + "   (recall)")
    print("-" * 105)
    for m in results:
        print(f"{m['label']:44s}{100*m['accuracy']:7.2f}{100*m['balanced_accuracy']:8.2f}{100*m['macro_f1']:7.2f} | "
              + " ".join(f"{100*x:6.1f}" for x in m["per_class_recall"]))

    out_dir = run_dir / "eval"
    save_json({"split": a.split, "classes": CLASSES, "results": results}, out_dir / f"eval_{a.split}.json")
    with open(out_dir / f"eval_{a.split}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "round", "accuracy", "balanced_accuracy", "macro_f1"]
                   + [f"recall_{c}" for c in CLASSES])
        for m in results:
            w.writerow([m["label"], m["kind"], m["round"], f"{m['accuracy']:.4f}",
                        f"{m['balanced_accuracy']:.4f}", f"{m['macro_f1']:.4f}"]
                       + [f"{x:.4f}" for x in m["per_class_recall"]])
    print(f"\n[Eval] Da luu: {out_dir}/eval_{a.split}.json|csv")

    if not a.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Eval] Khong co matplotlib -> bo qua ve hinh")
            return
        # Ma tran nham lan
        n = len(results)
        cols = min(n, 4)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, m in zip(axes.ravel(), results):
            ax.axis("on")
            cm = np.array(m["confusion_matrix"], dtype=float)
            cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(CLASSES)), short, fontsize=7)
            ax.set_yticks(range(len(CLASSES)), short, fontsize=7)
            for i in range(len(CLASSES)):
                for j in range(len(CLASSES)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_title(f"{m['label']}\nacc {100*m['accuracy']:.1f}%", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{a.split}.png", dpi=150)
        plt.close(fig)
        # Duong cong server theo round
        srv = [m for m in results if m["kind"] == "server"]
        if len(srv) > 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            xs = [m["round"] for m in srv]
            ax.plot(xs, [100 * m["accuracy"] for m in srv], "o-", label="Accuracy")
            ax.plot(xs, [100 * m["balanced_accuracy"] for m in srv], "s--", label="Balanced acc.")
            ax.plot(xs, [100 * m["macro_f1"] for m in srv], "^:", label="Macro-F1")
            ax.set_xlabel("Round")
            ax.set_ylabel("%")
            ax.set_title(f"Mo hinh server theo round ({a.split})")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"server_by_round_{a.split}.png", dpi=150)
            plt.close(fig)
        print(f"[Eval] Da ve hinh trong {out_dir}")


if __name__ == "__main__":
    main()
