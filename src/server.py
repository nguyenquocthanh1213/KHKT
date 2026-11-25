"""
server.py - Server trung uong KHONG mo cong mang. Chi lam viec voi file.

  # Buoc 0: tao mo hinh toan cuc ban dau (round 0) -> gui file nay cho cac client
  python server.py init --run_dir ./runs/exp1

  # Moi round: sau khi nhan du file cap nhat cua cac client vao updates/round_XX/
  python server.py aggregate --run_dir ./runs/exp1 --round 1
  -> tao global/global_round_01.pt, gui lai cho client o round ke tiep.

Cau truc thu muc:
  runs/exp1/global/global_round_00.pt, global_round_01.pt, ...
  runs/exp1/updates/round_01/client_A.pt, client_B.pt, ...
  runs/exp1/logs/aggregate_round_01.json
"""
import argparse
import math
from pathlib import Path

import torch

from common import (NUM_CLASSES, TRAIN_MODES, atomic_save, build_model, load_global, safe_load,
                    save_json, set_seed)


def cmd_init(a):
    set_seed(a.seed)
    try:
        model = build_model(pretrained=not a.no_pretrained)
    except Exception as e:  # khong tai duoc trong so ImageNet (offline...)
        raise SystemExit(f"Khong tai duoc trong so pretrained ({e}). Chay lai voi mang, hoac them --no_pretrained.")
    out = Path(a.run_dir) / "global" / "global_round_00.pt"
    atomic_save({"round": 0, "num_classes": NUM_CLASSES,
                 "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, out)
    print(f"[Server] Da tao mo hinh toan cuc ban dau: {out}  (pretrained={not a.no_pretrained})")


def cmd_aggregate(a):
    run_dir = Path(a.run_dir)
    r = a.round
    prev_state, prev_round = load_global(run_dir / "global" / f"global_round_{r - 1:02d}.pt")
    if prev_round != r - 1:
        raise SystemExit(f"File global_round_{r-1:02d}.pt thuc ra la round {prev_round}.")

    files = sorted((run_dir / "updates" / f"round_{r:02d}").glob("client_*.pt"))
    if a.clients:
        files = [f for f in files if f.stem.replace("client_", "") in a.clients]
    if len(files) < a.min_clients:
        raise SystemExit(f"Moi co {len(files)} file cap nhat, can toi thieu {a.min_clients}.")

    updates, rejected, modes = [], [], set()
    for f in files:
        try:
            u = safe_load(f)  # weights_only=True: file doc hai khong the chay ma
            assert u["round"] == r, f"sai round ({u['round']} != {r})"
            assert u["num_samples"] > 0, "num_samples khong hop le"
            for k, v in u["delta"].items():
                assert k in prev_state, f"tham so la: {k}"
                assert v.shape == prev_state[k].shape, f"sai shape: {k}"
                assert torch.isfinite(v).all(), f"co NaN/Inf: {k}"
            modes.add(u["train_mode"])
            updates.append(u)
        except Exception as e:
            rejected.append({"file": f.name, "ly_do": str(e)})
            print(f"[Server] BO QUA {f.name}: {e}")
    if len(updates) < a.min_clients:
        raise SystemExit("Khong du cap nhat hop le de hop nhat.")
    if len(modes) > 1:
        raise SystemExit(f"Cac client dung train_mode khac nhau: {modes}")

    # Tuy chon: cat chuan delta (phong client loi / doc hai)
    norms = {}
    for u in updates:
        nrm = math.sqrt(sum(float(v.double().pow(2).sum()) for v in u["delta"].values()))
        norms[u["client_id"]] = nrm
        if a.max_delta_norm > 0 and nrm > a.max_delta_norm:
            s = a.max_delta_norm / nrm
            u["delta"] = {k: v * s for k, v in u["delta"].items()}

    # FedAvg tren delta:  w_r = w_{r-1} + server_lr * sum_k (n_k / N) * delta_k
    if a.weighting == "samples":
        total = sum(u["num_samples"] for u in updates)
        weights = {u["client_id"]: u["num_samples"] / total for u in updates}
    else:
        weights = {u["client_id"]: 1.0 / len(updates) for u in updates}

    new_state = {k: v.clone() for k, v in prev_state.items()}
    for u in updates:
        for k, d in u["delta"].items():
            new_state[k] = new_state[k] + a.server_lr * weights[u["client_id"]] * d.to(new_state[k].dtype)

    out = run_dir / "global" / f"global_round_{r:02d}.pt"
    atomic_save({"round": r, "num_classes": NUM_CLASSES, "state_dict": new_state}, out)

    eps = [u["dp_epsilon"] for u in updates if u["dp_enabled"]]
    log = {"round": r, "clients": [u["client_id"] for u in updates], "weights": weights,
           "delta_norm": norms, "train_loss": {u["client_id"]: u["train_loss"] for u in updates},
           "dp_epsilon": {u["client_id"]: u["dp_epsilon"] for u in updates},
           "rejected": rejected, "server_lr": a.server_lr, "max_delta_norm": a.max_delta_norm}
    save_json(log, run_dir / "logs" / f"aggregate_round_{r:02d}.json")
    print(f"[Server] Round {r}: hop nhat {len(updates)} client {list(weights)} -> {out}")
    print("         trong so:", {k: round(v, 3) for k, v in weights.items()},
          "| ||delta||:", {k: round(v, 3) for k, v in norms.items()})
    if eps:
        print(f"         DP: epsilon tich luy lon nhat giua cac client = {max(eps):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("aggregate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--clients", nargs="*", default=None, help="Chi hop nhat cac client nay (mac dinh: tat ca)")
    p.add_argument("--min_clients", type=int, default=2)
    p.add_argument("--weighting", choices=["samples", "uniform"], default="samples")
    p.add_argument("--server_lr", type=float, default=1.0)
    p.add_argument("--max_delta_norm", type=float, default=0.0, help=">0: cat chuan L2 cua delta moi client")
    p.set_defaults(fn=cmd_aggregate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
