"""
client.py - Chay RIENG LE tai moi benh vien. Khong ket noi mang.

Quy trinh moi round:
  1. Doc file mo hinh toan cuc  (global_round_{r-1}.pt, do server gui qua file)
  2. Huan luyen cuc bo tren du lieu rieng (FedProx + tuy chon DP-SGD)
  3. Ghi file cap nhat           (updates/round_{r}/client_{id}.pt) -> gui file nay cho server.
     File chi chua DELTA cua cac tham so duoc huan luyen + vai thong so; KHONG co anh.

Vi du (du lieu that cua benh vien, thu muc <lop>/*.jpg):
  python client.py --client_id A --round 1 --run_dir ./runs/exp1 --data_dir ./hospital_A \
                   --mu 0.05 --dp --noise_multiplier 1.0 --max_grad_norm 1.0
Vi du (mo phong, dung manifest tu partition.py):
  python client.py --client_id A --round 1 --run_dir ./runs/exp1 --manifest ./partitions/client_A.json
"""
import argparse
import math
import secrets
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad_and_value, vmap
from torch.utils.data import DataLoader

from common import (CachedImageDataset, NUM_CLASSES, TRAIN_MODES, atomic_save, build_model,
                    list_class_folder, load_global, load_json, pick_device,
                    prepare_for_local_training, set_seed, train_transform)
from privacy import ClientPrivacyLedger


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client_id", required=True)
    ap.add_argument("--round", type=int, required=True, help="So round hien tai (bat dau tu 1)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--global_model", default=None, help="Mac dinh: <run_dir>/global/global_round_{r-1}.pt")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_dir", help="Thu muc anh cuc bo: <data_dir>/<glioma|meningioma|notumor|pituitary>/*")
    src.add_argument("--manifest", help="File JSON tu partition.py (che do mo phong)")
    # Huan luyen
    ap.add_argument("--train_mode", choices=list(TRAIN_MODES), default="last_blocks")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["ce", "balanced"], default="balanced",
                    help="balanced = Balanced-Softmax (logit adjustment) xu ly lech nhan cuc bo")
    ap.add_argument("--tau", type=float, default=1.0, help="Cuong do logit adjustment")
    ap.add_argument("--mu", type=float, default=0.05, help="He so FedProx (0 = FedAvg thuan)")
    # DP-SGD
    ap.add_argument("--dp", action="store_true", help="Bat DP-SGD (per-sample clipping + nhieu Gauss)")
    ap.add_argument("--noise_multiplier", type=float, default=1.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max_epsilon", type=float, default=float("inf"),
                    help="Du toan epsilon tich luy vuot nguong nay -> tu choi huan luyen/gui file")
    ap.add_argument("--state_dir", default=None, help="Noi luu so cai quyen rieng tu (mac dinh <run_dir>/client_state)")
    # He thong
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true", help="Tat autocast fp16 cho phan dong bang (chi GPU)")
    ap.add_argument("--no_secure_rng", dest="secure_rng", action="store_false", default=True)
    ap.add_argument("--verify_dp", action="store_true")
    ap.add_argument("--auto_clip", action="store_true")
    ap.add_argument("--auto_clip_quantile", type=float, default=0.9)
    ap.add_argument("--auto_clip_batches", type=int, default=3)
    ap.add_argument("--target_epsilon", type=float, default=None)
    ap.add_argument("--planned_rounds", type=int, default=None,
                    help="Total rounds in this run; target_epsilon is interpreted as the whole-run budget")
    return ap.parse_args()


class PoissonBatchSampler:
    """Poisson sampling dung nhu gia dinh cua RDP accountant: moi mau duoc chon doc lap voi
    xac suat q = batch_size / N tai moi buoc."""

    def __init__(self, n, q, steps, seed, secure=True):
        self.n, self.q, self.steps = n, q, steps
        self.rng = np.random.default_rng(seed)
        self.secure = secure

    def __iter__(self):
        for _ in range(self.steps):
            # Keep empty Poisson draws. Resampling until non-empty changes the
            # sampling distribution assumed by the RDP accountant.
            if self.secure:
                # OS-backed uniform draws; threshold rounding is upward by <2^-32,
                # so the actual inclusion probability does not exceed accounted q.
                u = np.frombuffer(secrets.token_bytes(4 * self.n), dtype=np.uint32)
                threshold = math.floor(self.q * (1 << 32))
                idx = np.flatnonzero(u < threshold)
            else:
                idx = np.flatnonzero(self.rng.random(self.n) < self.q)
            yield idx.tolist()

    def __len__(self):
        return self.steps


def poisson_collate(batch):
    """Default DataLoader collation rejects empty batches; represent one explicitly."""
    if not batch:
        return torch.empty((0, 3, 224, 224)), torch.empty((0,), dtype=torch.long)
    return torch.utils.data.default_collate(batch)


def find_sigma_for_epsilon(q, steps, delta, target_eps, base_rdp=None,
                           lo=0.1, hi=50.0, tol=0.01, max_iter=60):
    from privacy import compute_rdp, rdp_to_eps
    if target_eps <= 0:
        raise ValueError("target_epsilon must be positive")
    def composed_eps(sigma):
        rdp = np.asarray(compute_rdp(q, sigma, steps), dtype=float)
        if base_rdp is not None:
            rdp += np.asarray(base_rdp, dtype=float)
        return rdp_to_eps(rdp, delta)
    if composed_eps(hi) > target_eps:
        raise ValueError("target_epsilon is below the already-spent budget or cannot be reached within sigma search bounds")
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        eps = composed_eps(mid)
        if abs(eps - target_eps) < tol:
            return mid, eps
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2
    return mid, composed_eps(mid)


def main():
    a = parse_args()
    run_dir = Path(a.run_dir)
    device = pick_device(a.device)
    set_seed(a.seed + 1000 * a.round + sum(map(ord, a.client_id)))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ---- 1. Mo hinh toan cuc ------------------------------------------------------------
    gpath = Path(a.global_model) if a.global_model else run_dir / "global" / f"global_round_{a.round - 1:02d}.pt"
    state, g_round = load_global(gpath)
    if g_round != a.round - 1:
        print(f"[!] Canh bao: file global la round {g_round}, ban dang chay round {a.round}")
    model = build_model(pretrained=False)
    model.load_state_dict(state)
    model.to(device)
    start, names = prepare_for_local_training(model, a.train_mode)
    names = set(names)
    trainable = {n: p for n, p in model.named_parameters() if n in names}
    global_ref = {n: p.detach().clone() for n, p in trainable.items()}
    n_params = sum(p.numel() for p in trainable.values())

    # ---- 2. Du lieu cuc bo --------------------------------------------------------------
    samples = [(p, l) for p, l in load_json(a.manifest)["samples"]] if a.manifest else list_class_folder(a.data_dir)
    t0 = time.time()
    ds = CachedImageDataset(samples, train_transform())
    N = len(ds)
    counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
    print(f"[Client {a.client_id}] round {a.round} | {device} | {N} mau | phan bo nhan {counts.tolist()} "
          f"| train_mode={a.train_mode} ({n_params:,} tham so) | cache anh {time.time()-t0:.1f}s")

    # Balanced-Softmax: cong log-prior vao logit luc huan luyen (Laplace smoothing cho nhan vang mat)
    prior = (counts + 1.0) / (N + NUM_CLASSES)
    logadj = (a.tau * torch.log(torch.tensor(prior, dtype=torch.float32))).to(device) if a.loss == "balanced" \
        else torch.zeros(NUM_CLASSES, device=device)

    steps_per_epoch = math.ceil(N / a.batch_size)
    q = min(1.0, a.batch_size / N)
    pin = device.type == "cuda"
    if a.dp:
        loader = DataLoader(ds, batch_sampler=PoissonBatchSampler(N, q, steps_per_epoch, a.seed + a.round,
                                                                  secure=a.secure_rng),
                            num_workers=a.workers, pin_memory=pin, collate_fn=poisson_collate)
        ledger = ClientPrivacyLedger(Path(a.state_dir or run_dir / "client_state") / f"client_{a.client_id}_dp.json")
        if a.target_epsilon is not None:
            from privacy import ORDERS
            total_rounds = a.planned_rounds or a.round
            if total_rounds < a.round:
                raise ValueError("planned_rounds cannot be less than the current round")
            remaining_steps = a.epochs * steps_per_epoch * (total_rounds - a.round + 1)
            base_rdp = (np.sum([np.asarray(v, dtype=float) for v in ledger.rounds.values()], axis=0)
                        if ledger.rounds else np.zeros(len(ORDERS)))
            a.noise_multiplier, eps_est = find_sigma_for_epsilon(
                q, remaining_steps, a.delta, a.target_epsilon, base_rdp=base_rdp)
            print(f"[Client {a.client_id}] whole_run_target_eps={a.target_epsilon} -> sigma={a.noise_multiplier:.3f} "
                  f"(projected cumulative eps={eps_est:.3f}; rounds remaining={total_rounds-a.round+1})")
        privacy_event_id = f"{run_dir.resolve()}:{a.round}:{uuid.uuid4().hex}"
        eps_proj = ledger.epsilon(a.delta, extra=(privacy_event_id, q, a.noise_multiplier,
                                                  a.epochs * steps_per_epoch))
        print(f"[Client {a.client_id}] DP-SGD: sigma={a.noise_multiplier}, C={a.max_grad_norm}, q={q:.4f}, "
              f"du toan epsilon tich luy sau round nay = {eps_proj:.2f} (delta={a.delta})")
        if eps_proj > a.max_epsilon:
            raise SystemExit(f"[Client {a.client_id}] Vuot ngan sach quyen rieng tu ({eps_proj:.2f} > {a.max_epsilon}). Dung.")
    else:
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=pin)

    noise_gen = None
    if a.dp and a.secure_rng:
        from privacy import secure_rng_seed
        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(secure_rng_seed())
        print(f"[Client {a.client_id}] Poisson sampling uses OS-backed randomness; Gaussian noise uses PyTorch PRNG seeded from OS entropy (not a cryptographic stream)")
    elif a.dp:
        print(f"[Client {a.client_id}] WARNING: deterministic Poisson sampling; Gaussian noise uses the seeded PyTorch PRNG")
    opt = torch.optim.AdamW(list(trainable.values()), lr=a.lr, weight_decay=a.weight_decay)
    amp = device.type == "cuda" and not a.no_amp

    def frozen_forward(x):
        """Phan dong bang (public, da pretrain): khong gradient, co the dung fp16."""
        if start == 0:
            return x
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            return model.forward_prefix(x, start).float()

    def add_prox():
        # FedProx: grad += mu * (w - w_global). Chi phu thuoc vao trong so (khong phu thuoc du lieu
        # truc tiep) nen khong ton them ngan sach DP khi ap dung SAU buoc lam nhieu.
        if a.mu > 0:
            for n, p in trainable.items():
                p.grad.add_(p.detach() - global_ref[n], alpha=a.mu)

    if a.dp:
        def loss_one(prm, h1, y1):
            out = functional_call(model, prm, (h1.unsqueeze(0),), {"start": start})
            return F.cross_entropy(out + logadj, y1.unsqueeze(0))
        per_sample = vmap(grad_and_value(loss_one), in_dims=(None, 0, 0), randomness="different")
        if a.auto_clip:
            if not (0.5 <= a.auto_clip_quantile <= 0.99) or a.auto_clip_batches < 1:
                raise ValueError("auto_clip_quantile must be 0.5..0.99 and batches >= 1")
            print(f"[Client {a.client_id}] Calibrating clip norm from {a.auto_clip_batches} batches")
            calib_norms = []
            for bi, (cx, cy) in enumerate(loader):
                if bi >= a.auto_clip_batches: break
                if len(cy) == 0: continue
                ch = frozen_forward(cx.to(device, non_blocking=True))
                cp = {n: p.detach() for n, p in trainable.items()}
                cg, _ = per_sample(cp, ch, cy.to(device, non_blocking=True))
                calib_norms.append(torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in cg.values())).cpu())
            if not calib_norms:
                raise RuntimeError("No batches available for clipping calibration")
            qval = torch.quantile(torch.cat(calib_norms), a.auto_clip_quantile).item()
            a.max_grad_norm = max(qval, 1e-6)
            print(f"[Client {a.client_id}] calibrated C={a.max_grad_norm:.6g}")
            loader = DataLoader(ds, batch_sampler=PoissonBatchSampler(N, q, steps_per_epoch, a.seed + a.round + 99991,
                                                                      secure=a.secure_rng),
                                num_workers=a.workers, pin_memory=pin, collate_fn=poisson_collate)

    # ---- 3. Huan luyen cuc bo -----------------------------------------------------------
    t0 = time.time()
    last_loss = 0.0
    for ep in range(a.epochs):
        tot, nb, clipped = 0.0, 0, 0.0
        for x, y in loader:
            # An empty Poisson draw is a no-op, but still counts as a sampled step.
            if len(y) == 0:
                nb += 1
                continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            h = frozen_forward(x)
            if a.dp:
                params = {n: p.detach() for n, p in trainable.items()}
                grads, losses = per_sample(params, h, y)
                norms = torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in grads.values()))
                factor = (a.max_grad_norm / (norms + 1e-6)).clamp(max=1.0)
                for n, p in trainable.items():
                    g = torch.einsum("b,b...->...", factor, grads[n])
                    noise = torch.randn(g.shape, generator=noise_gen, device=g.device, dtype=g.dtype) if noise_gen is not None else torch.randn_like(g)
                    g = g + noise * (a.noise_multiplier * a.max_grad_norm)
                    p.grad = g / a.batch_size          # chia cho kich thuoc batch ky vong
                add_prox()
                opt.step()
                loss_val = losses.mean().item()
                clipped += (norms > a.max_grad_norm).float().mean().item()
            else:
                loss = F.cross_entropy(model(h, start=start) + logadj, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                add_prox()
                opt.step()
                loss_val = loss.item()
            tot += loss_val
            nb += 1
        last_loss = tot / max(nb, 1)
        extra = f" | ti le bi clip {100*clipped/max(nb,1):.0f}%" if a.dp else ""
        print(f"   [Epoch {ep+1}/{a.epochs}] loss {last_loss:.4f}{extra}")
    print(f"[Client {a.client_id}] huan luyen xong trong {time.time()-t0:.1f}s")

    # ---- 4. Ghi file cap nhat (KHONG chua du lieu goc) ----------------------------------
    eps = -1.0
    if a.dp:
        ledger.record(privacy_event_id, q, a.noise_multiplier, a.epochs * steps_per_epoch)
        eps = ledger.epsilon(a.delta)
        print(f"[Client {a.client_id}] epsilon tich luy (delta={a.delta}) = {eps:.3f}")
        if a.verify_dp:
            from privacy import verify_accountant
            chk = verify_accountant(q, a.noise_multiplier, a.epochs * steps_per_epoch, a.delta)
            print(f"[Client {a.client_id}] Accountant check: ours={chk['our_eps']:.3f}, Opacus={chk['opacus_eps']} [{chk['ok']}]")

    delta_w = {n: (p.detach().cpu() - global_ref[n].cpu()).contiguous() for n, p in trainable.items()}
    payload = {
        "format": 1, "client_id": a.client_id, "round": a.round, "num_samples": int(N),
        "train_mode": a.train_mode, "delta": delta_w, "train_loss": float(last_loss),
        "fedprox_mu": float(a.mu), "dp_enabled": bool(a.dp),
        "dp_noise_multiplier": float(a.noise_multiplier if a.dp else 0.0),
        "dp_max_grad_norm": float(a.max_grad_norm if a.dp else 0.0),
        "dp_epsilon": float(eps), "dp_delta": float(a.delta if a.dp else 0.0),
    }
    out = run_dir / "updates" / f"round_{a.round:02d}" / f"client_{a.client_id}.pt"
    atomic_save(payload, out)
    print(f"[Client {a.client_id}] DA GHI file cap nhat: {out}  ({out.stat().st_size/1e6:.2f} MB)")
    print("   -> Gui file nay cho server (khong co du lieu anh).")


if __name__ == "__main__":
    main()
