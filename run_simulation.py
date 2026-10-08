"""
run_simulation.py - Chay tron ven mot thi nghiem tren MOT may, mo phong quy trinh trao doi file.

Moi client duoc chay bang mot tien trinh rieng (python client.py ...), server bang
python server.py ... -> dung y het quy trinh that, chi khac la file "gui" qua thu muc chung.

Vi du:
  python run_simulation.py --data_root ./brain-tumor-mri-dataset --tag fedprox_dp \
         --rounds 10 --mu 0.05 --dp --noise_multiplier 1.0 --train_mode head
  python run_simulation.py --data_root ... --tag fedavg --mu 0          # doi chung
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def sh(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--tag", required=True, help="Ten thi nghiem -> runs/<tag>")
    ap.add_argument("--runs_dir", default="./runs")
    ap.add_argument("--partition_dir", default="./partitions")
    ap.add_argument("--clients", nargs="+", default=["A:glioma", "B:pituitary"])
    ap.add_argument("--samples_per_client", type=int, default=1200)
    ap.add_argument("--dominant_frac", type=float, default=0.9)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train_mode", default="last_blocks")
    ap.add_argument("--loss", default="balanced")
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--noise_multiplier", type=float, default=1.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--max_epsilon", type=float, default=float("inf"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--repartition", action="store_true", help="Chia lai du lieu du da co manifest")
    ap.add_argument("--skip_eval", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    run_dir = Path(a.runs_dir) / a.tag
    pdir = Path(a.partition_dir)
    client_ids = [c.split(":")[0] for c in a.clients]

    if a.repartition or not (pdir / "global_val.json").exists():
        sh([py, HERE / "partition.py", "--data_root", a.data_root, "--out_dir", pdir, "--clients", *a.clients,
            "--samples_per_client", a.samples_per_client, "--dominant_frac", a.dominant_frac, "--seed", a.seed])

    sh([py, HERE / "server.py", "init", "--run_dir", run_dir, "--seed", a.seed]
       + (["--no_pretrained"] if a.no_pretrained else []))

    for r in range(1, a.rounds + 1):
        print(f"\n================ ROUND {r}/{a.rounds} ================")
        for cid in client_ids:
            cmd = [py, HERE / "client.py", "--client_id", cid, "--round", r, "--run_dir", run_dir,
                   "--manifest", pdir / f"client_{cid}.json", "--train_mode", a.train_mode,
                   "--epochs", a.epochs, "--batch_size", a.batch_size, "--lr", a.lr, "--loss", a.loss,
                   "--mu", a.mu, "--seed", a.seed, "--device", a.device]
            if a.dp:
                cmd += ["--dp", "--noise_multiplier", a.noise_multiplier, "--max_grad_norm", a.max_grad_norm,
                        "--delta", a.delta, "--max_epsilon", a.max_epsilon]
            sh(cmd)
        sh([py, HERE / "server.py", "aggregate", "--run_dir", run_dir, "--round", r,
            "--min_clients", len(client_ids)])

    if not a.skip_eval:
        sh([py, HERE / "evaluate.py", "--run_dir", run_dir, "--data_root", a.data_root,
            "--all_rounds", "--device", a.device])
        sh([py, HERE / "evaluate.py", "--run_dir", run_dir, "--split", "val",
            "--partition_dir", pdir, "--device", a.device, "--no_plots"])


if __name__ == "__main__":
    main()
