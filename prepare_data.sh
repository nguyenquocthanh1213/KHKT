#!/usr/bin/env bash
# prepare_data.sh - Tu dong chuan bi du lieu cho he thong Federated Learning.
#
#   1. Tai Brain Tumor MRI Dataset (Kaggle) bang curl      (bo qua neu da co du lieu / file zip hop le)
#   2. Giai nen, chuan hoa cau truc ve  <DATA_DIR>/Training/<lop>/  va  <DATA_DIR>/Testing/<lop>/
#   3. Kiem tra: du 4 lop, dem anh, phat hien anh hong
#   4. Chia Training theo kich ban Non-IID cuc doan -> ./partitions (manifest cho tung client + validation toan cuc)
#
# Chay:   bash prepare_data.sh
# Tuy chinh qua bien moi truong, vi du:
#   CLIENTS="A:glioma B:meningioma C:notumor D:pituitary" SAMPLES_PER_CLIENT=1000 bash prepare_data.sh
#   REPARTITION=1 bash prepare_data.sh        # chia lai du lieu (vd sau khi doi CLIENTS)
# Neu tai tu dong that bai (Kaggle doi hoi dang nhap...), tai zip thu cong tu trang Kaggle,
# dat vao ./brain-tumor-mri-dataset.zip roi chay lai script.
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-./brain-tumor-mri-dataset}"
ZIP_FILE="${ZIP_FILE:-./brain-tumor-mri-dataset.zip}"
DATASET_URL="${DATASET_URL:-https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset}"
PARTITION_DIR="${PARTITION_DIR:-./partitions}"
CLIENTS="${CLIENTS:-A:glioma B:pituitary}"       # <id>:<lop chu dao>
SAMPLES_PER_CLIENT="${SAMPLES_PER_CLIENT:-1200}"
DOMINANT_FRAC="${DOMINANT_FRAC:-0.9}"            # 90% lop chu dao, phan con lai chia deu
VAL_PER_CLASS="${VAL_PER_CLASS:-100}"            # validation toan cuc tach tu Training
SEED="${SEED:-42}"
REPARTITION="${REPARTITION:-0}"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3; elif command -v python >/dev/null 2>&1; then PY=python
  else echo "[-] Khong tim thay Python (can python3)."; exit 1; fi
fi
command -v curl >/dev/null 2>&1 || { echo "[-] Can cai 'curl'."; exit 1; }

have_dataset() {
  for s in Training Testing; do
    for c in glioma meningioma notumor pituitary; do
      [ -d "$DATA_DIR/$s/$c" ] || return 1
    done
  done
}

# ---- 1. Tai ---------------------------------------------------------------------------
if have_dataset; then
  echo "[*] Da co du lieu tai $DATA_DIR -> bo qua buoc tai/giai nen."
else
  if [ -f "$ZIP_FILE" ] && "$PY" -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$ZIP_FILE"; then
    echo "[*] Da co file zip hop le: $ZIP_FILE -> bo qua buoc tai."
  else
    rm -f "$ZIP_FILE"
    echo "[*] Dang tai dataset (~150 MB) ..."
    curl -L -f --retry 3 --retry-delay 3 -o "$ZIP_FILE" "$DATASET_URL" || {
      rm -f "$ZIP_FILE"
      echo "[-] Tai that bai. Hay tai thu cong tu Kaggle, luu thanh $ZIP_FILE roi chay lai."; exit 1; }
  fi
fi

# ---- 2 + 3. Giai nen, chuan hoa cau truc, kiem tra -----------------------------------
"$PY" - "$ZIP_FILE" "$DATA_DIR" <<'PY'
import shutil, sys, zipfile
from pathlib import Path

zip_path, data_dir = Path(sys.argv[1]), Path(sys.argv[2])
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_root(base):
    if (base / "Training").is_dir() and (base / "Testing").is_dir():
        return base
    for t in sorted(base.rglob("Training")):
        if "__MACOSX" not in t.parts and t.is_dir() and (t.parent / "Testing").is_dir():
            return t.parent
    return None


root = find_root(data_dir) if data_dir.exists() else None
if root is None:
    if not zipfile.is_zipfile(zip_path):
        sys.exit(f"[-] {zip_path} khong phai file zip hop le (co the Kaggle tra ve trang dang nhap). Hay tai thu cong.")
    print(f"[*] Giai nen {zip_path} -> {data_dir} ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    root = find_root(data_dir)
    if root is None:
        sys.exit("[-] Khong thay thu muc Training/ va Testing/ trong file zip.")
if root != data_dir:  # zip co them thu muc boc ngoai -> dua len cap DATA_DIR
    for s in ("Training", "Testing"):
        shutil.move(str(root / s), str(data_dir / s))

from PIL import Image  # noqa: E402

bad, table = [], {}
for split in ("Training", "Testing"):
    for c in CLASSES:
        d = data_dir / split / c
        if not d.is_dir():
            sys.exit(f"[-] Thieu thu muc: {d}")
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
        if not files:
            sys.exit(f"[-] Thu muc rong: {d}")
        for p in files:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                bad.append(str(p))
        table[(split, c)] = len(files)
if bad:
    print("[-] Anh hong:\n   " + "\n   ".join(bad[:20]) + (f"\n   ... (+{len(bad)-20})" if len(bad) > 20 else ""))
    sys.exit("    Xoa/thay cac file tren roi chay lai.")

print(f"\n[+] Du lieu hop le tai {data_dir.resolve()}")
print(f"    {'lop':14s}{'Training':>10s}{'Testing':>10s}")
for c in CLASSES:
    print(f"    {c:14s}{table[('Training', c)]:>10d}{table[('Testing', c)]:>10d}")
print(f"    {'TONG':14s}{sum(table[('Training', c)] for c in CLASSES):>10d}"
      f"{sum(table[('Testing', c)] for c in CLASSES):>10d}")
PY

# ---- 4. Chia du lieu theo kich ban Non-IID cuc doan -----------------------------------
if [ "$REPARTITION" != "1" ] && [ -f "$PARTITION_DIR/global_val.json" ]; then
  echo "[*] Da co $PARTITION_DIR -> giu nguyen (dung REPARTITION=1 de chia lai)."
else
  read -r -a CLIENT_ARR <<< "$CLIENTS"
  "$PY" partition.py --data_root "$DATA_DIR" --out_dir "$PARTITION_DIR" --clients "${CLIENT_ARR[@]}" \
        --samples_per_client "$SAMPLES_PER_CLIENT" --dominant_frac "$DOMINANT_FRAC" \
        --val_per_class "$VAL_PER_CLASS" --seed "$SEED"
fi

cat <<EOF

[+] SAN SANG. Cac buoc tiep theo:
    python run_simulation.py --data_root $DATA_DIR --partition_dir $PARTITION_DIR --tag fedprox_dp --mu 0.05 --dp --train_mode head
    (hoac chay le: python server.py init ... / python client.py --manifest $PARTITION_DIR/client_A.json ...)
    python evaluate.py --run_dir runs/<tag> --data_root $DATA_DIR --all_rounds
EOF
