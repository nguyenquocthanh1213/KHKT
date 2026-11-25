"""
common.py - Tien ich dung chung cho client / server / evaluate.

Thiet ke chinh:
  * FLNet: MobileNetV3-Small duoc "cat" thanh danh sach tang (stage) de co the dong bang
    phan dau (khong tinh gradient) va chi huan luyen phan duoi (suffix).
  * BatchNorm LUON o che do eval (dung thong ke ImageNet da hoc): tranh hien tuong BN-statistics
    bi lech theo tung client khi du lieu Non-IID cuc doan, dong thoi tuong thich per-sample
    gradient cua DP-SGD (BN o che do train tron cac mau trong batch -> khong the tinh DP).
  * Chi trao doi file chua tensor/so/chuoi va doc bang torch.load(weights_only=True)
    -> khong the chay ma doc hai tu file client gui len.
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ----------------------------------------------------------------------------------------
# Hang so
# ----------------------------------------------------------------------------------------
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # thu tu nhan co dinh
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Chi so "tang" dau tien duoc huan luyen.
# Tang: features[0..12] -> 0..12 | pool+flatten -> 13 | classifier[0..3] -> 14..17
TRAIN_MODES = {
    "linear": 17,       # chi lop Linear cuoi (4.1k tham so)  - hop nhat voi DP
    "head": 14,         # toan bo classifier (~0.6M)
    "last_blocks": 10,  # features[10:] + classifier (~1.2M)
    "all": 0,           # toan mang (~1.5M+)
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# ----------------------------------------------------------------------------------------
# Mo hinh
# ----------------------------------------------------------------------------------------
class FLNet(nn.Module):
    """MobileNetV3-Small chia tang de dong bang prefix / huan luyen suffix."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.features = base.features
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.classifier = base.classifier

    def stages(self):
        return list(self.features) + [self.pool] + list(self.classifier)

    def forward_prefix(self, x, start: int):
        for m in self.stages()[:start]:
            x = m(x)
        return x

    def forward(self, x, start: int = 0):
        for m in self.stages()[start:]:
            x = m(x)
        return x


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> FLNet:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    base = mobilenet_v3_small(weights=weights)
    base.classifier[3] = nn.Linear(base.classifier[3].in_features, num_classes)
    return FLNet(base)


def stage_index_of(param_name: str) -> int:
    parts = param_name.split(".")
    if parts[0] == "features":
        return int(parts[1])
    if parts[0] == "classifier":
        return 14 + int(parts[1])
    raise ValueError(param_name)


def trainable_names(model: nn.Module, mode: str):
    start = TRAIN_MODES[mode]
    return [n for n, _ in model.named_parameters() if stage_index_of(n) >= start]


def prepare_for_local_training(model: FLNet, mode: str):
    """Dong bang phan khong huan luyen, BN o eval, chi bat Dropout cua suffix."""
    start = TRAIN_MODES[mode]
    names = set(trainable_names(model, mode))
    for n, p in model.named_parameters():
        p.requires_grad_(n in names)
    model.eval()
    for m in model.stages()[start:]:
        for sub in m.modules():
            if isinstance(sub, nn.Dropout):
                sub.train()
    return start, [n for n in names]


# ----------------------------------------------------------------------------------------
# IO an toan (chi tensor / so / chuoi)
# ----------------------------------------------------------------------------------------
def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_global(path):
    ckpt = safe_load(path)
    return ckpt["state_dict"], int(ckpt["round"])


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------
# Du lieu
# ----------------------------------------------------------------------------------------
def list_class_folder(root):
    """root/<class_name>/*.jpg -> [(duong_dan, nhan)] theo CLASS_TO_IDX co dinh."""
    root = Path(root)
    samples = []
    for cname in CLASSES:
        d = root / cname
        if not d.is_dir():
            raise FileNotFoundError(f"Khong thay thu muc lop: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), CLASS_TO_IDX[cname]))
    return samples


def eval_transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_transform():
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CachedImageDataset(Dataset):
    """Doc + resize anh ve 224x224 MOT LAN, giu trong RAM (nhanh hon nhieu so voi doc lai
    dia moi epoch). Augmentation chay tren anh PIL da cache."""

    def __init__(self, samples, transform):
        self.labels = [int(l) for _, l in samples]
        self.transform = transform
        self.images = []
        for path, _ in samples:
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def make_eval_loader(samples, batch_size=128, workers=0):
    ds = CachedImageDataset(samples, eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)


# ----------------------------------------------------------------------------------------
# Danh gia
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        ps.append(out.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(float)
    support = cm.sum(1).astype(float)
    pred_cnt = cm.sum(0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_cnt, out=np.zeros_like(tp), where=pred_cnt > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }
