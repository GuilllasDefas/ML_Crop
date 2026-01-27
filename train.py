# train.py
"""
Treino do localizador de crop (versão com GIoU + metadata para inferência).
Uso:
  python train.py                   # tenta dataset/origin + dataset/cropped
  python train.py --origins X --crops Y
Saídas:
  models/crop_localizer.pt    (melhor por val IoU)
  models/last_crop_localizer.pt
  worst_iou.json
"""
import os, json, random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2, numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models

# ---------- utils ----------
def xyxy_from_corners(corners: np.ndarray, H:int, W:int) -> Optional[np.ndarray]:
    xs, ys = corners[:,0], corners[:,1]
    x1,y1,x2,y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    x1 = max(0.0, min(W-1.0, x1)); x2 = max(0.0, min(W-1.0, x2))
    y1 = max(0.0, min(H-1.0, y1)); y2 = max(0.0, min(H-1.0, y2))
    if x2 <= x1 or y2 <= y1: return None
    return np.array([x1,y1,x2,y2], dtype=np.float32)

def compute_bbox_orb(origin_path: Path, crop_path: Path) -> Optional[np.ndarray]:
    origin = cv2.imread(str(origin_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if origin is None or crop is None: return None
    H,W = origin.shape[:2]
    orb = cv2.ORB_create(1200)
    k1,d1 = orb.detectAndCompute(origin, None)
    k2,d2 = orb.detectAndCompute(crop, None)
    if d1 is None or d2 is None: return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = bf.match(d2, d1)
    except Exception:
        return None
    if len(matches) < 6: return None
    pts_crop = np.float32([k2[m.queryIdx].pt for m in matches])
    pts_origin = np.float32([k1[m.trainIdx].pt for m in matches])
    M, mask = cv2.estimateAffinePartial2D(pts_crop, pts_origin, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if M is None: return None
    h,w = crop.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    transformed = cv2.transform(corners, M).reshape(-1,2)
    xyxy = xyxy_from_corners(transformed, H, W)
    if xyxy is None: return None
    x1,y1,x2,y2 = xyxy
    return np.array([x1/W, y1/H, x2/W, y2/H], dtype=np.float32)

def compute_bbox_template(origin_path: Path, crop_path: Path) -> Optional[np.ndarray]:
    origin = cv2.imread(str(origin_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if origin is None or crop is None: return None
    res = cv2.matchTemplate(origin, crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.45: return None
    H,W = origin.shape[:2]; h,w = crop.shape[:2]
    x1,y1 = max_loc; x2,y2 = x1+w, y1+h
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(W-1,x2); y2 = min(H-1,y2)
    return np.array([x1/W, y1/H, x2/W, y2/H], dtype=np.float32)

# ---------- dataset ----------
class CropLocalizationDataset(Dataset):
    def __init__(self, origin_dir: str, crop_dir: str, resize: Tuple[int,int]=(320,320), augment: bool=False):
        self.origin_dir = Path(origin_dir)
        self.crop_dir = Path(crop_dir)
        self.resize = resize
        self.augment = augment
        self.samples: List[Tuple[Path,Path,np.ndarray]] = []
        self._build_index()
        # apenas photometric augmentations (NÃO alterar geometria sem ajustar bbox)
        trans = [T.Resize(self.resize)]
        if self.augment:
            trans.append(T.ColorJitter(0.12,0.12,0.12,0.02))
        trans.append(T.ToTensor()) # [0,1]
        self.base_transform = T.Compose(trans)

    def _build_index(self):
        origins = sorted([p for p in self.origin_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]) if self.origin_dir.exists() else []
        crops = sorted([p for p in self.crop_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]) if self.crop_dir.exists() else []
        crop_map = {p.name: p for p in crops}
        for o in origins:
            c = crop_map.get(o.name)
            if c is None:
                candidates = [p for p in crops if p.stem in o.stem or o.stem in p.stem]
                c = candidates[0] if candidates else None
            if c is None: continue
            bbox = compute_bbox_orb(o,c)
            if bbox is None:
                bbox = compute_bbox_template(o,c)
            if bbox is None: continue
            self.samples.append((o,c,bbox))
        if len(self.samples) == 0:
            raise RuntimeError(f"Nenhuma amostra válida encontrada em {self.origin_dir} / {self.crop_dir}.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        origin_path, _, bbox = self.samples[idx]
        img = Image.open(origin_path).convert("RGB")
        img_t = self.base_transform(img)
        return img_t, torch.from_numpy(bbox).float()

# ---------- stats ----------
def compute_dataset_mean_std(dataset: CropLocalizationDataset, sample_limit: int = 500):
    import numpy as np
    n = min(len(dataset), sample_limit)
    idxs = random.sample(range(len(dataset)), n)
    sums = np.zeros(3); sums_sq = np.zeros(3); px = 0
    for i in idxs:
        img_t, _ = dataset[i]
        arr = img_t.numpy()
        h,w = arr.shape[1], arr.shape[2]; px += h*w
        sums += arr.reshape(3, -1).sum(axis=1)
        sums_sq += (arr.reshape(3, -1)**2).sum(axis=1)
    mean = sums/px; var = (sums_sq/px) - (mean**2); std = np.sqrt(np.maximum(var,1e-6))
    return mean.tolist(), std.tolist()

def compute_bbox_stats(dataset: CropLocalizationDataset):
    arr = np.stack([s[2] for s in dataset.samples], axis=0)
    widths = (arr[:,2]-arr[:,0]); heights = (arr[:,3]-arr[:,1])
    stats = {"width_p": [float(np.percentile(widths, p)) for p in [50,75,90,95,99]],
             "height_p":[float(np.percentile(heights,p)) for p in [50,75,90,95,99]],
             "width_mean": float(widths.mean()), "height_mean": float(heights.mean())}
    return stats

# ---------- model ----------
def build_model(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f,128), nn.ReLU(inplace=True), nn.Linear(128,4), nn.Sigmoid())
    return m

# ---------- GIoU ----------
def giou_tensor(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    # pred, target: N x 4 (xyxy normalized)
    px1, py1, px2, py2 = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    tx1, ty1, tx2, ty2 = target[:,0], target[:,1], target[:,2], target[:,3]
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    cw = (cx2 - cx1).clamp(min=0); ch = (cy2 - cy1).clamp(min=0)
    convex = cw * ch + eps
    giou = iou - (convex - union) / convex
    return giou.clamp(min=-1.0, max=1.0)

# ---------- training loop ----------
def train_loop(train_loader, val_loader, device, epochs=12, lr=1e-4, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # Removed verbose argument for compatibility with older torch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    crit = nn.SmoothL1Loss()
    best_val = 0.0
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep} train")
        run = 0.0
        for imgs, targets in pbar:
            imgs = imgs.to(device); targets = targets.to(device)
            preds = model(imgs).clamp(0.0,1.0)
            l1 = crit(preds, targets)
            g = giou_tensor(preds, targets).mean()
            giou_loss = 1.0 - g
            loss = 0.6 * l1 + 0.4 * giou_loss
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss.item())
            pbar.set_postfix({"loss": run/(pbar.n+1)})
        val_iou, worst = validate(model, val_loader, device)
        print(f"Epoch {ep} val_iou={val_iou:.4f}")
        scheduler.step(val_iou)
        ck = {"model_state": model.state_dict(), "metadata": {"resize": train_loader.dataset.dataset.resize if hasattr(train_loader.dataset,'dataset') else train_loader.dataset.resize}}
        torch.save(ck, os.path.join(save_dir, "last_crop_localizer.pt"))
        if val_iou > best_val:
            best_val = val_iou
            torch.save(ck, os.path.join(save_dir, "crop_localizer.pt"))
        with open("worst_iou.json","w",encoding="utf-8") as fh:
            json.dump({"worst":[{"iou":float(i),"path":p} for i,p in worst[:200]]}, fh, indent=2, ensure_ascii=False)
    print("Treino finalizado. Best val IoU:", best_val)

def validate(model, val_loader, device):
    model.eval()
    ious=[]; worst=[]
    with torch.no_grad():
        idx=0
        dataset_obj = val_loader.dataset.dataset if hasattr(val_loader.dataset,"dataset") else val_loader.dataset
        samples = getattr(dataset_obj, "samples", None)
        for imgs, targets in val_loader:
            imgs = imgs.to(device); targets = targets.to(device)
            preds = model(imgs).clamp(0.0,1.0)
            # compute IoU
            px1,py1,px2,py2 = preds[:,0],preds[:,1],preds[:,2],preds[:,3]
            tx1,ty1,tx2,ty2 = targets[:,0],targets[:,1],targets[:,2],targets[:,3]
            ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
            ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
            iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
            inter = (iw*ih).cpu().numpy()
            area_p = ((px2-px1).clamp(min=0)*(py2-py1).clamp(min=0)).cpu().numpy()
            area_t = ((tx2-tx1).clamp(min=0)*(ty2-ty1).clamp(min=0)).cpu().numpy()
            union = area_p + area_t - inter + 1e-6
            iou_batch = (inter/union).tolist()
            for b,iou in enumerate(iou_batch):
                ious.append(float(iou))
                if samples is not None:
                    origin_path = samples[idx + b][0]
                    worst.append((float(iou), str(origin_path)))
            idx += len(iou_batch)
    mean_iou = float(np.mean(ious)) if len(ious)>0 else 0.0
    worst.sort(key=lambda x:x[0])
    return mean_iou, worst

# ---------- main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins"); parser.add_argument("--crops")
    parser.add_argument("--resize", type=int, nargs=2, default=(400,400))
    parser.add_argument("--epochs", type=int, default=25); parser.add_argument("--batch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--val-split", type=float, default=0.12)
    args = parser.parse_args()

    origins = Path(args.origins) if args.origins else Path("dataset/origin")
    crops = Path(args.crops) if args.crops else Path("dataset/cropped")
    if not origins.exists() or not crops.exists():
        print("Diretórios padrão não encontrados. Verifique dataset/origin e dataset/cropped ou passe --origins/--crops")
        return

    print(f"Usando origins = {origins}\nUsando crops   = {crops}")
    ds = CropLocalizationDataset(str(origins), str(crops), resize=tuple(args.resize), augment=False)
    mean,std = compute_dataset_mean_std(ds, sample_limit=500)
    bbox_stats = compute_bbox_stats(ds)

    # rebuild transform with normalization (no geometric augment)
    t = [T.Resize(tuple(args.resize)), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    ds.base_transform = T.Compose(t)

    n = len(ds); idxs = list(range(n)); random.shuffle(idxs)
    split = int(n * args.val_split)
    val_idxs = idxs[:split] if split>0 else []
    train_idxs = idxs[split:] if split < n else idxs
    train_ds = Subset(ds, train_idxs)
    val_ds = Subset(ds, val_idxs) if len(val_idxs)>0 else Subset(ds, train_idxs[: max(1, len(train_idxs)//10)])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    os.makedirs("models", exist_ok=True)
    metadata = {"resize": tuple(args.resize), "mean": mean, "std": std, "bbox_stats": bbox_stats, "output_format": "xyxy"}
    torch.save({"model_state": build_model(pretrained=True).state_dict(), "metadata": metadata}, os.path.join("models","init_crop_localizer.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, save_dir="models")

if __name__ == "__main__":
    main()
