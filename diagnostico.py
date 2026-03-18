import pickle
import torch
import numpy as np
from train import MarginAwareCropModel, CropDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Carregar cache
with open("models/bbox_cache.pkl", "rb") as f:
    cache = pickle.load(f)

orig_paths = cache["orig_paths"]
crop_paths = cache["crop_paths"]
bbox_data  = cache["bboxes"]

# Recriar o val_loader igual ao train.py
_, val_orig, _, val_crop, _, val_bbox = train_test_split(
    orig_paths, crop_paths, bbox_data, test_size=0.1, random_state=42
)

checkpoint = torch.load("models/best_model.pth", map_location="cpu")
IMG_SIZE = checkpoint.get("img_size", 320)

val_dataset = CropDataset(val_orig, val_crop, val_bbox, img_size=IMG_SIZE)
val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Carregar modelo
model = MarginAwareCropModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Coletar predições
all_preds, all_targets = [], []
with torch.no_grad():
    for inputs, targets in val_loader:
        preds = model(inputs)
        all_preds.append(preds.numpy())
        all_targets.append(targets.numpy())

preds_np   = np.concatenate(all_preds)
targets_np = np.concatenate(all_targets)

print("=== VARIÂNCIA DAS PREDIÇÕES ===")
for i, label in enumerate(['x1', 'y1', 'x2', 'y2']):
    pred_std   = preds_np[:, i].std()
    target_std = targets_np[:, i].std()
    ratio      = pred_std / (target_std + 1e-8)
    status     = "✅ Aprendendo" if ratio > 0.5 else "⚠️ Colapsando"
    print(f"  {label}: pred_std={pred_std:.4f} | target_std={target_std:.4f} | ratio={ratio:.2f} → {status}")