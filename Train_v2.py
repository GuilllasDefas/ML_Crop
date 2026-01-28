import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ DATASET OTIMIZADO (PRÉ-CARREGAMENTO COMPLETO) ============
class CropDataset(Dataset):
    def __init__(self, original_paths, cropped_paths, img_size=300):
        self.img_size = img_size
        self.pairs = []
        
        print("Pré-carregando e pré-processando dataset...")
        for orig_path, crop_path in tqdm(zip(original_paths, cropped_paths), total=len(original_paths)):
            try:
                orig = cv2.imread(orig_path)
                crop = cv2.imread(crop_path)
                if orig is None or crop is None:
                    continue
                
                orig_h, orig_w = orig.shape[:2]
                
                # Template matching com precisão subpixel para extrair bbox EXATO
                gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                # Usar método mais robusto para encontrar posição exata
                result = cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                x1, y1 = max_loc
                x2, y2 = x1 + crop.shape[1], y1 + crop.shape[0]
                
                # Normalizar coordenadas para [0,1]
                bbox_norm = np.array([
                    x1 / orig_w,
                    y1 / orig_h,
                    x2 / orig_w,
                    y2 / orig_h
                ], dtype=np.float32)
                
                # Pré-processar imagem original
                orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                orig_resized = cv2.resize(orig_rgb, (img_size, img_size))
                orig_tensor = transforms.ToTensor()(orig_resized)
                orig_tensor = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(orig_tensor)
                
                self.pairs.append({
                    'input': orig_tensor,
                    'bbox': bbox_norm,
                    'orig_size': np.array([orig_w, orig_h], dtype=np.float32)
                })
            except Exception as e:
                print(f"Erro ao processar {orig_path}: {e}")
                continue
        
        print(f"Dataset carregado: {len(self.pairs)} pares válidos")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]['input'], self.pairs[idx]['bbox']


# ============ MODELO COM ATENÇÃO EXTREMA ÀS MARGENS ============
class MarginAwareCropModel(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B0 como backbone
        try:
            # PyTorch >= 0.13
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except AttributeError:
            # PyTorch < 0.13
            backbone = models.efficientnet_b0(pretrained=True)
        
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Regressor com inicialização focada em margens
        self.regressor = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1280, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 4)  # (x1, y1, x2, y2) normalizados
        )
        
        # Inicialização inteligente para saída próxima de margens de 5-10%
        nn.init.constant_(self.regressor[-1].bias, 0.0)
        self.regressor[-1].bias.data[0] = 0.07  # x1 ~ 7%
        self.regressor[-1].bias.data[1] = 0.07  # y1 ~ 7%
        self.regressor[-1].bias.data[2] = 0.93  # x2 ~ 93%
        self.regressor[-1].bias.data[3] = 0.93  # y2 ~ 93%
    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        return torch.sigmoid(self.regressor(x))  # Forçar saída em [0,1]


# ============ LOSS FUNCTIONS ESPECIALIZADAS PARA MARGENS ============
def iou_loss(pred, target):
    """IoU loss robusto"""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    target = target.clamp(1e-6, 1 - 1e-6)
    
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    
    union = pred_area + target_area - inter + 1e-6
    iou = inter / union
    return 1 - iou.mean()

def margin_aware_loss(pred, target):
    """
    LOSS CRÍTICA: Penaliza DEVIATION DAS MARGENS RELATIVAS
    Foca na diferença percentual das margens em relação ao conteúdo
    """
    # Calcular margens normalizadas (esquerda, topo, direita, baixo)
    pred_margins = torch.stack([
        pred[:, 0],                    # esquerda
        pred[:, 1],                    # topo
        1.0 - pred[:, 2],              # direita
        1.0 - pred[:, 3]               # baixo
    ], dim=1)
    
    target_margins = torch.stack([
        target[:, 0],
        target[:, 1],
        1.0 - target[:, 2],
        1.0 - target[:, 3]
    ], dim=1)
    
    # Erro relativo das margens (normalizado pelo tamanho do conteúdo)
    content_width = pred[:, 2] - pred[:, 0]
    content_height = pred[:, 3] - pred[:, 1]
    
    # Penalização mais forte para margens pequenas (onde erro é mais perceptível)
    margin_error = torch.abs(pred_margins - target_margins)
    margin_error[:, 0] /= (content_width + 1e-6)  # margem esquerda relativa à largura
    margin_error[:, 2] /= (content_width + 1e-6)  # margem direita
    margin_error[:, 1] /= (content_height + 1e-6) # margem topo
    margin_error[:, 3] /= (content_height + 1e-6) # margem baixo
    
    return margin_error.mean()

def combined_loss(pred, target, alpha=0.5):
    """
    Combinação balanceada:
    - alpha * IoU loss (precisão do bbox)
    - (1-alpha) * Margin loss (precisão DAS MARGENS - PRIORIDADE)
    """
    return alpha * iou_loss(pred, target) + (1 - alpha) * margin_aware_loss(pred, target)


# ============ TREINAMENTO INTELIGENTE ============
def train():
    # Configurações
    IMG_SIZE = 300
    BATCH_SIZE = 24  # Ajustado para RTX 3060 Ti (8GB VRAM)
    NUM_WORKERS = max(6, os.cpu_count() // 2)
    EPOCHS = 100
    PATIENCE = 20  # Mais tolerante para convergência fina das margens
    
    # Carregar paths
    orig_dir = "dataset/origin"
    crop_dir = "dataset/cropped"
    
    if not os.path.exists(orig_dir) or not os.path.exists(crop_dir):
        raise FileNotFoundError(f"Pastas não encontradas. Estrutura esperada:\n  {orig_dir}\n  {crop_dir}")
    
    orig_files = sorted([os.path.join(orig_dir, f) for f in os.listdir(orig_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    crop_files = sorted([os.path.join(crop_dir, f) for f in os.listdir(crop_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and '_editado' in f.lower()])
    
    # Mapear pares por nome base
    orig_dict = {os.path.splitext(os.path.basename(f))[0].replace('_editado', ''): f for f in orig_files}
    crop_dict = {os.path.splitext(os.path.basename(f))[0].replace('_editado', ''): f for f in crop_files}
    common_names = set(orig_dict.keys()) & set(crop_dict.keys())
    
    orig_paths = [orig_dict[name] for name in common_names]
    crop_paths = [crop_dict[name] for name in common_names]
    
    print(f"Encontrados {len(common_names)} pares de imagens")
    if len(common_names) < 10:
        raise ValueError("Dataset muito pequeno! Mínimo recomendado: 50 pares")
    
    # Divisão treino/validação
    train_orig, val_orig, train_crop, val_crop = train_test_split(
        orig_paths, crop_paths, test_size=0.1, random_state=42
    )
    
    # Criar datasets (pré-carregamento completo)
    train_dataset = CropDataset(train_orig, train_crop, img_size=IMG_SIZE)
    val_dataset = CropDataset(val_orig, val_crop, img_size=IMG_SIZE)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset vazio após pré-processamento!")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2
    )
    
    # Modelo e otimizador
    model = MarginAwareCropModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Scheduler SEM verbose (compatível com todas versões)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6
    )
    
    # Mixed Precision para Tensor Cores
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping
    best_iou = 0.0
    patience_counter = 0
    best_margin_err = float('inf')
    
    # Treinamento
    for epoch in range(EPOCHS):
        # ===== TREINO =====
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(inputs)
                    loss = combined_loss(preds, targets, alpha=0.45)  # Dar mais peso às margens!
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(inputs)
                loss = combined_loss(preds, targets, alpha=0.45)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        # ===== VALIDAÇÃO =====
        model.eval()
        val_iou = 0.0
        val_margin_error = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        preds = model(inputs)
                else:
                    preds = model(inputs)
                
                # Calcular IoU
                x1 = torch.max(preds[:, 0], targets[:, 0])
                y1 = torch.max(preds[:, 1], targets[:, 1])
                x2 = torch.min(preds[:, 2], targets[:, 2])
                y2 = torch.min(preds[:, 3], targets[:, 3])
                
                inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
                pred_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
                target_area = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
                union = pred_area + target_area - inter + 1e-6
                iou = inter / union
                val_iou += iou.sum().item()
                
                # Calcular erro das margens
                margin_err = margin_aware_loss(preds, targets)
                val_margin_error += margin_err.item() * inputs.size(0)
        
        avg_iou = val_iou / len(val_dataset)
        avg_margin_err = val_margin_error / len(val_dataset)
        
        # Atualizar learning rate manualmente (sem verbose)
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_iou)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr < prev_lr:
            print(f"  ⬇️ LR reduzido: {prev_lr:.2e} -> {curr_lr:.2e}")
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.6f} | "
              f"Val IoU={avg_iou:.6f} | Margin Err={avg_margin_err:.6f}")
        
        # Early stopping baseado na COMBINAÇÃO de IoU e erro de margem
        score = avg_iou - (avg_margin_err * 0.3)  # Priorizar margens!
        
        if score > best_iou - (best_margin_err * 0.3):
            best_iou = avg_iou
            best_margin_err = avg_margin_err
            patience_counter = 0
            # Salvar melhor modelo
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'iou': best_iou,
                'margin_error': best_margin_err,
                'img_size': IMG_SIZE,
                'epoch': epoch + 1
            }, "models/best_model.pth")
            print(f"  ✅ Novo melhor modelo salvo (IoU: {best_iou:.6f} | Margin Err: {best_margin_err:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping ativado após {epoch+1} épocas")
                break
    
    print(f"\n✅ Treinamento concluído! Melhor IoU: {best_iou:.6f} | Erro Margem: {best_margin_err:.6f}")


if __name__ == "__main__":
    train()