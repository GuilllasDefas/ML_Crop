import os
import time
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

# ============ DATASET OTIMIZADO (CARREGAMENTO SOB DEMANDA) ============
class CropDataset(Dataset):
    def __init__(self, original_paths, cropped_paths, bbox_data, img_size=300):
        self.img_size = img_size
        self.pairs = []
        
        print("Preparando dataset...")
        for orig_path, crop_path, bbox_norm in zip(original_paths, cropped_paths, bbox_data):
            try:
                # Armazenar apenas os caminhos e bounding boxes já calculados
                self.pairs.append({
                    'orig_path': orig_path,
                    'crop_path': crop_path,
                    'bbox': bbox_norm.astype(np.float32),
                })
            except Exception as e:
                print(f"Erro ao processar {orig_path}: {e}")
                continue
        
        print(f"Dataset preparado: {len(self.pairs)} pares válidos")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Carregar imagem original somente quando necessário
        orig = cv2.imread(pair['orig_path'])
        if orig is None:
            # Retornar uma imagem padrão em caso de erro
            dummy_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
            dummy_bbox = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)
            return dummy_img, dummy_bbox
        
        # Pré-processar imagem original
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_resized = cv2.resize(orig_rgb, (self.img_size, self.img_size))
        orig_tensor = transforms.ToTensor()(orig_resized)
        orig_tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(orig_tensor)
        
        return orig_tensor, pair['bbox']

# Função para pré-calcular bounding boxes (executada uma vez)
def compute_bounding_boxes(orig_paths, crop_paths):
    """Função para calcular bounding boxes uma única vez"""
    bbox_list = []
    
    print("Calculando bounding boxes (isso será feito apenas uma vez)...")
    for orig_path, crop_path in tqdm(zip(orig_paths, crop_paths), total=len(orig_paths)):
        try:
            orig = cv2.imread(orig_path)
            crop = cv2.imread(crop_path)
            
            if orig is None or crop is None:
                bbox_list.append(np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32))  # bbox padrão
                continue
            
            orig_h, orig_w = orig.shape[:2]
            
            # Template matching otimizado
            gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Usar método mais eficiente de template matching
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
            
            bbox_list.append(bbox_norm)
            
        except Exception as e:
            print(f"Erro ao calcular bbox para {orig_path}: {e}")
            bbox_list.append(np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32))  # bbox padrão
    
    return bbox_list


# ============ MODELO OTIMIZADO ============
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
        
        # Remover a cabeça final para manter apenas features
        self.features = backbone.features
        
        # Adicionar global average pooling explícito
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regressor otimizado
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),  # Reduzindo levemente o dropout
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),  # Adicionando batch normalization
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),  # Reduzindo levemente o dropout
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.regressor(x))  # Forçar saída em [0,1]


# ============ LOSS FUNCTIONS OTIMIZADAS ============
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


# ============ TREINAMENTO OTIMIZADO ============
def train():
    # Configurações otimizadas
    IMG_SIZE = 300
    BATCH_SIZE = 16  # Pode ser aumentado dependendo da GPU
    NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 4
    EPOCHS = 100
    PATIENCE = 15  # Mais tolerante para convergência fina das margens
    
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
    
    # Calcular bounding boxes uma única vez (otimização principal)
    bbox_data = compute_bounding_boxes(orig_paths, crop_paths)
    
    # Divisão treino/validação
    train_orig, val_orig, train_crop, val_crop, train_bbox, val_bbox = train_test_split(
        orig_paths, crop_paths, bbox_data, test_size=0.1, random_state=42
    )
    
    # Criar datasets (agora com bounding boxes pré-calculados)
    train_dataset = CropDataset(train_orig, train_crop, train_bbox, img_size=IMG_SIZE)
    val_dataset = CropDataset(val_orig, val_crop, val_bbox, img_size=IMG_SIZE)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset vazio após pré-processamento!")
    
    # Dataloaders otimizados
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True
    )
    
    # Modelo e otimizador otimizados
    model = MarginAwareCropModel().to(DEVICE)
    # Otimizador com lookahead (opcional, descomentar se quiser usar)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Scheduler otimizado
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=6  # Modo 'min' para loss
    )
    
    # Mixed Precision para Tensor Cores
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Treinamento otimizado
    for epoch in range(EPOCHS):
        # ===== TREINO =====
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(inputs)
                    loss = combined_loss(preds, targets, alpha=0.45)  # Dar mais peso às margens!
                scaler.scale(loss).backward()
                # Gradient clipping para estabilidade
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(inputs)
                loss = combined_loss(preds, targets, alpha=0.45)
                loss.backward()
                # Gradient clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        # ===== VALIDAÇÃO =====
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_margin_error = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        preds = model(inputs)
                        loss = combined_loss(preds, targets, alpha=0.45)
                else:
                    preds = model(inputs)
                    loss = combined_loss(preds, targets, alpha=0.45)
                
                val_loss += loss.item() * inputs.size(0)
                
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
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_dataset)
        avg_iou = val_iou / len(val_dataset)
        avg_margin_err = val_margin_error / len(val_dataset)
        
        # Atualizar learning rate
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        curr_lr = optimizer.param_groups[0]['lr']
        if curr_lr < prev_lr:
            print(f"  ⬇️ LR reduzido: {prev_lr:.2e} -> {curr_lr:.2e}")
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f} | Val Loss={avg_val_loss:.6f} | "
              f"Val IoU={avg_iou:.6f} | Margin Err={avg_margin_err:.6f}")
        
        # Early stopping baseado na loss de validação
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Salvar melhor modelo
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': best_loss,
                'iou': avg_iou,
                'margin_error': avg_margin_err,
                'img_size': IMG_SIZE,
                'epoch': epoch + 1
            }, "models/best_model.pth")
            print(f"  ✅ Novo melhor modelo salvo (Val Loss: {best_loss:.6f} | IoU: {avg_iou:.6f} | Margin Err: {avg_margin_err:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️ Early stopping ativado após {epoch+1} épocas")
                break
    
    print(f"\n✅ Treinamento concluído! Melhor Val Loss: {best_loss:.6f} | IoU: {avg_iou:.6f} | Erro Margem: {avg_margin_err:.6f}")


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Tempo total de execução: {elapsed_hms}")