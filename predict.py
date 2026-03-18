import datetime
import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import tkinter as tk
from tkinter import filedialog, messagebox
import concurrent.futures


# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join("logs", f"{script_name}_{timestamp}.log")
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("predict")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logging.captureWarnings(True)
    return logger


log: logging.Logger = logging.getLogger("predict")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============ MODELO (IDÊNTICO AO TREINAMENTO) ============
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
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 4),
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
        return torch.sigmoid(self.regressor(x))


# ============ FUNÇÃO DE CORTE COM PRECISÃO EXTREMA ============
def predict_crop(model, image_path, img_size=360):  # corrigir default
    img = cv2.imread(image_path)
    if img is None:
        log.error(f"Falha ao carregar imagem: {image_path}")
        raise ValueError(f"Não foi possível carregar {image_path}")

    orig_h, orig_w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    tensor = transforms.ToTensor()(img_resized)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                pred = model(tensor)[0].cpu().numpy()
        else:
            pred = model(tensor)[0].cpu().numpy()

    # Desnormalizar — SEM min_margin, igual ao treino
    x1 = int(np.clip(np.round(pred[0] * orig_w), 0, orig_w - 1))
    y1 = int(np.clip(np.round(pred[1] * orig_h), 0, orig_h - 1))
    x2 = int(np.clip(np.round(pred[2] * orig_w), x1 + 1, orig_w))
    y2 = int(np.clip(np.round(pred[3] * orig_h), y1 + 1, orig_h))

    if x2 <= x1 or y2 <= y1:
        log.error(f"Predição inválida para {image_path}: x1={x1} y1={y1} x2={x2} y2={y2}")
        raise ValueError("Predição inválida: coordenadas incorretas")

    log.debug(f"Crop {image_path}: x1={x1} y1={y1} x2={x2} y2={y2} (orig {orig_w}x{orig_h})")
    cropped = img[y1:y2, x1:x2]
    return cropped, (x1, y1, x2 - x1, y2 - y1)


def _load_model() -> tuple[MarginAwareCropModel, int, float, float]:
    log.info("Carregando modelo...")
    checkpoint_path = "models/best_model.pth"
    if not os.path.exists(checkpoint_path):
        log.error(f"Modelo não encontrado em: {checkpoint_path}")
        raise FileNotFoundError(f"Modelo não encontrado em: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model = MarginAwareCropModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    img_size = checkpoint.get("img_size", 360)
    train_iou = checkpoint.get("iou", 0.0)
    train_margin_err = checkpoint.get("margin_error", 0.0)

    log.info(f"Modelo carregado (IoU: {train_iou:.4f} | Erro Margem: {train_margin_err:.6f})")
    return model, img_size, train_iou, train_margin_err


def _choose_mode(root: tk.Tk) -> bool:
    return messagebox.askyesno(
        "Modo de Processamento",
        "Processar pasta inteira?\n\nSim = pasta inteira\nNão = uma imagem",
        parent=root,
    )


def _choose_path(root: tk.Tk, is_folder: bool) -> str:
    if is_folder:
        return filedialog.askdirectory(title="Selecione a pasta com imagens originais", parent=root)
    return filedialog.askopenfilename(
        title="Selecione uma imagem para cortar",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP")],
        parent=root,
    )


def _process(model: MarginAwareCropModel, img_size: int, path: str, is_folder: bool):
    try:
        if is_folder:
            images = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                and "_editado" not in f.lower()
            ]
            total = len(images)
            if total == 0:
                raise ValueError("Nenhuma imagem encontrada na pasta!")

            output_dir = os.path.join(path, "out")
            os.makedirs(output_dir, exist_ok=True)

            success = 0
            errors = []
            max_workers = min(8, (os.cpu_count() or 4))

            log.info(f"Processando {total} imagens com {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(predict_crop, model, img, img_size): img for img in images}
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    img_path = futures[future]
                    base_name = os.path.basename(img_path)
                    try:
                        cropped, coords = future.result()
                        base = os.path.splitext(base_name)[0]
                        base = base.replace("_editado", "").replace("_edited", "")
                        out_path = os.path.join(output_dir, f"{base}_editado.jpg")
                        cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
                        success += 1
                        log.info(f"[{i}/{total}] {base_name} -> x={coords[0]} y={coords[1]} w={coords[2]} h={coords[3]}")
                    except Exception as e:
                        err = f"{base_name}: {str(e)}"
                        errors.append(err)
                        log.error(f"[{i}/{total}] {err}")

            log.info(f"Processamento concluido: {success}/{total} sucesso, {len(errors)} erros. Saida: {output_dir}")

            msg = (
                f"Processamento concluído!\n\n"
                f"Sucesso: {success}/{total}\n"
                f"Erros: {len(errors)}\n\n"
                f"Saída: {output_dir}"
            )
            if errors:
                msg += f"\n\nPrimeiros erros:\n" + "\n".join(errors[:3])

            messagebox.showinfo("Concluído", msg)

        else:
            cropped, coords = predict_crop(model, path, img_size)
            output_dir = os.path.join(os.path.dirname(path), "out")
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            base = base.replace("_editado", "").replace("_edited", "")
            out_path = os.path.join(output_dir, f"{base}_editado.jpg")
            cv2.imwrite(out_path, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

            messagebox.showinfo(
                "Sucesso",
                f"Imagem cortada!\n\n"
                f"Coordenadas: x={coords[0]} y={coords[1]} w={coords[2]} h={coords[3]}\n"
                f"Salva em: {out_path}"
            )

    except Exception as e:
        log.error(f"Erro durante processamento: {e}", exc_info=True)
        messagebox.showerror("Erro", f"Erro durante processamento:\n{str(e)}")


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        model, img_size, train_iou, train_margin_err = _load_model()
    except Exception as e:
        log.error(f"Falha ao carregar modelo: {e}", exc_info=True)
        messagebox.showerror("Erro Fatal", f"Não foi possível carregar o modelo:\n{str(e)}")
        root.destroy()
        return

    log.info(f"Device: {DEVICE}")

    is_folder = _choose_mode(root)
    path = _choose_path(root, is_folder)
    if not path:
        root.destroy()
        return

    _process(model, img_size, path, is_folder)
    root.destroy()


if __name__ == "__main__":
    _setup_logging()
    main()
