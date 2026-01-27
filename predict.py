# predict.py
"""
Inferência GUI com TTA flip + cap por percentis (evita cortes gigantes).
Comportamento:
 - tenta automaticamente models/crop_localizer.pt (ou .pth)
 - pergunta se processa pasta inteira (Yes) ou 1 imagem (No)
 - se for pasta: cria pasta_out = SELECTED_FOLDER/out/ e salva lá com sufixo "_editado"
 - se for 1 imagem: pede onde salvar o arquivo, salva com sufixo "_editado"
"""
import os
import traceback
from pathlib import Path
from tkinter import Tk, filedialog, messagebox
import argparse  # Para lidar com argumentos de linha de comando
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

def build_model():
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_f, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 4),
        nn.Sigmoid()
    )
    return backbone

def _focus_window(root):
    root.attributes("-topmost", True); root.update(); root.lift(); root.focus_force()

def list_images(path: Path):
    p = Path(path)
    if p.is_file(): 
        yield p
    else:
        for img in sorted(p.rglob("*")):
            if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                yield img

def load_checkpoint(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    metadata = ckpt.get("metadata", {})
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device); model.eval()
    resize = tuple(metadata.get("resize", (800,800)))
    mean = metadata.get("mean", [0.485,0.456,0.406])
    std = metadata.get("std", [0.229,0.224,0.225])
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, transform, metadata

def preds_flip_average(model, transform, pil_img, device):
    """
    TTA: original + horizontal flip (flip image before transform).
    Return averaged normalized xyxy.
    """
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        p1 = model(t).cpu().numpy()[0]
    pil_f = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    tf = transform(pil_f).unsqueeze(0).to(device)
    with torch.no_grad():
        p2 = model(tf).cpu().numpy()[0]
    x1f, y1f, x2f, y2f = p2.tolist()
    p2_mapped = np.array([1.0 - x2f, y1f, 1.0 - x1f, y2f], dtype=np.float32)
    avg = (p1 + p2_mapped) / 2.0
    avg = np.clip(avg, 0.0, 1.0)
    if avg[2] < avg[0]:
        avg[0], avg[2] = avg[2], avg[0]
    if avg[3] < avg[1]:
        avg[1], avg[3] = avg[3], avg[1]
    return avg

def cap_bbox_by_stats(x1, y1, x2, y2, stats, cap_multiplier=1.5, rel_threshold=0.15):
    """
    Corta boxes muito grandes com base nas estatísticas do dataset, MAS:
     - só reduz se a diferença for relevante (rel_threshold)
     - usa cap_multiplier maior por padrão para evitar cortes agressivos
    """
    if not stats:
        return x1, y1, x2, y2

    width = x2 - x1
    height = y2 - y1
    try:
        w99 = stats["width_p"][-1]
        h99 = stats["height_p"][-1]
    except Exception:
        return x1, y1, x2, y2

    max_w = min(cap_multiplier * w99, 0.95)
    max_h = min(cap_multiplier * h99, 0.95)

    # Se já está dentro do limite, nada a fazer
    if width <= max_w and height <= max_h:
        return x1, y1, x2, y2

    # Só reduzimos se a diferença for significativa (porcentagem relativa)
    shrink_w = (width > max_w) and (((width - max_w) / width) > rel_threshold)
    shrink_h = (height > max_h) and (((height - max_h) / height) > rel_threshold)

    new_w = max_w if shrink_w else width
    new_h = max_h if shrink_h else height

    # Se nenhuma dimensão for reduzida significativamente, preserva original
    if new_w == width and new_h == height:
        return x1, y1, x2, y2

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1n = max(0.0, cx - new_w / 2.0)
    x2n = min(1.0, cx + new_w / 2.0)
    y1n = max(0.0, cy - new_h / 2.0)
    y2n = min(1.0, cy + new_h / 2.0)
    return x1n, y1n, x2n, y2n


def adjust_margin(x1,y1,x2,y2, stats=None, margin_cap=0.02):
    w = x2 - x1; h = y2 - y1
    base = 0.015
    rel = 0.06 * max(w,h)
    margin = max(base, rel); margin = min(margin, margin_cap)
    if stats:
        margin = min(margin, 0.15 * max(stats.get("width_mean",0.0), stats.get("height_mean",0.0)))
    x1n = max(0.0, x1 - margin); y1n = max(0.0, y1 - margin)
    x2n = min(1.0, x2 + margin); y2n = min(1.0, y2 + margin)
    return x1n,y1n,x2n,y2n

def crop_image(img_bgr, xyxy_norm, out_size=None):
    H,W = img_bgr.shape[:2]
    x1 = int(round(xyxy_norm[0]*W)); y1 = int(round(xyxy_norm[1]*H))
    x2 = int(round(xyxy_norm[2]*W)); y2 = int(round(xyxy_norm[3]*H))
    x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
    if x2 <= x1 or y2 <= y1:
        cw = int(W*0.8); ch = int(H*0.8); sx = max(0,(W-cw)//2); sy = max(0,(H-ch)//2)
        return img_bgr[sy:sy+ch, sx:sx+cw]
    crop = img_bgr[y1:y2, x1:x2]
    if out_size:
        crop = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    return crop

def ensure_editado_suffix(path: Path) -> Path:
    """
    Return a Path with suffix '_editado' before extension, avoid duplicating if already present.
    """
    p = Path(path)
    stem = p.stem
    if stem.endswith("_editado"):
        return p
    new_name = stem + "_editado" + p.suffix
    return p.with_name(new_name)

def choose_paths():
    root = Tk(); root.withdraw()
    model_path = Path("models/crop_localizer.pt")
    if not model_path.exists():
        alt = Path("models/crop_localizer.pth")
        if alt.exists(): model_path = alt
    if not model_path.exists():
        _focus_window(root); messagebox.showerror("Modelo não encontrado", f"Coloque 'models/crop_localizer.pt'.", parent=root)
        root.destroy(); raise SystemExit("Modelo ausente.")
    _focus_window(root)
    process_folder = messagebox.askyesno("Entrada", "Deseja processar uma pasta inteira de imagens?", parent=root)
    if process_folder:
        input_dir = filedialog.askdirectory(title="Selecione a pasta com imagens", parent=root)
        if not input_dir: root.destroy(); raise SystemExit("Pasta não selecionada.")
        out_dir = Path(input_dir) / "out"; out_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(input_dir); output = out_dir
    else:
        img_file = filedialog.askopenfilename(title="Selecione a imagem", filetypes=[("Imagens","*.png;*.jpg;*.jpeg;*.bmp")], parent=root)
        if not img_file: root.destroy(); raise SystemExit("Imagem não selecionada.")
        # ask where to save, but ensure we will add _editado suffix
        save_as = filedialog.asksaveasfilename(title="Salvar crop como (será adicionado _editado)", defaultextension=".jpg", parent=root)
        if not save_as: root.destroy(); raise SystemExit("Saída não selecionada.")
        # convert to Path and ensure suffix
        save_path = ensure_editado_suffix(Path(save_as))
        input_path = Path(img_file); output = save_path
    root.destroy()
    return model_path, input_path, output

def predict_and_save(model, transform, metadata, image_path: Path, output_target):
    device = next(model.parameters()).device
    pil = Image.open(str(image_path)).convert("RGB")
    avg = preds_flip_average(model, transform, pil, device)
    stats = metadata.get("bbox_stats", None)
    x1,y1,x2,y2 = adjust_margin(avg[0], avg[1], avg[2], avg[3], stats=stats, margin_cap=0.005)
    x1,y1,x2,y2 = cap_bbox_by_stats(x1, y1, x2, y2, stats, cap_multiplier=1.5, rel_threshold=0.15)
    img_bgr = cv2.imread(str(image_path))
    out_crop = crop_image(img_bgr, (x1,y1,x2,y2), out_size=None)
    if isinstance(output_target, Path) and output_target.is_dir():
        # preserve original extension, add _editado suffix
        out_name = ensure_editado_suffix(Path(image_path.name))
        out_path = output_target / out_name.name
    else:
        out_path = Path(output_target)
        # ensure suffix _editado
        out_path = ensure_editado_suffix(out_path)
    cv2.imwrite(str(out_path), out_crop)
    print(f"Saved: {out_path}")
    return out_path

def debug_predict(model, transform, metadata, image_path: Path, output_dir: Path):
    """
    Gera imagens com diferentes configurações de margin_cap, cap_multiplier e rel_threshold,
    pintando as áreas previstas ao invés de cortar e escrevendo os parâmetros na imagem.
    """
    device = next(model.parameters()).device
    pil = Image.open(str(image_path)).convert("RGB")
    avg = preds_flip_average(model, transform, pil, device)
    stats = metadata.get("bbox_stats", None)
    img_bgr = cv2.imread(str(image_path))

    # Configurações de debug
    margin_caps = [0.005, 0.01, 0.02]
    cap_multipliers = [1.0, 1.5, 2.0]
    rel_thresholds = [0.1, 0.15, 0.2]

    for margin_cap in margin_caps:
        for cap_multiplier in cap_multipliers:
            for rel_threshold in rel_thresholds:
                x1, y1, x2, y2 = adjust_margin(avg[0], avg[1], avg[2], avg[3], stats=stats, margin_cap=margin_cap)
                x1, y1, x2, y2 = cap_bbox_by_stats(x1, y1, x2, y2, stats, cap_multiplier=cap_multiplier, rel_threshold=rel_threshold)
                H, W = img_bgr.shape[:2]
                x1_px, y1_px = int(x1 * W), int(y1 * H)
                x2_px, y2_px = int(x2 * W), int(y2 * H)

                # Cria uma cópia da imagem e pinta a área prevista
                debug_img = img_bgr.copy()
                cv2.rectangle(debug_img, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 0), 2)

                # Escreve os parâmetros na imagem
                text = f"margin_cap: {margin_cap}, cap_multiplier: {cap_multiplier}, rel_threshold: {rel_threshold}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                color = (0, 255, 0)  # Verde
                text_position = (10, 30)  # Posição do texto na imagem
                cv2.putText(debug_img, text, text_position, font, font_scale, color, thickness, cv2.LINE_AA)

                # Gera o nome do arquivo de saída
                out_name = f"{image_path.stem}_m{margin_cap}_c{cap_multiplier}_r{rel_threshold}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), debug_img)
                print(f"Saved debug image: {out_path}")

def process_images_async(model, transform, metadata, input_path: Path, output_target):
    """
    Processa imagens de uma pasta de forma assíncrona para melhorar a performance.
    """
    processed = 0
    failed = []

    def process_image(image_path):
        try:
            predict_and_save(model, transform, metadata, image_path, output_target)
            return image_path, None
        except Exception as e:
            return image_path, str(e)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, p): p for p in list_images(input_path)}
        for future in as_completed(futures):
            image_path, error = future.result()
            if error:
                failed.append((str(image_path), error))
            else:
                processed += 1

    return processed, failed

def gui_main():
    parser = argparse.ArgumentParser(description="Crop prediction tool")
    parser.add_argument("--debug", action="store_true", help="Ativa o modo debug")
    args = parser.parse_args()

    if args.debug:
        root = Tk(); root.withdraw()
        _focus_window(root)
        img_file = filedialog.askopenfilename(title="Selecione a imagem para debug", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")], parent=root)
        if not img_file:
            root.destroy()
            raise SystemExit("Imagem não selecionada.")
        save_dir = filedialog.askdirectory(title="Selecione a pasta para salvar os resultados", parent=root)
        if not save_dir:
            root.destroy()
            raise SystemExit("Pasta de saída não selecionada.")
        root.destroy()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = Path("models/crop_localizer.pt")
        if not model_path.exists():
            alt = Path("models/crop_localizer.pth")
            if alt.exists():
                model_path = alt
        if not model_path.exists():
            raise SystemExit("Modelo ausente.")
        model, transform, metadata = load_checkpoint(model_path, device)
        model.to(device)

        debug_predict(model, transform, metadata, Path(img_file), Path(save_dir))
        return

    try:
        model_path, input_path, output_target = choose_paths()
    except SystemExit:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, metadata = load_checkpoint(model_path, device)
    model.to(device)
    processed = 0
    failed = []
    try:
        if input_path.is_file():
            try:
                predict_and_save(model, transform, metadata, input_path, output_target)
                processed += 1
            except Exception as e:
                failed.append((str(input_path), str(e)))
        else:
            # Processamento assíncrono para pastas
            processed, failed = process_images_async(model, transform, metadata, input_path, output_target)
    except Exception:
        traceback.print_exc()
    finally:
        messagebox.showinfo("Concluído", f"Processados: {processed}\nFalhas: {len(failed)}")
        if failed:
            print("Failures:")
            for a, b in failed:
                print(a, b)

if __name__ == "__main__":
    gui_main()
