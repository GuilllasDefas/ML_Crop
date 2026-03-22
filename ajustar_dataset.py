"""Auditoria e ajuste do dataset com comparação modelo vs dataset.

Workflow:
1. Carrega todos os pares origin/cropped do dataset
2. Calcula predição do modelo vigente para cada imagem
3. Ordena por discrepância (menor IoU = maior prioridade)
4. Exibe: retângulo verde interativo (dataset atual) + cyan tracejado (modelo)
5. Ações:
   - Pular (←): manter crop atual, próxima imagem
   - Aceitar Modelo (M): substituir crop pelo do modelo
   - Aceitar Ajuste (→/Enter): substituir crop pelo ajuste manual
   - Voltar (⌫), Desfazer (Ctrl+Z), Reset (R)
"""

import datetime
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from tqdm import tqdm

from predict import MarginAwareCropModel, _load_model, DEVICE

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
HANDLE_SIZE = 6
HANDLE_HIT = 12
MIN_CROP_DIM = 20


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
    logger = logging.getLogger("ajustar_dataset")
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


log: logging.Logger = logging.getLogger("ajustar_dataset")


# ── Utilidades ────────────────────────────────────────────────────────────────
def _compute_bbox_cv2(orig_path: str, crop_path: str) -> np.ndarray:
    """Template matching para encontrar bbox normalizado do crop no original."""
    try:
        orig = cv2.imread(orig_path)
        crop = cv2.imread(crop_path)
        if orig is None or crop is None:
            return np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)

        orig_h, orig_w = orig.shape[:2]
        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        x1, y1 = max_loc
        x2, y2 = x1 + crop.shape[1], y1 + crop.shape[0]

        return np.array([
            x1 / orig_w, y1 / orig_h,
            x2 / orig_w, y2 / orig_h,
        ], dtype=np.float32)
    except Exception as e:
        log.error(f"Erro ao calcular bbox para {orig_path}: {e}")
        return np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre dois bboxes normalizados [x1, y1, x2, y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _denorm_bbox(bbox: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    """Desnormaliza bbox [0,1] para coordenadas de pixel."""
    x1 = int(np.clip(np.round(bbox[0] * w), 0, w - 1))
    y1 = int(np.clip(np.round(bbox[1] * h), 0, h - 1))
    x2 = int(np.clip(np.round(bbox[2] * w), x1 + 1, w))
    y2 = int(np.clip(np.round(bbox[3] * h), y1 + 1, h))
    return x1, y1, x2, y2


def _load_dataset_pairs() -> Tuple[List[Path], List[Path]]:
    """Retorna listas ordenadas (orig_paths, crop_paths) do dataset."""
    project_root = Path(__file__).resolve().parent
    origin_dir = project_root / "dataset" / "origin"
    cropped_dir = project_root / "dataset" / "cropped"

    if not origin_dir.exists() or not cropped_dir.exists():
        return [], []

    crop_dict: Dict[str, Path] = {}
    for f in cropped_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS and "_editado" in f.stem.lower():
            base = f.stem.lower().replace("_editado", "")
            crop_dict[base] = f

    orig_paths: List[Path] = []
    crop_paths: List[Path] = []
    for f in sorted(origin_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            base = f.stem.lower()
            if base in crop_dict:
                orig_paths.append(f)
                crop_paths.append(crop_dict[base])

    return orig_paths, crop_paths


def _load_dataset_bboxes(
    orig_paths: List[Path],
    crop_paths: List[Path],
    max_workers: int = 8,
) -> List[np.ndarray]:
    """Carrega bboxes do cache ou calcula via template matching."""
    cache_path = os.path.join("models", "bbox_cache.pkl")

    cached_dict: Dict[str, np.ndarray] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            for cp, bbox in zip(cache.get("orig_paths", []), cache.get("bboxes", [])):
                cached_dict[os.path.basename(cp)] = bbox
            log.info(f"Cache de bboxes carregado ({len(cached_dict)} entradas)")
        except Exception as e:
            log.warning(f"Falha ao ler cache de bbox: {e}")

    result: List[np.ndarray] = []
    missing_indices: List[int] = []

    for i, op in enumerate(orig_paths):
        key = os.path.basename(str(op))
        if key in cached_dict:
            result.append(cached_dict[key])
        else:
            result.append(None)  # type: ignore[arg-type]
            missing_indices.append(i)

    if missing_indices:
        log.info(f"Calculando {len(missing_indices)} bboxes faltantes...")
        args = [
            (str(orig_paths[i]), str(crop_paths[i])) for i in missing_indices
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            computed = list(tqdm(
                executor.map(lambda a: _compute_bbox_cv2(*a), args),
                total=len(args), desc="Bboxes",
            ))
        for i, bbox in zip(missing_indices, computed):
            result[i] = bbox

    return result


def _batch_predict(
    model: MarginAwareCropModel,
    orig_paths: List[Path],
    img_size: int,
    batch_size: int = 64,
    max_workers: int = 8,
    progress_callback=None,
) -> List[np.ndarray]:
    """Prediz bboxes normalizados para todas as imagens em lotes."""
    model.eval()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )

    def _load_one(path: Path) -> Optional[torch.Tensor]:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size), Image.LANCZOS)
            tensor = transforms.ToTensor()(img)
            return normalize(tensor)
        except Exception:
            return None

    predictions: List[Optional[np.ndarray]] = [None] * len(orig_paths)
    default_bbox = np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)
    processed = 0

    for start in range(0, len(orig_paths), batch_size):
        batch_paths = orig_paths[start:start + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tensors = list(executor.map(_load_one, batch_paths))

        valid_indices: List[int] = []
        valid_tensors: List[torch.Tensor] = []
        for i, t in enumerate(tensors):
            if t is not None:
                valid_indices.append(start + i)
                valid_tensors.append(t)
            else:
                predictions[start + i] = default_bbox.copy()

        if valid_tensors:
            batch_tensor = torch.stack(valid_tensors).to(DEVICE)
            with torch.no_grad():
                if DEVICE.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        preds = model(batch_tensor).cpu().numpy()
                else:
                    preds = model(batch_tensor).cpu().numpy()

            for idx, pred in zip(valid_indices, preds):
                predictions[idx] = pred.astype(np.float32)

        processed += len(batch_paths)
        if progress_callback:
            progress_callback(processed, len(orig_paths))

    return predictions  # type: ignore[return-value]


# ── Canvas interativo ─────────────────────────────────────────────────────────
class CropCanvas(tk.Canvas):
    """Canvas com imagem, retângulo interativo e retângulo de referência."""

    def __init__(self, master, on_crop_change=None, **kwargs):
        kwargs.setdefault("bg", "#1e1e1e")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)

        self._pil_img: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._scale: float = 1.0
        self._offset: Tuple[int, int] = (0, 0)
        self._img_display_size: Tuple[int, int] = (0, 0)
        self._crop: List[int] = [0, 0, 1, 1]
        self._original_crop: List[int] = [0, 0, 1, 1]
        self._reference_crop: Optional[List[int]] = None

        self._overlay_ids: List[int] = []
        self._rect_id: Optional[int] = None
        self._ref_rect_id: Optional[int] = None
        self._handle_ids: List[int] = []
        self._on_crop_change = on_crop_change

        self._drag_mode: Optional[str] = None
        self._drag_start: Optional[Tuple[int, int]] = None
        self._crop_at_drag_start: Optional[List[int]] = None
        self._render_job: Optional[str] = None

        self.bind("<Configure>", self._on_configure)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Motion>", self._on_hover)

    # ── API pública ──────────────────────────────────────────────

    def set_image(self, pil_img: Image.Image):
        self._pil_img = pil_img.convert("RGB")
        self._render()

    def set_crop(self, x1: int, y1: int, x2: int, y2: int):
        self._crop = [x1, y1, x2, y2]
        self._original_crop = [x1, y1, x2, y2]
        self._draw_overlay()

    def set_reference_crop(self, x1: int, y1: int, x2: int, y2: int):
        """Retângulo de referência (não interativo) — predição do modelo."""
        self._reference_crop = [x1, y1, x2, y2]
        self._draw_overlay()

    def get_crop(self) -> Tuple[int, int, int, int]:
        return tuple(self._crop)  # type: ignore[return-value]

    def reset_crop(self):
        self._crop = list(self._original_crop)
        self._draw_overlay()
        if self._on_crop_change:
            self._on_crop_change()

    # ── Conversão de coordenadas ─────────────────────────────────

    def _img_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        return ix * self._scale + self._offset[0], iy * self._scale + self._offset[1]

    def _canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        if self._scale == 0:
            return 0.0, 0.0
        return (cx - self._offset[0]) / self._scale, (cy - self._offset[1]) / self._scale

    # ── Renderização ─────────────────────────────────────────────

    def _on_configure(self, _evt=None):
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(50, self._render)

    def _render(self):
        self._render_job = None
        if self._pil_img is None:
            return
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self._pil_img.size
        sx = cw / iw
        sy = ch / ih
        self._scale = min(sx, sy)
        dw = int(iw * self._scale)
        dh = int(ih * self._scale)
        self._img_display_size = (dw, dh)
        self._offset = ((cw - dw) // 2, (ch - dh) // 2)

        resized = self._pil_img.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.delete("all")
        self.create_image(self._offset[0], self._offset[1], anchor="nw", image=self._photo)
        self._draw_overlay()

    def _draw_overlay(self):
        for item_id in self._overlay_ids:
            self.delete(item_id)
        self._overlay_ids.clear()
        if self._rect_id is not None:
            self.delete(self._rect_id)
            self._rect_id = None
        if self._ref_rect_id is not None:
            self.delete(self._ref_rect_id)
            self._ref_rect_id = None
        for hid in self._handle_ids:
            self.delete(hid)
        self._handle_ids.clear()

        if self._pil_img is None:
            return

        cx1, cy1 = self._img_to_canvas(self._crop[0], self._crop[1])
        cx2, cy2 = self._img_to_canvas(self._crop[2], self._crop[3])
        ix1, iy1 = self._offset
        ix2 = ix1 + self._img_display_size[0]
        iy2 = iy1 + self._img_display_size[1]

        # Sombreamento fora do crop
        for coords in [
            (ix1, iy1, ix2, cy1),
            (ix1, cy2, ix2, iy2),
            (ix1, cy1, cx1, cy2),
            (cx2, cy1, ix2, cy2),
        ]:
            rid = self.create_rectangle(
                *coords, fill="#000000", stipple="gray50", outline="",
            )
            self._overlay_ids.append(rid)

        # Retângulo interativo (dataset atual) — verde
        self._rect_id = self.create_rectangle(
            cx1, cy1, cx2, cy2, outline="#00ff00", width=2,
        )

        # Retângulo de referência (modelo) — cyan tracejado
        if self._reference_crop is not None:
            rx1, ry1 = self._img_to_canvas(self._reference_crop[0], self._reference_crop[1])
            rx2, ry2 = self._img_to_canvas(self._reference_crop[2], self._reference_crop[3])
            self._ref_rect_id = self.create_rectangle(
                rx1, ry1, rx2, ry2, outline="#00d4ff", width=2, dash=(8, 4),
            )

        # Alças nos 4 cantos
        for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]:
            hid = self.create_rectangle(
                hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                fill="#00ff00", outline="#ffffff", width=1,
            )
            self._handle_ids.append(hid)

        # Alças nos pontos médios das arestas
        cmx = (cx1 + cx2) / 2
        cmy = (cy1 + cy2) / 2
        for hx, hy in [(cmx, cy1), (cmx, cy2), (cx1, cmy), (cx2, cmy)]:
            hid = self.create_rectangle(
                hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                fill="#ffffff", outline="#00ff00", width=1,
            )
            self._handle_ids.append(hid)

    # ── Interação ────────────────────────────────────────────────

    def _detect_handle(self, cx: float, cy: float) -> Optional[str]:
        cx1, cy1 = self._img_to_canvas(self._crop[0], self._crop[1])
        cx2, cy2 = self._img_to_canvas(self._crop[2], self._crop[3])
        cmx = (cx1 + cx2) / 2
        cmy = (cy1 + cy2) / 2

        for name, hx, hy in [
            ("nw", cx1, cy1), ("ne", cx2, cy1),
            ("sw", cx1, cy2), ("se", cx2, cy2),
        ]:
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name
        for name, hx, hy in [
            ("n", cmx, cy1), ("s", cmx, cy2),
            ("w", cx1, cmy), ("e", cx2, cmy),
        ]:
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name
        return None

    def _is_inside_box(self, cx: float, cy: float) -> bool:
        cx1, cy1 = self._img_to_canvas(self._crop[0], self._crop[1])
        cx2, cy2 = self._img_to_canvas(self._crop[2], self._crop[3])
        return cx1 <= cx <= cx2 and cy1 <= cy <= cy2

    def _on_hover(self, evt):
        handle = self._detect_handle(evt.x, evt.y)
        if handle:
            self.config(cursor="crosshair")
        elif self._is_inside_box(evt.x, evt.y):
            self.config(cursor="fleur")
        else:
            self.config(cursor="")

    def _on_press(self, evt):
        handle = self._detect_handle(evt.x, evt.y)
        if handle:
            self._drag_mode = handle
        elif self._is_inside_box(evt.x, evt.y):
            self._drag_mode = "move"
        else:
            self._drag_mode = None
            return
        self._drag_start = (evt.x, evt.y)
        self._crop_at_drag_start = list(self._crop)

    def _on_motion(self, evt):
        if self._drag_mode is None or self._pil_img is None:
            return
        ix, iy = self._canvas_to_img(evt.x, evt.y)
        sx, sy = self._canvas_to_img(self._drag_start[0], self._drag_start[1])
        dx = ix - sx
        dy = iy - sy
        iw, ih = self._pil_img.size
        c = self._crop_at_drag_start

        if self._drag_mode == "move":
            bw = c[2] - c[0]
            bh = c[3] - c[1]
            new_x1 = int(np.clip(c[0] + dx, 0, iw - bw))
            new_y1 = int(np.clip(c[1] + dy, 0, ih - bh))
            self._crop = [new_x1, new_y1, new_x1 + bw, new_y1 + bh]
        else:
            x1, y1, x2, y2 = c[0], c[1], c[2], c[3]
            if "w" in self._drag_mode:
                x1 = int(np.clip(c[0] + dx, 0, x2 - MIN_CROP_DIM))
            if "e" in self._drag_mode:
                x2 = int(np.clip(c[2] + dx, x1 + MIN_CROP_DIM, iw))
            if "n" in self._drag_mode:
                y1 = int(np.clip(c[1] + dy, 0, y2 - MIN_CROP_DIM))
            if "s" in self._drag_mode:
                y2 = int(np.clip(c[3] + dy, y1 + MIN_CROP_DIM, ih))
            self._crop = [x1, y1, x2, y2]

        self._draw_overlay()
        if self._on_crop_change:
            self._on_crop_change()

    def _on_release(self, _evt):
        self._drag_mode = None
        self._drag_start = None
        self._crop_at_drag_start = None


# ── Lista de arquivos ─────────────────────────────────────────────────────────
class FileList(ttk.Frame):
    def __init__(self, master, on_select):
        super().__init__(master)
        self.on_select = on_select

        self.listbox = tk.Listbox(
            self, width=36, activestyle="dotbox", exportselection=False,
            bg="#3a3a3a", fg="#ffffff", selectbackground="#0078d4",
            selectforeground="#ffffff", font=("Consolas", 9),
        )
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.scroll.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def populate(self, items: List[Tuple[str, float]]):
        """items: lista de (filename, iou)."""
        self.listbox.delete(0, tk.END)
        for name, iou_val in items:
            self.listbox.insert(tk.END, f"{iou_val:.3f} | {name}")

    def mark_modified(self, index: int):
        try:
            label = self.listbox.get(index)
        except tk.TclError:
            return
        if not label.startswith("✓ "):
            self.listbox.delete(index)
            self.listbox.insert(index, f"✓ {label}")

    def unmark_modified(self, index: int):
        try:
            label = self.listbox.get(index)
        except tk.TclError:
            return
        if label.startswith("✓ "):
            self.listbox.delete(index)
            self.listbox.insert(index, label[2:])

    def _on_listbox_select(self, _evt):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.on_select(sel[0])

    def select_index(self, index: int):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
        self.on_select(index)


# ── App principal ─────────────────────────────────────────────────────────────
class AdjusterApp:
    def __init__(
        self,
        root: tk.Tk,
        pairs_data: List[dict],
        model: MarginAwareCropModel,
        img_size: int,
    ):
        self.root = root
        self.pairs = pairs_data
        self.model = model
        self.img_size = img_size
        self.current_index = 0
        self.modified: Set[int] = set()
        self._pil_img: Optional[Image.Image] = None
        self._history: List[int] = []
        self._undo_stack: List[Tuple[int, bytes]] = []

        self._apply_dark_theme()
        self._build_ui()
        self._bind_keys()
        if self.pairs:
            items = [(p["orig_path"].name, p["iou"]) for p in self.pairs]
            self.file_list.populate(items)
            self.file_list.select_index(0)

    def _apply_dark_theme(self):
        self.root.configure(bg="#2e2e2e")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#2e2e2e")
        style.configure("TLabel", background="#2e2e2e", foreground="#ffffff")
        style.configure("TButton", background="#444444", foreground="#ffffff")
        style.configure("TScrollbar", background="#444444", troughcolor="#3a3a3a")
        style.map("TButton", background=[("active", "#555555")])

    def _bind_keys(self):
        self.root.bind("<Left>", lambda _e: self.skip_current())
        self.root.bind("<Right>", lambda _e: self.accept_adjustment())
        self.root.bind("<Return>", lambda _e: self.accept_adjustment())
        self.root.bind("<m>", lambda _e: self.accept_model())
        self.root.bind("<M>", lambda _e: self.accept_model())
        self.root.bind("<BackSpace>", lambda _e: self.go_back())
        self.root.bind("<Control-z>", lambda _e: self.undo_last())
        self.root.bind("<Control-Z>", lambda _e: self.undo_last())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<r>", lambda _e: self._reset_crop())
        self.root.bind("<R>", lambda _e: self._reset_crop())

    def _build_ui(self):
        self.root.title("Ajustar Dataset - Auditoria com Modelo")
        self.root.minsize(1120, 660)
        self.root.geometry("1400x800")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Lista de arquivos
        self.file_list = FileList(self.root, self._on_select)
        self.file_list.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)

        # Painel direito
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.lbl_title = ttk.Label(right, text="", font=("Segoe UI", 10))
        self.lbl_title.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        canvas_container = ttk.Frame(right)
        canvas_container.grid(row=1, column=0, sticky="nsew")
        canvas_container.grid_propagate(False)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        self.canvas = CropCanvas(canvas_container, on_crop_change=self._on_crop_change)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Legenda
        legend = ttk.Frame(right)
        legend.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        tk.Label(
            legend, text="■ Verde = Dataset atual (interativo)",
            bg="#2e2e2e", fg="#00ff00", font=("Consolas", 9),
        ).pack(side="left", padx=(0, 20))
        tk.Label(
            legend, text="■ Cyan = Predição do modelo (referência)",
            bg="#2e2e2e", fg="#00d4ff", font=("Consolas", 9),
        ).pack(side="left")

        # Info do recorte
        self.lbl_crop_info = ttk.Label(right, text="", anchor="center")
        self.lbl_crop_info.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        # Botões
        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        for i in range(7):
            btn_frame.columnconfigure(i, weight=1)

        ttk.Button(
            btn_frame, text="Pular (←)", command=self.skip_current,
        ).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Aceitar Modelo (M)", command=self.accept_model,
        ).grid(row=0, column=1, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Aceitar Ajuste (→)", command=self.accept_adjustment,
        ).grid(row=0, column=2, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Voltar (⌫)", command=self.go_back,
        ).grid(row=0, column=3, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Desfazer (Ctrl+Z)", command=self.undo_last,
        ).grid(row=0, column=4, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Reset (R)", command=self._reset_crop,
        ).grid(row=0, column=5, padx=(0, 6), sticky="ew")
        ttk.Button(
            btn_frame, text="Sair (Esc)", command=self.root.destroy,
        ).grid(row=0, column=6, sticky="ew")

        # Status
        self.status_var = tk.StringVar()
        ttk.Label(right, textvariable=self.status_var, anchor="center").grid(
            row=5, column=0, sticky="ew", pady=(8, 0),
        )

    # ── Navegação ────────────────────────────────────────────────

    def _on_select(self, index: int):
        if hasattr(self, "_history") and self.current_index != index:
            self._history.append(self.current_index)
        self.current_index = index
        pair = self.pairs[index]
        iou_val = pair["iou"]
        self.lbl_title.config(
            text=f"{pair['orig_path'].name}  |  IoU dataset↔modelo: {iou_val:.4f}",
        )

        try:
            self._pil_img = Image.open(pair["orig_path"]).convert("RGB")
        except OSError:
            self._pil_img = Image.new("RGB", (100, 100), "#444")

        self.canvas.set_image(self._pil_img)

        w, h = self._pil_img.size
        dx1, dy1, dx2, dy2 = _denorm_bbox(pair["dataset_bbox"], w, h)
        mx1, my1, mx2, my2 = _denorm_bbox(pair["model_bbox"], w, h)

        self.canvas.set_crop(dx1, dy1, dx2, dy2)
        self.canvas.set_reference_crop(mx1, my1, mx2, my2)
        self._on_crop_change()
        self._update_status()

    def _on_crop_change(self):
        x1, y1, x2, y2 = self.canvas.get_crop()
        w = x2 - x1
        h = y2 - y1
        if self._pil_img:
            iw, ih = self._pil_img.size
            area_pct = (w * h) / (iw * ih) * 100
            self.lbl_crop_info.config(
                text=f"Recorte: {w}x{h}px  |  Area: {area_pct:.1f}%  |  ({x1}, {y1}) -> ({x2}, {y2})",
            )

    def _reset_crop(self):
        self.canvas.reset_crop()

    # ── Ações ────────────────────────────────────────────────────

    def skip_current(self):
        """Manter o crop atual, ir para próxima."""
        log.debug(f"Pulada: {self.pairs[self.current_index]['orig_path'].name}")
        self._advance()

    def accept_model(self):
        """Substituir crop atual pelo do modelo."""
        if self._pil_img is None:
            return
        pair = self.pairs[self.current_index]
        w, h = self._pil_img.size
        mx1, my1, mx2, my2 = _denorm_bbox(pair["model_bbox"], w, h)
        self._save_crop(mx1, my1, mx2, my2, source="modelo")

    def accept_adjustment(self):
        """Substituir crop atual pelo ajuste manual do usuário."""
        if self._pil_img is None:
            return
        x1, y1, x2, y2 = self.canvas.get_crop()
        pair = self.pairs[self.current_index]
        w, h = self._pil_img.size
        dx1, dy1, dx2, dy2 = _denorm_bbox(pair["dataset_bbox"], w, h)

        # Se coordenadas são idênticas ao dataset, é o mesmo que pular
        if (x1, y1, x2, y2) == (dx1, dy1, dx2, dy2):
            log.debug(f"Sem alteração: {pair['orig_path'].name}")
            self._advance()
            return

        self._save_crop(x1, y1, x2, y2, source="ajuste manual")

    def _save_crop(self, x1: int, y1: int, x2: int, y2: int, source: str):
        """Salva o novo crop substituindo o existente."""
        pair = self.pairs[self.current_index]
        crop_path = pair["crop_path"]

        # Backup do arquivo antigo para undo
        try:
            with open(crop_path, "rb") as f:
                old_bytes = f.read()
        except OSError as e:
            log.error(f"Falha ao ler crop antigo: {e}")
            old_bytes = b""

        # Salvar novo crop
        try:
            cropped = self._pil_img.crop((x1, y1, x2, y2))
            cropped.save(str(crop_path), "JPEG", quality=98)
        except OSError as e:
            messagebox.showerror("Erro ao salvar", str(e), parent=self.root)
            return

        self._undo_stack.append((self.current_index, old_bytes))
        self.modified.add(self.current_index)
        self.file_list.mark_modified(self.current_index)
        log.info(
            f"Modificada ({source}): {pair['orig_path'].name} "
            f"crop=({x1},{y1})-({x2},{y2})",
        )
        self._update_status()
        self._advance()

    def go_back(self):
        if not self._history:
            return
        prev = self._history.pop()
        self.file_list.select_index(prev)

    def undo_last(self):
        if not self._undo_stack:
            return
        idx, old_bytes = self._undo_stack.pop()
        pair = self.pairs[idx]
        crop_path = pair["crop_path"]

        if old_bytes:
            try:
                with open(crop_path, "wb") as f:
                    f.write(old_bytes)
                log.info(f"Desfeito: {pair['orig_path'].name}")
            except OSError as e:
                messagebox.showerror("Erro ao desfazer", str(e), parent=self.root)
                return

        self.modified.discard(idx)
        self.file_list.unmark_modified(idx)
        self._update_status()
        self.file_list.select_index(idx)

    def _advance(self):
        total = len(self.pairs)
        if total == 0:
            return
        next_idx = self.current_index + 1
        if next_idx >= total:
            messagebox.showinfo(
                "Fim", f"Todas as {total} imagens foram revisadas.\n"
                f"Modificadas: {len(self.modified)}",
            )
            return
        self.file_list.select_index(next_idx)

    def _update_status(self):
        total = len(self.pairs)
        done = len(self.modified)
        pair = self.pairs[self.current_index]
        mark = " (modificada)" if self.current_index in self.modified else ""
        self.status_var.set(
            f"{self.current_index + 1}/{total}  |  "
            f"Modificadas: {done}  |  IoU: {pair['iou']:.4f}{mark}",
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _setup_logging()

    # Splash / progresso
    root = tk.Tk()
    root.title("Ajustar Dataset - Carregando...")
    root.configure(bg="#2e2e2e")
    root.geometry("520x160")
    root.resizable(False, False)

    status_label = tk.Label(
        root, text="Carregando...", bg="#2e2e2e", fg="#ffffff",
        font=("Segoe UI", 11),
    )
    status_label.pack(expand=True, pady=(20, 5))

    style = ttk.Style(root)
    style.theme_use("clam")
    progress = ttk.Progressbar(root, mode="determinate", length=440)
    progress.pack(pady=(0, 20), padx=40)
    root.update()

    # 1. Carregar pares
    status_label.config(text="Carregando pares do dataset...")
    root.update()
    orig_paths, crop_paths = _load_dataset_pairs()
    if not orig_paths:
        messagebox.showerror("Erro", "Nenhum par origin/cropped encontrado no dataset.")
        root.destroy()
        return
    log.info(f"Encontrados {len(orig_paths)} pares no dataset")

    # 2. Carregar modelo
    status_label.config(text="Carregando modelo...")
    root.update()
    try:
        model, img_size, train_iou, train_margin_err = _load_model()
        log.info(f"Modelo carregado (IoU: {train_iou:.4f})")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao carregar modelo:\n{e}")
        root.destroy()
        return

    # 3. Bboxes do dataset
    status_label.config(text="Carregando bboxes do dataset...")
    root.update()
    dataset_bboxes = _load_dataset_bboxes(orig_paths, crop_paths)

    # 4. Predições do modelo
    progress.config(maximum=len(orig_paths))

    def _on_progress(current: int, total: int):
        progress["value"] = current
        status_label.config(text=f"Predições do modelo: {current}/{total}")
        root.update()

    predictions = _batch_predict(
        model, orig_paths, img_size,
        batch_size=64, max_workers=8,
        progress_callback=_on_progress,
    )

    # 5. Calcular discrepâncias e ordenar
    status_label.config(text="Calculando discrepâncias...")
    root.update()

    pairs_data: List[dict] = []
    for i in range(len(orig_paths)):
        iou_val = _iou(dataset_bboxes[i], predictions[i])
        pairs_data.append({
            "orig_path": orig_paths[i],
            "crop_path": crop_paths[i],
            "dataset_bbox": dataset_bboxes[i],
            "model_bbox": predictions[i],
            "iou": iou_val,
        })

    pairs_data.sort(key=lambda x: x["iou"])  # Menor IoU primeiro
    log.info(
        f"Discrepâncias calculadas. "
        f"Min IoU: {pairs_data[0]['iou']:.4f} | "
        f"Max IoU: {pairs_data[-1]['iou']:.4f} | "
        f"Median IoU: {pairs_data[len(pairs_data)//2]['iou']:.4f}",
    )

    # 6. Abrir app principal
    root.destroy()
    app_root = tk.Tk()
    AdjusterApp(app_root, pairs_data, model, img_size)
    app_root.mainloop()


if __name__ == "__main__":
    main()
