"""Rotulagem assistida por modelo para geração rápida de dataset.

Workflow:
1. Selecione pasta com imagens originais
2. Modelo prediz coordenadas de recorte automaticamente
3. Aceite (→), ajuste arrastando, ou pule (←)
4. Pares aceitos vão para dataset/origin + dataset/cropped
"""

import datetime
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms

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
    logger = logging.getLogger("gerar_dataset")
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


log: logging.Logger = logging.getLogger("gerar_dataset")


# ── Predição ──────────────────────────────────────────────────────────────────
def _predict_bbox(
    model: MarginAwareCropModel, pil_img: Image.Image, img_size: int
) -> Tuple[int, int, int, int]:
    orig_w, orig_h = pil_img.size
    img_resized = pil_img.resize((img_size, img_size), Image.LANCZOS)
    tensor = transforms.ToTensor()(img_resized)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(tensor).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = model(tensor)[0].cpu().numpy()
        else:
            pred = model(tensor)[0].cpu().numpy()

    x1 = int(np.clip(np.round(pred[0] * orig_w), 0, orig_w - 1))
    y1 = int(np.clip(np.round(pred[1] * orig_h), 0, orig_h - 1))
    x2 = int(np.clip(np.round(pred[2] * orig_w), x1 + 1, orig_w))
    y2 = int(np.clip(np.round(pred[3] * orig_h), y1 + 1, orig_h))
    return x1, y1, x2, y2


def _scan_images(folder: Path) -> List[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "_editado" not in p.stem.lower()
    )


# ── Canvas interativo ────────────────────────────────────────────────────────
class CropCanvas(tk.Canvas):
    """Canvas com imagem e retângulo de recorte interativo."""

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

        self._overlay_ids: List[int] = []
        self._rect_id: Optional[int] = None
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

    def get_crop(self) -> Tuple[int, int, int, int]:
        return tuple(self._crop)

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

        stip = "gray50"
        oc = "#000000"
        for coords in [
            (ix1, iy1, ix2, cy1),   # topo
            (ix1, cy2, ix2, iy2),   # base
            (ix1, cy1, cx1, cy2),   # esquerda
            (cx2, cy1, ix2, cy2),   # direita
        ]:
            rid = self.create_rectangle(*coords, fill=oc, stipple=stip, outline="")
            self._overlay_ids.append(rid)

        self._rect_id = self.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ff00", width=2)

        # Alças nos 4 cantos
        for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]:
            hid = self.create_rectangle(
                hx - HANDLE_SIZE, hy - HANDLE_SIZE,
                hx + HANDLE_SIZE, hy + HANDLE_SIZE,
                fill="#00ff00", outline="#ffffff", width=1,
            )
            self._handle_ids.append(hid)

        # Alças nos 4 pontos médios das arestas
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

        # Cantos primeiro (prioridade sobre arestas)
        for name, hx, hy in [
            ("nw", cx1, cy1), ("ne", cx2, cy1),
            ("sw", cx1, cy2), ("se", cx2, cy2),
        ]:
            if abs(cx - hx) <= HANDLE_HIT and abs(cy - hy) <= HANDLE_HIT:
                return name

        # Arestas
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
            self,
            activestyle="dotbox",
            exportselection=False,
            bg="#3a3a3a",
            fg="#ffffff",
            selectbackground="#0078d4",
            selectforeground="#ffffff",
            font=("Consolas", 9),
        )
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.scroll.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def populate(self, paths: List[Path]):
        self.listbox.delete(0, tk.END)
        for p in paths:
            self.listbox.insert(tk.END, p.name)

    def mark_accepted(self, index: int):
        try:
            label = self.listbox.get(index)
        except tk.TclError:
            return
        if not label.startswith("✓ "):
            self.listbox.delete(index)
            self.listbox.insert(index, f"✓ {label}")

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


# ── Aplicação principal ──────────────────────────────────────────────────────
class LabelerApp:
    def __init__(
        self,
        root: tk.Tk,
        image_paths: List[Path],
        model: MarginAwareCropModel,
        img_size: int,
    ):
        self.root = root
        self.image_paths = image_paths
        self.model = model
        self.img_size = img_size
        self.current_index = 0
        self.accepted: Set[int] = set()
        self.project_root = Path(__file__).resolve().parent
        self._pil_img: Optional[Image.Image] = None
        self._predictions: dict[int, Tuple[int, int, int, int]] = {}

        self._apply_dark_theme()
        self._build_ui()
        self._bind_keys()
        if self.image_paths:
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
        self.root.bind("<Right>", lambda _e: self.accept_current())
        self.root.bind("<Left>", lambda _e: self.skip_current())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<r>", lambda _e: self._reset_crop())
        self.root.bind("<R>", lambda _e: self._reset_crop())

    def _build_ui(self):
        self.root.title("Rotulagem Assistida - Geração de Dataset")
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

        # Container do canvas (propagate=False evita cascata de resize)
        canvas_container = ttk.Frame(right)
        canvas_container.grid(row=1, column=0, sticky="nsew")
        canvas_container.grid_propagate(False)
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        self.canvas = CropCanvas(canvas_container, on_crop_change=self._on_crop_change)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Informações do recorte
        self.lbl_crop_info = ttk.Label(right, text="", anchor="center")
        self.lbl_crop_info.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # Botões
        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for i in range(5):
            btn_frame.columnconfigure(i, weight=1)

        ttk.Button(btn_frame, text="Aceitar (→)", command=self.accept_current).grid(
            row=0, column=0, padx=(0, 6), sticky="ew"
        )
        ttk.Button(btn_frame, text="Pular (←)", command=self.skip_current).grid(
            row=0, column=1, padx=(0, 6), sticky="ew"
        )
        ttk.Button(btn_frame, text="Reset (R)", command=self._reset_crop).grid(
            row=0, column=2, padx=(0, 6), sticky="ew"
        )
        ttk.Button(btn_frame, text="Abrir Pasta", command=self._choose_directory).grid(
            row=0, column=3, padx=(0, 6), sticky="ew"
        )
        ttk.Button(btn_frame, text="Sair (Esc)", command=self.root.destroy).grid(
            row=0, column=4, sticky="ew"
        )

        # Status
        self.status_var = tk.StringVar()
        ttk.Label(right, textvariable=self.status_var, anchor="center").grid(
            row=4, column=0, sticky="ew", pady=(8, 0)
        )

    # ── Navegação ────────────────────────────────────────────────

    def _on_select(self, index: int):
        self.current_index = index
        path = self.image_paths[index]
        self.lbl_title.config(text=f"Original: {path.name}")

        try:
            self._pil_img = Image.open(path).convert("RGB")
        except OSError:
            self._pil_img = Image.new("RGB", (100, 100), "#444")

        self.canvas.set_image(self._pil_img)

        if index not in self._predictions:
            try:
                bbox = _predict_bbox(self.model, self._pil_img, self.img_size)
                self._predictions[index] = bbox
            except Exception as e:
                log.error(f"Erro na predição para {path.name}: {e}")
                w, h = self._pil_img.size
                bbox = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
                self._predictions[index] = bbox

        self.canvas.set_crop(*self._predictions[index])
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
                text=f"Recorte: {w}×{h}px  |  Área: {area_pct:.1f}%  |  ({x1}, {y1}) → ({x2}, {y2})"
            )

    def _reset_crop(self):
        self.canvas.reset_crop()

    # ── Ações ────────────────────────────────────────────────────

    def accept_current(self):
        if self.current_index in self.accepted:
            self._advance()
            return

        path = self.image_paths[self.current_index]
        x1, y1, x2, y2 = self.canvas.get_crop()

        try:
            self._save_pair(path, x1, y1, x2, y2)
        except OSError as e:
            messagebox.showerror("Erro ao salvar", str(e), parent=self.root)
            return

        self.accepted.add(self.current_index)
        self.file_list.mark_accepted(self.current_index)
        log.info(f"Aceita: {path.name} crop=({x1},{y1})-({x2},{y2})")
        self._update_status()
        self._advance()

    def skip_current(self):
        self._advance()

    def _save_pair(self, orig_path: Path, x1: int, y1: int, x2: int, y2: int):
        dataset = self.project_root / "dataset"
        origin_dir = dataset / "origin"
        cropped_dir = dataset / "cropped"
        origin_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)

        # Copiar original
        dest_orig = origin_dir / orig_path.name
        if dest_orig.exists():
            log.warning(f"Substituindo {dest_orig} já existente")
        shutil.copy2(orig_path, dest_orig)

        # Recortar e salvar
        cropped = self._pil_img.crop((x1, y1, x2, y2))
        crop_name = f"{orig_path.stem}_editado.jpg"
        cropped.save(str(cropped_dir / crop_name), "JPEG", quality=98)

    def _advance(self):
        total = len(self.image_paths)
        if total == 0:
            return
        if len(self.accepted) == total:
            messagebox.showinfo("Concluído", "Todas as imagens foram processadas.")
            self.root.destroy()
            return

        next_idx = (self.current_index + 1) % total
        attempts = 0
        while attempts < total and next_idx in self.accepted:
            next_idx = (next_idx + 1) % total
            attempts += 1

        if attempts >= total:
            messagebox.showinfo("Concluído", "Todas as imagens foram processadas.")
            self.root.destroy()
            return

        self.file_list.select_index(next_idx)

    def _update_status(self):
        total = len(self.image_paths)
        done = len(self.accepted)
        remaining = total - done
        mark = " (aceita)" if self.current_index in self.accepted else ""
        self.status_var.set(
            f"{self.current_index + 1}/{total}  |  Restantes: {remaining}  |  Aceitas: {done}{mark}"
        )

    def _choose_directory(self):
        chosen = filedialog.askdirectory(
            parent=self.root, title="Selecione pasta com imagens originais"
        )
        if not chosen:
            return
        new_paths = _scan_images(Path(chosen))
        if not new_paths:
            messagebox.showinfo(
                "Sem imagens", "Nenhuma imagem encontrada.", parent=self.root
            )
            return
        self.image_paths = new_paths
        self.accepted.clear()
        self._predictions.clear()
        self.current_index = 0
        self._pil_img = None
        self.root.title(f"Rotulagem Assistida - {chosen}")
        self.file_list.populate(new_paths)
        self.file_list.select_index(0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _setup_logging()

    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title="Selecione pasta com imagens originais")
    if not chosen:
        root.destroy()
        return

    folder = Path(chosen)
    image_paths = _scan_images(folder)
    if not image_paths:
        messagebox.showinfo("Sem imagens", "Nenhuma imagem encontrada na pasta.")
        root.destroy()
        return

    log.info(f"Pasta selecionada: {folder} ({len(image_paths)} imagens)")

    try:
        model, img_size, train_iou, train_margin_err = _load_model()
        log.info(
            f"Modelo carregado (IoU: {train_iou:.4f}, Margem: {train_margin_err:.6f})"
        )
    except Exception as e:
        log.error(f"Falha ao carregar modelo: {e}")
        messagebox.showerror("Erro", f"Falha ao carregar modelo:\n{e}")
        root.destroy()
        return

    root.destroy()

    app_root = tk.Tk()
    LabelerApp(app_root, image_paths, model, img_size)
    app_root.mainloop()


if __name__ == "__main__":
    main()
