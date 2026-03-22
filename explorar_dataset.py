"""Explorador interativo do dataset de recorte.

Exibe gráficos de distribuição por faixa (excesso/escassez) com exemplos visuais
navegáveis. Não salva arquivos — toda visualização é em memória.
"""

import datetime
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from analise_dataset import (
    IMAGE_EXTENSIONS,
    DEFAULT_BINS,
    PairRecord,
    BinStatus,
    list_images,
    load_cache_bboxes,
    resolve_pairs,
    summarize_numeric,
    classify_bins,
)


# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    script_name = Path(__file__).stem
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = Path("logs") / f"{script_name}_{timestamp}.log"
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("explorar_dataset")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_filename), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logging.captureWarnings(True)
    return logger


log = logging.getLogger("explorar_dataset")

# ── Constantes visuais ────────────────────────────────────────────────────────
BG = "#1e1e1e"
BG2 = "#2e2e2e"
FG = "#e0e0e0"
ACCENT = "#0078d4"
RED = "#e45756"
ORANGE = "#f58518"
GREEN = "#54a24b"
BLUE = "#4c78a8"
TEAL = "#72b7b2"
STATUS_COLORS = {"EXCESSO": RED, "FALTANDO": ORANGE, "OK": GREEN}
THUMB_H = 720


# ── Helpers de imagem ─────────────────────────────────────────────────────────
def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    oh, ow = img.shape[:2]
    if oh <= 0 or ow <= 0:
        return np.zeros((h, h, 3), dtype=np.uint8)
    scale = h / oh
    return cv2.resize(img, (max(1, int(ow * scale)), h), interpolation=cv2.INTER_AREA)


def _draw_bbox(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    x1 = int(np.clip(bbox[0] * w, 0, w - 1))
    y1 = int(np.clip(bbox[1] * h, 0, h - 1))
    x2 = int(np.clip(bbox[2] * w, 0, w - 1))
    y2 = int(np.clip(bbox[3] * h, 0, h - 1))
    cv2.rectangle(out, (x1, y1), (x2, y2), (35, 220, 35), 2)
    return out


def _build_panel(rec: PairRecord) -> Optional[Image.Image]:
    orig = cv2.imread(str(rec.orig_path))
    crop = cv2.imread(str(rec.crop_path))
    if orig is None or crop is None:
        return None
    left = _resize_h(_draw_bbox(orig, rec.bbox), THUMB_H)
    right = _resize_h(crop, THUMB_H)
    sep = np.full((THUMB_H, 6, 3), 40, dtype=np.uint8)
    panel = np.concatenate([left, sep, right], axis=1)
    panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    return Image.fromarray(panel_rgb)


# ── Matplotlib → PIL ─────────────────────────────────────────────────────────
def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1)
    plt.close(fig)
    return img.convert("RGB")


# ── Geração dos gráficos ─────────────────────────────────────────────────────
def _chart_balance(
    area_balance: List[BinStatus],
) -> Image.Image:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG)

    labels = [b.label for b in area_balance]
    counts = [b.count for b in area_balance]
    ideal = [b.ideal_count for b in area_balance]
    colors = [STATUS_COLORS[b.status] for b in area_balance]

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="#555", linewidth=0.6)
    ax.plot(x, ideal, color=ACCENT, linestyle="--", linewidth=2, label="ideal", marker="o", markersize=4)

    for bar, b in zip(bars, area_balance):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{b.count}\n{b.status}",
            ha="center", va="bottom", fontsize=8, color=FG, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color=FG)
    ax.set_ylabel("Quantidade", color=FG)
    ax.set_title("Balanceamento por Faixa de Área", color=FG, fontsize=13, fontweight="bold")
    ax.legend(facecolor=BG2, edgecolor="#555", labelcolor=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#555")
    fig.tight_layout()
    return _fig_to_pil(fig)


def _chart_area_hist(areas: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG)
    ax.hist(areas, bins=25, color=TEAL, edgecolor="#333", alpha=0.9)
    ax.set_title("Distribuição de Área do Recorte", color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Área normalizada", color=FG)
    ax.set_ylabel("Quantidade", color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#555")
    fig.tight_layout()
    return _fig_to_pil(fig)


def _chart_margins(margins: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG)
    bp = ax.boxplot(
        [margins[:, i] for i in range(4)],
        labels=["Esquerda", "Topo", "Direita", "Base"],
        patch_artist=True,
        boxprops={"facecolor": TEAL, "alpha": 0.8, "edgecolor": FG},
        whiskerprops={"color": FG},
        capprops={"color": FG},
        medianprops={"color": RED, "linewidth": 2},
        flierprops={"markeredgecolor": ORANGE, "markersize": 3},
    )
    ax.set_title("Distribuição das Margens", color=FG, fontsize=12, fontweight="bold")
    ax.set_ylabel("Margem normalizada", color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#555")
    fig.tight_layout()
    return _fig_to_pil(fig)


def _chart_aspect_scatter(records: List[PairRecord], areas: np.ndarray) -> Image.Image:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG)
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    sc = ax.scatter(widths, heights, c=areas, cmap="viridis", alpha=0.65, s=18, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Área bbox", color=FG)
    cbar.ax.tick_params(colors=FG)
    ax.set_title("Resolução das Imagens", color=FG, fontsize=12, fontweight="bold")
    ax.set_xlabel("Largura", color=FG)
    ax.set_ylabel("Altura", color=FG)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#555")
    fig.tight_layout()
    return _fig_to_pil(fig)


def _chart_summary(
    total: int,
    n_pairs: int,
    n_missing: int,
    n_orphan: int,
    area_stats: Dict[str, float],
) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.patch.set_facecolor(BG2)

    ax0 = axes[0]
    ax0.set_facecolor(BG)
    cats = ["Pares válidos", "Sem crop", "Crop órfão"]
    vals = [n_pairs, n_missing, n_orphan]
    cols = [GREEN, ORANGE, RED]
    ax0.barh(cats, vals, color=cols, edgecolor="#555")
    for i, v in enumerate(vals):
        ax0.text(v + max(vals) * 0.02, i, str(v), va="center", color=FG, fontweight="bold")
    ax0.set_title("Integridade", color=FG, fontsize=12, fontweight="bold")
    ax0.tick_params(colors=FG)
    ax0.invert_yaxis()
    for spine in ax0.spines.values():
        spine.set_color("#555")

    ax1 = axes[1]
    ax1.axis("off")
    stat_lines = [
        f"Total originais: {total}",
        f"Pares válidos: {n_pairs}",
        "",
        f"Área mín: {area_stats['min']:.4f}   máx: {area_stats['max']:.4f}",
        f"Área média: {area_stats['mean']:.4f}   mediana: {area_stats['median']:.4f}",
        f"Área p25: {area_stats['p25']:.4f}   p75: {area_stats['p75']:.4f}",
        f"Desvio padrão: {area_stats['std']:.4f}",
    ]
    ax1.text(
        0.05, 0.95, "\n".join(stat_lines),
        transform=ax1.transAxes, va="top", fontsize=11, color=FG,
        family="monospace", linespacing=1.6,
    )
    ax1.set_title("Resumo Estatístico", color=FG, fontsize=12, fontweight="bold")

    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Motor de análise ─────────────────────────────────────────────────────────
class AnalysisResult:
    __slots__ = (
        "records", "missing_crop", "orphan_crop",
        "areas", "margins", "area_stats", "aspect_stats", "margin_stats",
        "area_balance", "total_origin", "total_crop",
    )

    def __init__(
        self,
        records: List[PairRecord],
        missing_crop: List[Path],
        orphan_crop: List[Path],
        area_balance: List[BinStatus],
        total_origin: int,
        total_crop: int,
    ):
        self.records = records
        self.missing_crop = missing_crop
        self.orphan_crop = orphan_crop
        self.area_balance = area_balance
        self.total_origin = total_origin
        self.total_crop = total_crop

        self.areas = np.array([r.area for r in records], dtype=np.float32)
        self.margins = np.vstack([r.margins for r in records])
        self.area_stats = summarize_numeric(self.areas)
        self.aspect_stats = summarize_numeric(
            np.array([r.aspect_ratio for r in records], dtype=np.float32)
        )
        self.margin_stats = {
            side: summarize_numeric(self.margins[:, i])
            for i, side in enumerate(["left", "top", "right", "bottom"])
        }


def run_analysis(
    origin_dir: Path,
    crop_dir: Path,
    cache_path: Path,
    bins: Sequence[float] = DEFAULT_BINS,
    threshold: float = 0.25,
    max_workers: int = 0,
) -> AnalysisResult:
    if max_workers <= 0:
        max_workers = max(1, (os.cpu_count() or 4) - 1)

    origin_files = list_images(origin_dir)
    crop_files = list_images(crop_dir)
    log.info("Originais: %d | Crops: %d", len(origin_files), len(crop_files))

    cache_map = load_cache_bboxes(cache_path)
    records, missing, orphan = resolve_pairs(origin_files, crop_files, cache_map, max_workers)
    log.info("Pares válidos: %d | Sem crop: %d | Orfãos: %d", len(records), len(missing), len(orphan))

    areas = np.array([r.area for r in records], dtype=np.float32)
    area_balance = classify_bins(areas, bins=bins, threshold=threshold)

    return AnalysisResult(
        records=records,
        missing_crop=missing,
        orphan_crop=orphan,
        area_balance=area_balance,
        total_origin=len(origin_files),
        total_crop=len(crop_files),
    )


# ── Interface ─────────────────────────────────────────────────────────────────
class ExplorerApp:
    def __init__(self, root: tk.Tk, result: AnalysisResult):
        self.root = root
        self.result = result
        self._photo_refs: List[ImageTk.PhotoImage] = []
        self._example_panels: Dict[int, List[Image.Image]] = {}
        self._current_bin: int = 0
        self._current_example: int = 0

        self._resize_jobs: Dict[str, Optional[str]] = {}
        self._chart_imgs: Dict[str, Image.Image] = {}

        self._apply_theme()
        self._build_ui()
        self._bind_keys()
        self._show_overview()
        if result.area_balance:
            self._select_bin(0)

    def _apply_theme(self):
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TButton", background="#444", foreground=FG)
        style.configure("TScrollbar", background="#444", troughcolor=BG2)
        style.map("TButton", background=[("active", "#555")])

        style.configure("Bin.TButton", padding=(8, 6), font=("Segoe UI", 9))
        for status, color in STATUS_COLORS.items():
            style.configure(
                f"{status}.TButton",
                background=color,
                foreground="#ffffff",
                padding=(8, 6),
                font=("Segoe UI", 9, "bold"),
            )
            style.map(f"{status}.TButton", background=[("active", color)])

    def _bind_keys(self):
        self.root.bind("<Right>", lambda _: self._next_example())
        self.root.bind("<Left>", lambda _: self._prev_example())
        self.root.bind("<Escape>", lambda _: self.root.destroy())
        for i in range(min(9, len(self.result.area_balance))):
            num = i + 1
            self.root.bind(str(num), lambda _, idx=i: self._select_bin(idx))

    def _build_ui(self):
        self.root.title("Explorador de Dataset")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ── Topo: overview geral ──────────────────────────────────
        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top.columnconfigure(0, weight=1)

        self.overview_label = ttk.Label(top, anchor="center")
        self.overview_label.grid(row=0, column=0, sticky="ew")

        # ── Centro: split esquerda/direita ────────────────────────
        center = ttk.Frame(self.root)
        center.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        center.columnconfigure(1, weight=1)
        center.rowconfigure(0, weight=1)

        # Painel esquerdo: botões das faixas + gráfico de barras
        left = ttk.Frame(center, width=420)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        bins_frame = ttk.Frame(left)
        bins_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.bin_buttons: List[ttk.Button] = []
        for i, b in enumerate(self.result.area_balance):
            btn_style = f"{b.status}.TButton"
            btn = ttk.Button(
                bins_frame,
                text=f"{b.label}\n{b.count} ({b.status})",
                style=btn_style,
                command=lambda idx=i: self._select_bin(idx),
            )
            btn.grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky="ew")
            bins_frame.columnconfigure(i % 3, weight=1)
            self.bin_buttons.append(btn)

        # Canvas para gráfico de barras
        chart_container = ttk.Frame(left)
        chart_container.grid(row=1, column=0, sticky="nsew")
        chart_container.grid_propagate(False)
        chart_container.columnconfigure(0, weight=1)
        chart_container.rowconfigure(0, weight=1)

        self.chart_label = ttk.Label(chart_container, anchor="center")
        self.chart_label.grid(row=0, column=0, sticky="nsew")
        chart_container.bind("<Configure>", lambda e: self._debounced_refit("chart", self.chart_label))

        # Painel direito: aba de gráficos / exemplos
        right = ttk.Frame(center)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Tabs de conteúdo
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=0, column=0, rowspan=2, sticky="nsew")

        # Tab: Exemplos
        self.tab_examples = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_examples, text="  Exemplos  ")
        self.tab_examples.columnconfigure(0, weight=1)
        self.tab_examples.rowconfigure(0, weight=1)

        ex_container = ttk.Frame(self.tab_examples)
        ex_container.grid(row=0, column=0, sticky="nsew")
        ex_container.grid_propagate(False)
        ex_container.columnconfigure(0, weight=1)
        ex_container.rowconfigure(0, weight=1)

        self.example_label = ttk.Label(ex_container, anchor="center")
        self.example_label.grid(row=0, column=0, sticky="nsew")
        ex_container.bind("<Configure>", lambda e: self._debounced_refit("example", self.example_label))

        nav_frame = ttk.Frame(self.tab_examples)
        nav_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        nav_frame.columnconfigure(1, weight=1)
        ttk.Button(nav_frame, text="← Anterior", command=self._prev_example).grid(
            row=0, column=0, padx=(0, 6)
        )
        self.example_info_var = tk.StringVar()
        ttk.Label(nav_frame, textvariable=self.example_info_var, anchor="center").grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(nav_frame, text="Próximo →", command=self._next_example).grid(
            row=0, column=2, padx=(6, 0)
        )

        # Tab: Histograma de áreas
        self.tab_hist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hist, text="  Histograma  ")
        self.tab_hist.columnconfigure(0, weight=1)
        self.tab_hist.rowconfigure(0, weight=1)
        self._hist_label = ttk.Label(self.tab_hist, anchor="center")
        self._hist_label.grid(row=0, column=0, sticky="nsew")
        self.tab_hist.bind("<Configure>", lambda e: self._debounced_refit("hist", self._hist_label))

        # Tab: Margens
        self.tab_margins = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_margins, text="  Margens  ")
        self.tab_margins.columnconfigure(0, weight=1)
        self.tab_margins.rowconfigure(0, weight=1)
        self._margins_label = ttk.Label(self.tab_margins, anchor="center")
        self._margins_label.grid(row=0, column=0, sticky="nsew")
        self.tab_margins.bind("<Configure>", lambda e: self._debounced_refit("margins", self._margins_label))

        # Tab: Resolução
        self.tab_scatter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_scatter, text="  Resolução  ")
        self.tab_scatter.columnconfigure(0, weight=1)
        self.tab_scatter.rowconfigure(0, weight=1)
        self._scatter_label = ttk.Label(self.tab_scatter, anchor="center")
        self._scatter_label.grid(row=0, column=0, sticky="nsew")
        self.tab_scatter.bind("<Configure>", lambda e: self._debounced_refit("scatter", self._scatter_label))

        # ── Rodapé: status ────────────────────────────────────────
        bottom = ttk.Frame(self.root)
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        bottom.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status_var, anchor="center").grid(
            row=0, column=0, sticky="ew"
        )

    # ── Debounced refit ──────────────────────────────────────────

    def _debounced_refit(self, key: str, label: ttk.Label):
        job = self._resize_jobs.get(key)
        if job:
            self.root.after_cancel(job)
        self._resize_jobs[key] = self.root.after(80, lambda: self._refit(key, label))

    def _refit(self, key: str, label: ttk.Label):
        self._resize_jobs[key] = None
        img = self._chart_imgs.get(key)
        if img:
            self._fit_photo(img, label)

    # ── Renderização ──────────────────────────────────────────────

    def _fit_photo(
        self, pil_img: Image.Image, label: ttk.Label, max_w: int = 0, max_h: int = 0
    ) -> ImageTk.PhotoImage:
        label.update_idletasks()
        w = max_w or max(label.winfo_width(), 300)
        h = max_h or max(label.winfo_height(), 300)
        iw, ih = pil_img.size
        scale = min(w / max(iw, 1), h / max(ih, 1))
        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))
        img = pil_img.resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo)
        self._photo_refs.append(photo)
        return photo

    def _show_overview(self):
        r = self.result
        summary = (
            f"Pares: {len(r.records)}  |  Sem crop: {len(r.missing_crop)}  |  "
            f"Órfãos: {len(r.orphan_crop)}  |  "
            f"Área média: {r.area_stats['mean']:.4f}  |  "
            f"Mediana: {r.area_stats['median']:.4f}"
        )
        self.overview_label.config(text=summary, font=("Segoe UI", 11))

        # Gráfico de barras (canto esquerdo)
        chart_img = _chart_balance(r.area_balance)
        self._chart_imgs["chart"] = chart_img
        self.root.after_idle(lambda: self._fit_photo(chart_img, self.chart_label))

        # Gráficos das demais abas
        self.root.after(100, self._render_secondary_charts)

    def _render_secondary_charts(self):
        r = self.result
        # Histograma
        hist_img = _chart_area_hist(r.areas)
        self._chart_imgs["hist"] = hist_img
        self._fit_photo(hist_img, self._hist_label)
        # Margens
        margins_img = _chart_margins(r.margins)
        self._chart_imgs["margins"] = margins_img
        self._fit_photo(margins_img, self._margins_label)
        # Resolução
        scatter_img = _chart_aspect_scatter(r.records, r.areas)
        self._chart_imgs["scatter"] = scatter_img
        self._fit_photo(scatter_img, self._scatter_label)

    # ── Navegação de faixas ───────────────────────────────────────

    def _select_bin(self, idx: int):
        self._current_bin = idx
        self._current_example = 0
        b = self.result.area_balance[idx]
        self.status_var.set(
            f"Faixa {b.label}  |  {b.count} imagens  |  Status: {b.status}  |  "
            f"Ideal: {b.ideal_count:.0f}"
        )
        self.notebook.select(self.tab_examples)
        self._load_bin_examples(idx)
        self._show_example()

    def _load_bin_examples(self, idx: int):
        if idx in self._example_panels:
            return
        b = self.result.area_balance[idx]
        indices = b.indices
        # Selecionar exemplos espaçados (até 20)
        max_ex = min(20, len(indices))
        if max_ex <= 0:
            self._example_panels[idx] = []
            return
        if len(indices) <= max_ex:
            selected = indices
        else:
            positions = np.linspace(0, len(indices) - 1, num=max_ex, dtype=int)
            selected = [indices[p] for p in positions]

        panels: List[Image.Image] = []
        for rec_idx in selected:
            rec = self.result.records[rec_idx]
            panel = _build_panel(rec)
            if panel:
                panels.append(panel)
        self._example_panels[idx] = panels
        log.info("Carregados %d exemplos para faixa %s", len(panels), b.label)

    def _show_example(self):
        panels = self._example_panels.get(self._current_bin, [])
        if not panels:
            self.example_label.configure(image="")
            self.example_info_var.set("Sem exemplos nesta faixa")
            return

        idx = self._current_example % len(panels)
        self._current_example = idx
        panel = panels[idx]

        b = self.result.area_balance[self._current_bin]
        rec_indices = b.indices
        max_ex = min(20, len(rec_indices))
        if len(rec_indices) <= max_ex:
            selected = rec_indices
        else:
            positions = np.linspace(0, len(rec_indices) - 1, num=max_ex, dtype=int)
            selected = [rec_indices[p] for p in positions]

        rec = self.result.records[selected[idx]]
        area_pct = rec.area * 100

        self._chart_imgs["example"] = panel
        self._fit_photo(panel, self.example_label)
        self.example_info_var.set(
            f"{idx + 1}/{len(panels)}  |  {rec.key}  |  "
            f"Área: {area_pct:.1f}%  |  {rec.width}×{rec.height}px"
        )

    def _next_example(self):
        panels = self._example_panels.get(self._current_bin, [])
        if panels:
            self._current_example = (self._current_example + 1) % len(panels)
            self._show_example()

    def _prev_example(self):
        panels = self._example_panels.get(self._current_bin, [])
        if panels:
            self._current_example = (self._current_example - 1) % len(panels)
            self._show_example()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _setup_logging()

    project_root = Path(__file__).resolve().parent
    origin_dir = project_root / "dataset" / "origin"
    crop_dir = project_root / "dataset" / "cropped"
    cache_path = project_root / "models" / "bbox_cache.pkl"

    if not origin_dir.exists() or not crop_dir.exists():
        log.error("Pastas dataset/origin e dataset/cropped não encontradas")
        return

    log.info("Calculando análise do dataset...")
    result = run_analysis(origin_dir, crop_dir, cache_path)

    if not result.records:
        log.error("Nenhum par válido encontrado")
        return

    log.info("Abrindo explorador interativo...")
    root = tk.Tk()
    ExplorerApp(root, result)
    root.mainloop()


if __name__ == "__main__":
    main()
