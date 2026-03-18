"""
analise_dataset.py
Ferramenta de análise do dataset: distribuição por faixas de área + visualização de exemplos.
Dependências: pickle, numpy, opencv-python, matplotlib, Pillow, tkinter (stdlib)
"""

import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

CACHE_PATH        = "models/bbox_cache.pkl"
CROP_DIR          = "dataset/cropped"
SAMPLES_PER_RANGE = 5

# ── Paleta ───────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
ACCENT  = "#4f8ef7"
DANGER  = "#f74f4f"
SUCCESS = "#4ff788"
TEXT    = "#e8eaf0"
SUBTEXT = "#7a7f99"
BORDER  = "#2a2d3e"

RANGE_COLORS = ["#4f8ef7","#a78bfa","#34d399","#f59e0b",
                "#f87171","#38bdf8","#fb923c","#a3e635"]


# ── Dados ────────────────────────────────────────────────────────────────────
def load_cache():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    bboxes  = np.array(cache["bboxes"])
    areas   = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return cache["orig_paths"], cache["crop_paths"], bboxes, areas


def find_crop_path(orig_path):
    base = os.path.splitext(os.path.basename(orig_path))[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        for suffix in ("_editado", ""):
            p = os.path.join(CROP_DIR, base + suffix + ext)
            if os.path.exists(p):
                return p
    return None


def compute_ranges(areas, breakpoints):
    edges = [0.0] + sorted(breakpoints) + [1.0]
    total = len(areas)
    result = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (areas >= lo) & (areas <= hi if i == len(edges) - 2 else areas < hi)
        count = int(mask.sum())
        result.append({
            "lo": lo, "hi": hi,
            "label": f"{lo*100:.0f}–{hi*100:.0f}%",
            "count": count,
            "pct": count / total * 100 if total else 0,
            "mask": mask,
            "color": RANGE_COLORS[i % len(RANGE_COLORS)],
        })
    return result


# ── App ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análise de Dataset — Faixas de Área")
        self.configure(bg=BG)
        self.minsize(900, 600)
        self.geometry("1300x820")

        self.orig_paths, self.crop_paths, self.bboxes, self.areas = load_cache()
        self.n = len(self.areas)
        self._ranges_data = []
        self._thumb_refs  = []   # evitar GC das imagens

        self._build_ui()
        self._refresh()

    # ── Layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Toda a janela em grid 3 linhas: header | toolbar | conteúdo
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # ── Header ──────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 0))

        tk.Label(hdr, text="ANÁLISE DE DATASET",
                 font=("Courier New", 17, "bold"), bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text=f"  {self.n} amostras carregadas",
                 font=("Courier New", 11), bg=BG, fg=SUBTEXT).pack(side="left", padx=8)

        # ── Toolbar ─────────────────────────────────────────
        toolbar = tk.Frame(self, bg=PANEL)
        toolbar.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        toolbar.columnconfigure(1, weight=1)   # bp_frame se expande

        tk.Label(toolbar, text="Pontos de corte (% de área):",
                 font=("Courier New", 10, "bold"), bg=PANEL, fg=TEXT
                 ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        self._bp_frame = tk.Frame(toolbar, bg=PANEL)
        self._bp_frame.grid(row=0, column=1, sticky="w", padx=4)
        self._bp_entries: list[tk.Entry] = []
        for v in ["15", "28", "40"]:
            self._add_entry(v)

        btn_frame = tk.Frame(toolbar, bg=PANEL)
        btn_frame.grid(row=0, column=2, padx=12, pady=8)

        def btn(parent, text, color, cmd, fg=BG):
            b = tk.Button(parent, text=text, font=("Courier New", 10),
                          bg=color, fg=fg, relief="flat", cursor="hand2",
                          activebackground=color, padx=10, pady=4, command=cmd)
            b.pack(side="left", padx=4)
            return b

        btn(btn_frame, "＋ Faixa",  ACCENT,   self._add_entry)
        btn(btn_frame, "－ Faixa",  BORDER,   self._remove_entry, fg=TEXT)
        btn(btn_frame, "▶  Aplicar", SUCCESS,  self._refresh)

        # ── Conteúdo principal: left (60%) | right (40%) ────
        content = tk.Frame(self, bg=BG)
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 14))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=3)   # painel esquerdo
        content.columnconfigure(1, weight=2)   # painel direito

        # ── Painel esquerdo ──────────────────────────────────
        left = tk.Frame(content, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=2)   # gráfico
        left.rowconfigure(1, weight=1)   # tabela
        left.columnconfigure(0, weight=1)

        # Gráfico matplotlib – tamanho inicial; redimensiona com a janela
        self._fig, (self._ax_bar, self._ax_pie) = plt.subplots(
            1, 2, facecolor=BG)
        self._fig.subplots_adjust(left=0.07, right=0.97, top=0.87,
                                  bottom=0.16, wspace=0.38)
        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=left)
        mpl_widget = self._mpl_canvas.get_tk_widget()
        mpl_widget.configure(bg=BG, highlightthickness=0)
        mpl_widget.grid(row=0, column=0, sticky="nsew")
        # Redimensionar figura quando o widget mudar de tamanho
        mpl_widget.bind("<Configure>", self._on_chart_resize)

        # Tabela
        table_frame = tk.Frame(left, bg=BG)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("D.Treeview", background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=28,
                        font=("Courier New", 10))
        style.configure("D.Treeview.Heading", background=BORDER,
                        foreground=ACCENT, font=("Courier New", 10, "bold"),
                        relief="flat")
        style.map("D.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", BG)])

        cols = ("Faixa", "Qtd", "%", "Status")
        self._tree = ttk.Treeview(table_frame, columns=cols,
                                  show="headings", style="D.Treeview")
        for col, w, anchor in zip(cols, [140, 80, 80, 130],
                                  ["center","center","center","center"]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor=anchor, stretch=True)

        vsb = ttk.Scrollbar(table_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

        # ── Painel direito (exemplos) ────────────────────────
        right = tk.Frame(content, bg=PANEL)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="EXEMPLOS DA FAIXA SELECIONADA",
                 font=("Courier New", 10, "bold"), bg=PANEL, fg=ACCENT
                 ).grid(row=0, column=0, pady=(10, 2))

        scroll_host = tk.Frame(right, bg=PANEL)
        scroll_host.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 8))
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)

        self._ex_canvas = tk.Canvas(scroll_host, bg=PANEL,
                                    highlightthickness=0)
        vsb2 = ttk.Scrollbar(scroll_host, orient="vertical",
                              command=self._ex_canvas.yview)
        self._sample_frame = tk.Frame(self._ex_canvas, bg=PANEL)
        self._win_id = self._ex_canvas.create_window(
            (0, 0), window=self._sample_frame, anchor="nw")

        self._sample_frame.bind(
            "<Configure>",
            lambda e: self._ex_canvas.configure(
                scrollregion=self._ex_canvas.bbox("all")))
        self._ex_canvas.bind(
            "<Configure>",
            lambda e: self._ex_canvas.itemconfig(
                self._win_id, width=e.width))
        self._ex_canvas.configure(yscrollcommand=vsb2.set)

        self._ex_canvas.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")

        # Scroll somente quando o mouse está sobre o painel de exemplos
        self._ex_canvas.bind(
            "<Enter>",
            lambda e: self._ex_canvas.bind_all(
                "<MouseWheel>",
                lambda ev: self._ex_canvas.yview_scroll(
                    -1 * (ev.delta // 120), "units")))
        self._ex_canvas.bind(
            "<Leave>",
            lambda e: self._ex_canvas.unbind_all("<MouseWheel>"))

        # Placeholder
        tk.Label(self._sample_frame,
                 text="Clique em uma faixa na tabela",
                 font=("Courier New", 9), bg=PANEL, fg=SUBTEXT
                 ).pack(pady=20)

    # ── Redimensionamento do gráfico ──────────────────────────────────────────
    def _on_chart_resize(self, event):
        if hasattr(self, "_resize_job"):
            self.after_cancel(self._resize_job)
        def _do_resize():
            dpi = self._fig.dpi
            w_in = max(event.width  / dpi, 2)
            h_in = max(event.height / dpi, 1.5)
            self._fig.set_size_inches(w_in, h_in, forward=False)
            self._mpl_canvas.draw_idle()
        self._resize_job = self.after(80, _do_resize)

    # ── Entry helpers ─────────────────────────────────────────────────────────
    def _add_entry(self, val=""):
        e = tk.Entry(self._bp_frame, width=5, font=("Courier New", 11),
                     bg=BORDER, fg=TEXT, insertbackground=TEXT,
                     relief="flat", justify="center")
        e.insert(0, str(val))
        e.pack(side="left", padx=3, ipady=4)
        self._bp_entries.append(e)

    def _remove_entry(self):
        if len(self._bp_entries) > 1:
            self._bp_entries.pop().destroy()

    # ── Refresh ───────────────────────────────────────────────────────────────
    def _refresh(self):
        bps = []
        for e in self._bp_entries:
            try:
                v = float(e.get().strip()) / 100.0
                if 0 < v < 1:
                    bps.append(v)
            except ValueError:
                pass
        self._ranges_data = compute_ranges(self.areas, sorted(set(bps)))
        self._draw_charts()
        self._fill_table()
        # Limpar painel de exemplos e resetar scroll
        self._thumb_refs.clear()
        for w in self._sample_frame.winfo_children():
            w.destroy()
        tk.Label(self._sample_frame,
                 text="Clique em uma faixa na tabela",
                 font=("Courier New", 9), bg=PANEL, fg=SUBTEXT).pack(pady=20)
        self._ex_canvas.yview_moveto(0)

    # ── Gráficos ──────────────────────────────────────────────────────────────
    def _draw_charts(self):
        for ax in (self._ax_bar, self._ax_pie):
            ax.clear()
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        rd     = self._ranges_data
        labels = [r["label"] for r in rd]
        counts = [r["count"] for r in rd]
        colors = [r["color"] for r in rd]
        pcts   = [r["pct"]   for r in rd]

        bars = self._ax_bar.bar(labels, counts, color=colors,
                                edgecolor=BG, linewidth=1)
        for bar, pct in zip(bars, pcts):
            h = bar.get_height()
            self._ax_bar.text(bar.get_x() + bar.get_width() / 2,
                              h + max(counts) * 0.02 if counts else 1,
                              f"{pct:.1f}%", ha="center", va="bottom",
                              color=TEXT, fontsize=8, fontfamily="monospace")
        self._ax_bar.set_title("Quantidade por faixa", color=TEXT,
                               fontsize=9, fontfamily="monospace", pad=6)
        self._ax_bar.tick_params(colors=SUBTEXT, labelsize=7)
        self._ax_bar.set_ylabel("amostras", color=SUBTEXT, fontsize=8)
        self._ax_bar.set_ylim(0, (max(counts) * 1.18) if counts else 1)
        plt.setp(self._ax_bar.get_xticklabels(),
                 rotation=30, ha="right", fontfamily="monospace", fontsize=7)

        self._ax_pie.pie(
            counts if any(c > 0 for c in counts) else [1],
            labels=labels, colors=colors,
            autopct="%1.1f%%", pctdistance=0.75,
            textprops={"color": TEXT, "fontsize": 7, "fontfamily": "monospace"},
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=1.5),
            startangle=90)
        self._ax_pie.set_title("Proporção", color=TEXT,
                               fontsize=9, fontfamily="monospace", pad=6)

        self._mpl_canvas.draw_idle()

    # ── Tabela ────────────────────────────────────────────────────────────────
    def _fill_table(self):
        self._tree.delete(*self._tree.get_children())
        max_pct = max((r["pct"] for r in self._ranges_data), default=0)
        for i, r in enumerate(self._ranges_data):
            if r["pct"] >= max_pct * 0.85:
                status = "🔴 EXCESSO"
            elif r["pct"] <= max_pct * 0.25:
                status = "🟡 ESCASSO"
            else:
                status = "🟢 OK"
            self._tree.insert("", "end", iid=str(i),
                              values=(r["label"], r["count"],
                                      f"{r['pct']:.1f}%", status))

    # ── Seleção → exemplos ────────────────────────────────────────────────────
    def _on_row_select(self, _event):
        sel = self._tree.selection()
        if sel:
            self._show_samples(self._ranges_data[int(sel[0])])

    def _show_samples(self, r):
        self._thumb_refs.clear()
        for w in self._sample_frame.winfo_children():
            w.destroy()

        indices = np.where(r["mask"])[0]
        if len(indices) == 0:
            tk.Label(self._sample_frame, text="Nenhuma amostra nesta faixa.",
                     bg=PANEL, fg=SUBTEXT, font=("Courier New", 10)).pack(pady=20)
            return

        sample_idx = np.random.choice(
            indices, size=min(SAMPLES_PER_RANGE, len(indices)), replace=False)

        tk.Label(self._sample_frame,
                 text=f"Faixa {r['label']}  —  {r['count']} imgs ({r['pct']:.1f}%)",
                 font=("Courier New", 10, "bold"), bg=PANEL,
                 fg=r["color"]).pack(pady=(6, 8))

        for ii, si in enumerate(sample_idx):
            orig_path = self.orig_paths[si]
            bbox      = self.bboxes[si]
            w_n  = bbox[2] - bbox[0]
            h_n  = bbox[3] - bbox[1]
            area = w_n * h_n

            card = tk.Frame(self._sample_frame, bg=BORDER)
            card.pack(fill="x", padx=6, pady=3)

            # Info à esquerda
            info = (f"#{ii+1}  área={area*100:.1f}%\n"
                    f"w={w_n*100:.1f}%  h={h_n*100:.1f}%\n"
                    f"{os.path.basename(orig_path)}")
            tk.Label(card, text=info, font=("Courier New", 8),
                     bg=BORDER, fg=SUBTEXT, justify="left",
                     width=22).pack(side="left", padx=8, pady=4)

            # Thumbnails
            imgs_frame = tk.Frame(card, bg=BORDER)
            imgs_frame.pack(side="right", padx=6, pady=4)

            orig_img = self._thumb(orig_path, bbox=bbox)
            if orig_img:
                self._thumb_refs.append(orig_img)
                col = tk.Frame(imgs_frame, bg=BORDER)
                col.pack(side="left", padx=3)
                tk.Label(col, image=orig_img, bg=BORDER).pack()
                tk.Label(col, text="ORIGINAL", font=("Courier New", 7),
                         bg=BORDER, fg=SUBTEXT).pack()

            crop_path = find_crop_path(orig_path)
            crop_img  = self._thumb(crop_path) if crop_path else None
            if crop_img:
                self._thumb_refs.append(crop_img)
                col2 = tk.Frame(imgs_frame, bg=BORDER)
                col2.pack(side="left", padx=3)
                tk.Label(col2, image=crop_img, bg=BORDER).pack()
                tk.Label(col2, text="CROP", font=("Courier New", 7),
                         bg=BORDER, fg=SUBTEXT).pack()
            elif crop_path is None:
                tk.Label(imgs_frame, text="crop não\nencontrado",
                         font=("Courier New", 7),
                         bg=BORDER, fg=DANGER).pack(side="left", padx=8)

        self._ex_canvas.yview_moveto(0)

    def _thumb(self, path, size=(200, 140), bbox=None):
        if not path or not os.path.exists(path):
            return None
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if bbox is not None:
                h, w = img.shape[:2]
                cv2.rectangle(img,
                              (int(bbox[0]*w), int(bbox[1]*h)),
                              (int(bbox[2]*w), int(bbox[3]*h)),
                              (79, 142, 247), max(2, w // 200))
            pil = Image.fromarray(img)
            pil.thumbnail(size, Image.LANCZOS)
            return ImageTk.PhotoImage(pil)
        except Exception:
            return None


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(CACHE_PATH):
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("Erro", f"Cache não encontrado:\n{CACHE_PATH}")
        r.destroy()
    else:
        App().mainloop()