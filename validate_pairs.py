import os
from pathlib import Path
from typing import List, Tuple, Optional, Set
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _find_edited(base_dir: Path, stem: str) -> Optional[Path]:
    edited_dir = base_dir / "out"
    if not edited_dir.exists():
        return None
    for ext in IMAGE_EXTENSIONS:
        p = edited_dir / f"{stem}_editado{ext}"
        if p.exists():
            return p
    for cand in edited_dir.glob(f"{stem}_editado.*"):
        if cand.suffix.lower() in IMAGE_EXTENSIONS:
            return cand
    return None

def build_pairs(base_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for f in sorted(base_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            edited = _find_edited(base_dir, f.stem)
            if edited:
                pairs.append((f, edited))
    return pairs

class ThumbList(ttk.Frame):
    def __init__(self, master, on_select, thumb_size=(96, 96)):
        super().__init__(master)
        self.on_select = on_select
        self.thumb_size = thumb_size
        self.canvas = tk.Canvas(self, width=150, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.grid(row=0, column=0, sticky="ns")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.photos: List[ImageTk.PhotoImage] = []
        self.buttons: List[tk.Button] = []
        self.current_index: Optional[int] = None

    def populate(self, pairs: List[Tuple[Path, Path]]):
        for b in self.buttons:
            b.destroy()
        self.photos.clear()
        self.buttons.clear()
        for idx, (orig, _) in enumerate(pairs):
            try:
                img = Image.open(orig).convert("RGB")
                img.thumbnail(self.thumb_size, Image.LANCZOS)
            except OSError:
                img = Image.new("RGB", self.thumb_size, "#555")
            ph = ImageTk.PhotoImage(img)
            self.photos.append(ph)
            btn = tk.Button(self.inner, image=ph, text=orig.name, compound="top",
                            width=self.thumb_size[0], padx=2, pady=2,
                            command=lambda i=idx: self._select(i))
            btn.grid(row=idx, column=0, sticky="ew", pady=2)
            self.buttons.append(btn)

    def _select(self, index: int):
        self.current_index = index
        self.on_select(index)

class ValidatorApp:
    def __init__(self, root: tk.Tk, base_dir: Path, pairs: List[Tuple[Path, Path]]):
        self.root = root
        self.base_dir = base_dir
        self.pairs = pairs
        self.processed: Set[int] = set()
        self.current_index = 0
        self.preview_size = (520, 520)
        self.photo_orig: Optional[ImageTk.PhotoImage] = None
        self.photo_edit: Optional[ImageTk.PhotoImage] = None
        self.img_orig: Optional[Image.Image] = None
        self.img_edit: Optional[Image.Image] = None
        self.project_root = Path(__file__).resolve().parent
        self._build_ui()
        self._bind_keys()
        if self.pairs:
            self.thumb_list._select(0)

    def _build_ui(self):
        self.root.title(f"Validação - {self.base_dir}")
        self.root.minsize(1120, 660)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.thumb_list = ThumbList(self.root, self._on_select)
        self.thumb_list.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)
        self.thumb_list.populate(self.pairs)

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)

        self.lbl_orig = ttk.Label(right, text="Original")
        self.lbl_edit = ttk.Label(right, text="Editada")
        self.lbl_orig.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.lbl_edit.grid(row=0, column=1, sticky="ew", pady=(0, 6))

        self.panel_orig = ttk.Label(right, relief="solid")
        self.panel_edit = ttk.Label(right, relief="solid")
        self.panel_orig.grid(row=1, column=0, padx=(0, 6), sticky="nsew")
        self.panel_edit.grid(row=1, column=1, sticky="nsew")
        self.panel_orig.bind("<Configure>", lambda e: self._on_panel_resize("orig", e.width, e.height))
        self.panel_edit.bind("<Configure>", lambda e: self._on_panel_resize("edit", e.width, e.height))

        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)
        self.btn_validate = ttk.Button(btn_frame, text="Validar (Enter)", command=self.validate_current)
        self.btn_skip = ttk.Button(btn_frame, text="Pular (Espaço)", command=self.skip_current)
        self.btn_exit = ttk.Button(btn_frame, text="Sair (Esc)", command=self.root.destroy)
        self.btn_open = ttk.Button(btn_frame, text="Abrir Pasta", command=self._choose_new_directory)
        self.btn_validate.grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.btn_skip.grid(row=0, column=1, padx=(0, 6), sticky="ew")
        self.btn_exit.grid(row=0, column=2, padx=(0, 6), sticky="ew")
        self.btn_open.grid(row=0, column=3, sticky="ew")

        self.status_var = tk.StringVar()
        ttk.Label(right, textvariable=self.status_var, anchor="center").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

    def _bind_keys(self):
        self.root.bind("<Return>", lambda _e: self.validate_current())
        self.root.bind("<space>", lambda _e: self.skip_current())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _on_select(self, index: int):
        self.current_index = index
        orig, edited = self.pairs[index]
        self.lbl_orig.config(text=f"Original - {orig.name}")
        self.lbl_edit.config(text=f"Editada - {edited.name}")
        self.img_orig = self._load_preview(orig)
        self.img_edit = self._load_preview(edited)
        self.panel_orig.update_idletasks()
        self.panel_edit.update_idletasks()
        self._render_to_panel(self.img_orig, self.panel_orig, "photo_orig")
        self._render_to_panel(self.img_edit, self.panel_edit, "photo_edit")
        self._update_status()

    def _load_preview(self, path: Path) -> Image.Image:
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except OSError:
            return Image.new("RGB", self.preview_size, "#444")

    def _render_to_panel(self, pil_img: Optional[Image.Image], panel: ttk.Label, attr: str,
                         width: Optional[int] = None, height: Optional[int] = None):
        if pil_img is None:
            panel.configure(image="")
            setattr(self, attr, None)
            return
        w = width if width and width > 1 else panel.winfo_width()
        h = height if height and height > 1 else panel.winfo_height()
        if w <= 1 or h <= 1:
            w, h = self.preview_size
        resized = pil_img.copy()
        resized.thumbnail((w, h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        panel.configure(image=photo)
        setattr(self, attr, photo)

    def _on_panel_resize(self, kind: str, width: int, height: int):
        if kind == "orig":
            self._render_to_panel(self.img_orig, self.panel_orig, "photo_orig", width, height)
        else:
            self._render_to_panel(self.img_edit, self.panel_edit, "photo_edit", width, height)

    def _update_status(self):
        total = len(self.pairs)
        done = len(self.processed)
        idx = self.current_index + 1
        mark = " (validada)" if self.current_index in self.processed else ""
        self.status_var.set(f"{idx}/{total}  |  Concluídas: {done}{mark}")

    def validate_current(self):
        if self.current_index in self.processed:
            self._advance()
            return
        orig, edited = self.pairs[self.current_index]
        try:
            self._move_pair(orig, edited)
        except OSError as e:
            messagebox.showerror("Erro ao mover", str(e), parent=self.root)
            return
        self.processed.add(self.current_index)
        self._update_status()
        self._advance()

    def skip_current(self):
        # NÃO salva, NÃO marca processado. Apenas avança.
        self._advance()

    def _move_pair(self, orig: Path, edited: Path):
        dataset = self.project_root / "dataset"
        origin_dir = dataset / "origin"
        cropped_dir = dataset / "cropped"
        origin_dir.mkdir(parents=True, exist_ok=True)
        cropped_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(orig, origin_dir / orig.name)
        shutil.move(edited, cropped_dir / edited.name)

    def _advance(self):
        total = len(self.pairs)
        if total == 0:
            return
        # Se todas validadas, encerra
        if len(self.processed) == total:
            messagebox.showinfo("Concluído", "Todas as imagens foram validadas.")
            self.root.destroy()
            return
        # Procurar próxima não validada
        next_idx = (self.current_index + 1) % total
        attempts = 0
        while attempts < total and next_idx in self.processed:
            next_idx = (next_idx + 1) % total
            attempts += 1
        # Se todas marcadas após loop
        if attempts >= total and next_idx in self.processed:
            messagebox.showinfo("Concluído", "Todas as imagens foram validadas.")
            self.root.destroy()
            return
        self.thumb_list._select(next_idx)

    def _choose_new_directory(self):
        chosen = filedialog.askdirectory(parent=self.root, title="Selecione a pasta com imagens originais")
        if not chosen:
            return
        new_base = Path(chosen)
        new_pairs = build_pairs(new_base)
        if not new_pairs:
            messagebox.showinfo("Sem pares", "Nenhum par (original + *_editado) encontrado.", parent=self.root)
            return
        self.base_dir = new_base
        self.pairs = new_pairs
        self.processed.clear()
        self.current_index = 0
        self.img_orig = None
        self.img_edit = None
        self.photo_orig = None
        self.photo_edit = None
        self.root.title(f"Validação - {self.base_dir}")
        self.thumb_list.populate(self.pairs)
        self.thumb_list._select(0)

def select_base_dir() -> Optional[Path]:
    root = tk.Tk()
    root.withdraw()
    chosen = filedialog.askdirectory(title="Selecione a pasta com imagens originais")
    if not chosen:
        root.destroy()
        return None
    root.destroy()
    return Path(chosen)

def main():
    base_dir = select_base_dir()
    if not base_dir:
        return
    pairs = build_pairs(base_dir)
    if not pairs:
        messagebox.showinfo("Sem pares", "Nenhum par (original + *_editado) encontrado.")
        return
    root = tk.Tk()
    app = ValidatorApp(root, base_dir, pairs)
    root.mainloop()

if __name__ == "__main__":
    main()
