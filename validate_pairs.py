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
    """A lightweight list widget for navigating image pairs.

    This avoids loading image thumbnails for all items, which can be very slow
    for large folders. It simply displays the filename list and only loads image
    previews on demand when a row is selected.
    """

    def __init__(self, master, on_select):
        super().__init__(master)
        self.on_select = on_select
        self.current_index: Optional[int] = None

        self.listbox = tk.Listbox(self, activestyle="dotbox", exportselection=False)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=self.scroll.set)
        self.listbox.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def populate(self, pairs: List[Tuple[Path, Path]]):
        self.listbox.delete(0, tk.END)
        for orig, _ in pairs:
            self.listbox.insert(tk.END, orig.name)

    def mark_processed(self, index: int):
        """Update the visual representation of an item that was validated."""
        if index is None:
            return
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
        index = sel[0]
        self.current_index = index
        self.on_select(index)

    def select_index(self, index: int):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.see(index)
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
        self._resize_job_orig: Optional[str] = None
        self._resize_job_edit: Optional[str] = None
        self._last_panel_size = {"orig": (0, 0), "edit": (0, 0)}
        self.project_root = Path(__file__).resolve().parent
        self._apply_dark_theme()  # Apply dark theme
        self._build_ui()
        self._bind_keys()
        if self.pairs:
            self.thumb_list.select_index(0)

    def _apply_dark_theme(self):
        """Apply a dark theme to the interface."""
        self.root.configure(bg="#2e2e2e")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#2e2e2e")
        style.configure("TLabel", background="#2e2e2e", foreground="#ffffff")
        style.configure("TButton", background="#444444", foreground="#ffffff")
        style.configure("TScrollbar", background="#444444", troughcolor="#3a3a3a")
        style.map("TButton", background=[("active", "#555555")])

    def _bind_keys(self):
        """Bind keys for navigation and actions."""
        self.root.bind("<Right>", lambda _e: self.validate_current())  # Right arrow for validation
        self.root.bind("<Left>", lambda _e: self.skip_current())       # Left arrow for skipping
        self.root.bind("<Escape>", lambda _e: self.root.destroy())     # Escape to exit

    def _build_ui(self):
        self.root.title(f"Validação - {self.base_dir}")
        self.root.minsize(1120, 660)
        self.root.geometry("1280x760")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.thumb_list = ThumbList(self.root, self._on_select)
        self.thumb_list.grid(row=0, column=0, sticky="nsw", padx=(8, 4), pady=8)
        self.thumb_list.populate(self.pairs)

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1, minsize=self.preview_size[0])
        right.columnconfigure(1, weight=1, minsize=self.preview_size[0])
        right.rowconfigure(1, weight=1, minsize=self.preview_size[1])

        self.lbl_orig = ttk.Label(right, text="Original")
        self.lbl_edit = ttk.Label(right, text="Editada")
        self.lbl_orig.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.lbl_edit.grid(row=0, column=1, sticky="ew", pady=(0, 6))

        self.panel_orig_container = ttk.Frame(right)
        self.panel_edit_container = ttk.Frame(right)
        self.panel_orig_container.grid(row=1, column=0, padx=(0, 6), sticky="nsew")
        self.panel_edit_container.grid(row=1, column=1, sticky="nsew")
        self.panel_orig_container.grid_propagate(False)
        self.panel_edit_container.grid_propagate(False)
        self.panel_orig_container.columnconfigure(0, weight=1)
        self.panel_orig_container.rowconfigure(0, weight=1)
        self.panel_edit_container.columnconfigure(0, weight=1)
        self.panel_edit_container.rowconfigure(0, weight=1)

        self.panel_orig = ttk.Label(self.panel_orig_container, relief="solid", anchor="center")
        self.panel_edit = ttk.Label(self.panel_edit_container, relief="solid", anchor="center")
        self.panel_orig.grid(row=0, column=0, sticky="nsew")
        self.panel_edit.grid(row=0, column=0, sticky="nsew")
        self.panel_orig.bind("<Configure>", lambda e: self._on_panel_resize("orig", e.width, e.height))
        self.panel_edit.bind("<Configure>", lambda e: self._on_panel_resize("edit", e.width, e.height))

        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for i in range(4):
            btn_frame.columnconfigure(i, weight=1)
        self.btn_validate = ttk.Button(btn_frame, text="Validar (→)", command=self.validate_current)
        self.btn_skip = ttk.Button(btn_frame, text="Pular (←)", command=self.skip_current)
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
        if width <= 1 or height <= 1:
            return

        if self._last_panel_size.get(kind) == (width, height):
            return
        self._last_panel_size[kind] = (width, height)

        if kind == "orig":
            if self._resize_job_orig:
                self.root.after_cancel(self._resize_job_orig)
            self._resize_job_orig = self.root.after(
                70,
                lambda: self._render_to_panel(self.img_orig, self.panel_orig, "photo_orig", width, height),
            )
        else:
            if self._resize_job_edit:
                self.root.after_cancel(self._resize_job_edit)
            self._resize_job_edit = self.root.after(
                70,
                lambda: self._render_to_panel(self.img_edit, self.panel_edit, "photo_edit", width, height),
            )

    def _update_status(self):
        total = len(self.pairs)
        done = len(self.processed)
        remaining = total - done

        processed_before = sum(1 for p in self.processed if p <= self.current_index)
        position = (self.current_index + 1) - processed_before

        mark = " (validada)" if self.current_index in self.processed else ""
        self.status_var.set(f"{position}/{remaining}  |  Concluídas: {done}{mark}")

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
        self.thumb_list.mark_processed(self.current_index)
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
        self.thumb_list.select_index(next_idx)

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
        self.thumb_list.select_index(0)

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
