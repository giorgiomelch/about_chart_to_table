#!/usr/bin/env python
"""
heatmap_labeler.py
Manual labeling tool for heatmap images via color bar interpolation.

Usage:
    python heatmap_labeler.py [image_path]

Workflow:
    1. Open a heatmap image
    2. Drag a rectangle over the color bar
    3. Enter Min and Max values, click Confirm
    4. Click on heatmap cells to extract numeric values
    5. Copy as TSV or Save JSON
"""

import sys
import json
import math
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

CANVAS_W = 860
CANVAS_H = 620
CB_N_SAMPLES = 256
CB_PREVIEW_W = 24
CB_PREVIEW_H = 160
DOT_RADIUS = 5
SAMPLE_RADIUS = 1
FONT_COLOR = "#FF2222"
DOT_COLOR = "#FF2222"
BOX_COLOR = "#0080FF"
LABEL_BG = "#FFFFFFCC"


# ─── ColorbarSampler ──────────────────────────────────────────────────────────

class ColorbarSampler:
    """Samples a color gradient from a rectangular region and maps RGB → value."""

    def __init__(self, img_array: np.ndarray, rect: tuple,
                 first_val: float, last_val: float, n_samples: int = CB_N_SAMPLES):
        x1, y1, x2, y2 = (
            min(rect[0], rect[2]), min(rect[1], rect[3]),
            max(rect[0], rect[2]), max(rect[1], rect[3]),
        )
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:
            raise ValueError("Selezione troppo piccola (minimo 2px per lato).")

        self.orientation = "vertical" if h >= w else "horizontal"
        if self.orientation == "vertical":
            cx = (x1 + x2) // 2
            spine = img_array[y1:y2, cx, :3].astype(np.float32)
        else:
            cy = (y1 + y2) // 2
            spine = img_array[cy, x1:x2, :3].astype(np.float32)

        if len(spine) < 4:
            raise ValueError("La selezione è troppo piccola: almeno 4px lungo l'asse principale.")

        idx = np.round(np.linspace(0, len(spine) - 1, n_samples)).astype(int)
        self.samples = spine[idx]        # (N, 3) float32
        self.n = n_samples
        self.first_val = first_val
        self.last_val = last_val
        self.rect = (x1, y1, x2, y2)
        self.is_grayscale = self._is_grayscale(self.samples)

    def color_to_value(self, query_rgb) -> float:
        query = np.array(query_rgb[:3], dtype=np.float32)

        if self.is_grayscale:
            w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
            lum_q = float(np.dot(query, w))
            lum_s = (self.samples * w).sum(axis=1)
            pos = lum_q / 255.0
            if lum_s[0] > lum_s[-1]:
                pos = 1.0 - pos
        else:
            diffs = self.samples - query          # (N, 3)
            dists = np.sqrt((diffs ** 2).sum(axis=1))  # (N,)
            weights = 1.0 / (dists + 1e-6) ** 2
            positions = np.linspace(0.0, 1.0, self.n)
            pos = float(np.dot(weights, positions) / weights.sum())

        pos = float(np.clip(pos, 0.0, 1.0))
        value = self.first_val + pos * (self.last_val - self.first_val)
        lo, hi = min(self.first_val, self.last_val), max(self.first_val, self.last_val)
        return round(float(np.clip(value, lo, hi)), 4)

    def make_preview_image(self, width=CB_PREVIEW_W, height=CB_PREVIEW_H) -> Image.Image:
        img = Image.new("RGB", (width, height))
        pixels = img.load()
        for y in range(height):
            idx = int(round(y / (height - 1) * (self.n - 1)))
            r, g, b = (int(v) for v in self.samples[idx])
            for x in range(width):
                pixels[x, y] = (r, g, b)
        return img

    @staticmethod
    def _is_grayscale(samples: np.ndarray, threshold: float = 20.0) -> bool:
        diff_rg = np.abs(samples[:, 0] - samples[:, 1])
        diff_rb = np.abs(samples[:, 0] - samples[:, 2])
        return float(diff_rg.mean() + diff_rb.mean()) < threshold


# ─── HeatmapLabeler ───────────────────────────────────────────────────────────

class HeatmapLabeler:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Heatmap Labeler")
        self.root.geometry("1240x780")
        self.root.minsize(900, 600)

        # Application state
        self.mode: str = "idle"
        self.image_path: Path | None = None
        self.orig_img: Image.Image | None = None
        self.orig_arr: np.ndarray | None = None
        self.display_img: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.scale: float = 1.0

        # Colorbar state
        self.cb_sampler: ColorbarSampler | None = None
        self.cb_rect_orig: tuple | None = None
        self._drag_start: tuple | None = None
        self._drag_rect_id = None

        # Extracted points: list of (orig_x, orig_y, value)
        self.points: list[tuple[int, int, float]] = []

        self._build_menu()
        self._build_layout()
        self._bind_events()
        self._set_mode("idle")

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)

        file_menu = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Apri immagine…  Ctrl+O", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Salva JSON…  Ctrl+S", command=self.save_json)
        file_menu.add_separator()
        file_menu.add_command(label="Esci", command=self.root.quit)

        edit_menu = tk.Menu(mb, tearoff=False)
        mb.add_cascade(label="Modifica", menu=edit_menu)
        edit_menu.add_command(label="Redefine Colorbar", command=self._start_colorbar_draw)
        edit_menu.add_separator()
        edit_menu.add_command(label="Undo ultimo punto  Ctrl+Z", command=self.undo_last)
        edit_menu.add_command(label="Cancella tutti i punti  Del", command=self.clear_all)

    def _build_layout(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5,
                                    sashrelief=tk.RAISED)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # ── Left: canvas + status ────────────────────────────────────────────
        left = tk.Frame(self.paned, bg="#2b2b2b")
        self.paned.add(left, minsize=400, stretch="always")

        self.canvas = tk.Canvas(left, bg="#1e1e1e", cursor="crosshair",
                                highlightthickness=0)
        h_scroll = tk.Scrollbar(left, orient=tk.HORIZONTAL,
                                command=self.canvas.xview)
        v_scroll = tk.Scrollbar(left, orient=tk.VERTICAL,
                                command=self.canvas.yview)
        self.canvas.config(xscrollcommand=h_scroll.set,
                           yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Apri un'immagine per iniziare.")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              anchor=tk.W, relief=tk.SUNKEN,
                              font=("Monospace", 9), bg="#333", fg="#ccc",
                              padx=6)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # ── Right: sidebar ───────────────────────────────────────────────────
        right = tk.Frame(self.paned, bg="#f0f0f0", padx=8, pady=8)
        self.paned.add(right, minsize=240, stretch="never")

        # Colorbar setup frame
        cb_frame = tk.LabelFrame(right, text="Color Bar", padx=6, pady=6,
                                 font=("TkDefaultFont", 10, "bold"))
        cb_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(cb_frame, text="Trascina un rettangolo\nsulla color bar →",
                 justify=tk.LEFT, font=("TkDefaultFont", 9)).pack(anchor=tk.W)

        grid = tk.Frame(cb_frame)
        grid.pack(fill=tk.X, pady=(6, 0))
        tk.Label(grid, text="Min (top/sx):").grid(row=0, column=0, sticky=tk.W)
        self.min_var = tk.StringVar()
        self.min_entry = tk.Entry(grid, textvariable=self.min_var, width=10)
        self.min_entry.grid(row=0, column=1, padx=(4, 0), pady=2)
        tk.Label(grid, text="Max (bottom/dx):").grid(row=1, column=0, sticky=tk.W)
        self.max_var = tk.StringVar()
        self.max_entry = tk.Entry(grid, textvariable=self.max_var, width=10)
        self.max_entry.grid(row=1, column=1, padx=(4, 0), pady=2)

        self.confirm_btn = tk.Button(cb_frame, text="Confirm Colorbar",
                                     command=self.confirm_colorbar,
                                     state=tk.DISABLED,
                                     font=("TkDefaultFont", 9, "bold"),
                                     bg="#0080FF", fg="white",
                                     activebackground="#005fcc")
        self.confirm_btn.pack(fill=tk.X, pady=(8, 4))

        # Preview strip + labels
        preview_row = tk.Frame(cb_frame)
        preview_row.pack(anchor=tk.W, pady=(4, 0))
        self.preview_canvas = tk.Canvas(preview_row, width=CB_PREVIEW_W,
                                        height=CB_PREVIEW_H,
                                        bg="#cccccc", highlightthickness=1,
                                        highlightbackground="#999")
        self.preview_canvas.pack(side=tk.LEFT)
        lbl_col = tk.Frame(preview_row)
        lbl_col.pack(side=tk.LEFT, padx=(6, 0))
        self.max_label = tk.Label(lbl_col, text="–", font=("Monospace", 9))
        self.max_label.pack(anchor=tk.W)
        tk.Label(lbl_col, text="").pack(expand=True, fill=tk.Y)
        self.min_label = tk.Label(lbl_col, text="–", font=("Monospace", 9))
        self.min_label.pack(anchor=tk.W, side=tk.BOTTOM)

        # Values frame
        vals_frame = tk.LabelFrame(right, text="Valori estratti", padx=6, pady=6,
                                   font=("TkDefaultFont", 10, "bold"))
        vals_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        text_container = tk.Frame(vals_frame)
        text_container.pack(fill=tk.BOTH, expand=True)
        self.vals_text = tk.Text(text_container, font=("Monospace", 9),
                                 width=28, state=tk.DISABLED,
                                 wrap=tk.NONE, relief=tk.FLAT,
                                 bg="#fafafa")
        vs = tk.Scrollbar(text_container, command=self.vals_text.yview)
        self.vals_text.config(yscrollcommand=vs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.vals_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_row = tk.Frame(vals_frame)
        btn_row.pack(fill=tk.X, pady=(4, 0))
        tk.Button(btn_row, text="Copy TSV", command=self.copy_as_tsv,
                  width=10).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row, text="Save JSON", command=self.save_json,
                  width=10).pack(side=tk.LEFT)

        btn_row2 = tk.Frame(vals_frame)
        btn_row2.pack(fill=tk.X, pady=(4, 0))
        tk.Button(btn_row2, text="Undo", command=self.undo_last,
                  width=10).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_row2, text="Clear All", command=self.clear_all,
                  width=10).pack(side=tk.LEFT)

        # Set initial pane widths after layout
        self.root.update_idletasks()
        self.paned.sash_place(0, 960, 0)

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_json())
        self.root.bind("<Control-z>", lambda e: self.undo_last())
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Delete>", lambda e: self.clear_all())

    # ── Mode management ───────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self.mode = mode
        msgs = {
            "idle": "Apri un'immagine per iniziare  (File > Apri immagine).",
            "draw_colorbar": "Trascina un rettangolo sulla color bar, poi inserisci Min/Max e clicca Confirm.",
            "click_cells": "Clicca sulle celle della heatmap per estrarne il valore.",
        }
        self._update_status(msgs.get(mode, ""))
        if mode == "draw_colorbar":
            self.canvas.config(cursor="crosshair")
        elif mode == "click_cells":
            self.canvas.config(cursor="tcross")

    # ── File operations ───────────────────────────────────────────────────────

    def open_image(self, path: Path | None = None):
        if path is None:
            p = filedialog.askopenfilename(
                title="Apri immagine heatmap",
                filetypes=[("Immagini", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
                           ("Tutti i file", "*.*")]
            )
            if not p:
                return
            path = Path(p)
        self._load_and_display(path)

    def _load_and_display(self, path: Path):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Errore", f"Impossibile aprire l'immagine:\n{exc}")
            return
        self.image_path = path
        self.orig_img = img
        self.orig_arr = np.array(img, dtype=np.uint8)
        self.cb_sampler = None
        self.cb_rect_orig = None
        self.points = []
        self._fit_image_to_canvas()
        self._refresh_values_list()
        self._set_mode("draw_colorbar")
        self.root.title(f"Heatmap Labeler — {path.name}")

    def _fit_image_to_canvas(self):
        self.root.update_idletasks()
        cw = self.canvas.winfo_width() or CANVAS_W
        ch = self.canvas.winfo_height() or CANVAS_H
        ow, oh = self.orig_img.size
        self.scale = min(cw / ow, ch / oh, 1.0)
        dw = max(1, int(ow * self.scale))
        dh = max(1, int(oh * self.scale))
        self.display_img = self.orig_img.resize((dw, dh), Image.LANCZOS)
        self.canvas.config(scrollregion=(0, 0, dw, dh))
        self._redraw_canvas()

    # ── Canvas event handlers ─────────────────────────────────────────────────

    def _on_canvas_press(self, event):
        cx, cy = self._scroll_coords(event)
        if self.mode == "draw_colorbar":
            self._drag_start = (cx, cy)
            if self._drag_rect_id:
                self.canvas.delete(self._drag_rect_id)
                self._drag_rect_id = None
        elif self.mode == "click_cells":
            self._on_cell_click(cx, cy)

    def _on_canvas_drag(self, event):
        if self.mode != "draw_colorbar" or self._drag_start is None:
            return
        cx, cy = self._scroll_coords(event)
        self._update_rubber_band(cx, cy)

    def _on_canvas_release(self, event):
        if self.mode != "draw_colorbar" or self._drag_start is None:
            return
        cx, cy = self._scroll_coords(event)
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
        if abs(cx - x0) < 3 and abs(cy - y0) < 3:
            return
        ox1, oy1 = self._canvas_to_orig(x0, y0)
        ox2, oy2 = self._canvas_to_orig(cx, cy)
        self.cb_rect_orig = (ox1, oy1, ox2, oy2)
        self.confirm_btn.config(state=tk.NORMAL)
        self._redraw_canvas()
        self._update_status("Rettangolo selezionato. Inserisci Min/Max e clicca Confirm Colorbar.")

    def _on_mouse_move(self, event):
        if self.orig_img is None:
            return
        cx, cy = self._scroll_coords(event)
        ox, oy = self._canvas_to_orig(cx, cy)
        if 0 <= ox < self.orig_img.width and 0 <= oy < self.orig_img.height:
            r, g, b = self.orig_arr[oy, ox]
            msg = f"Pixel: ({ox}, {oy})  RGB: ({r}, {g}, {b})"
            if self.cb_sampler and self.mode == "click_cells":
                val = self.cb_sampler.color_to_value((r, g, b))
                msg += f"  →  {val:.4f}"
            self._update_status(msg)

    def _on_escape(self, event):
        if self.mode == "draw_colorbar" and self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None
            self._drag_start = None

    # ── Rubber-band rectangle ─────────────────────────────────────────────────

    def _update_rubber_band(self, cx: int, cy: int):
        if self._drag_start is None:
            return
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
        x0, y0 = self._drag_start
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, cx, cy,
            outline=BOX_COLOR, width=2, dash=(5, 4)
        )

    # ── Colorbar workflow ─────────────────────────────────────────────────────

    def _start_colorbar_draw(self):
        if self.orig_img is None:
            return
        self.cb_sampler = None
        self.cb_rect_orig = None
        self.points = []
        self.confirm_btn.config(state=tk.DISABLED)
        self._refresh_values_list()
        self._redraw_canvas()
        self._update_colorbar_preview(None)
        self._set_mode("draw_colorbar")

    def confirm_colorbar(self):
        if self.cb_rect_orig is None:
            messagebox.showwarning("Attenzione", "Prima seleziona la color bar trascinando un rettangolo.")
            return
        try:
            first_val = float(self.min_var.get())
            last_val = float(self.max_var.get())
        except ValueError:
            messagebox.showerror("Errore", "Min e Max devono essere numeri validi.")
            return
        if first_val == last_val:
            messagebox.showwarning("Attenzione", "Min e Max sono uguali: tutti i valori saranno identici.")
        try:
            sampler = ColorbarSampler(self.orig_arr, self.cb_rect_orig,
                                      first_val, last_val)
        except ValueError as exc:
            messagebox.showerror("Errore selezione", str(exc))
            return
        self.cb_sampler = sampler
        self._update_colorbar_preview(sampler)
        self._set_mode("click_cells")

    def _update_colorbar_preview(self, sampler: ColorbarSampler | None):
        self.preview_canvas.delete("all")
        if sampler is None:
            self.max_label.config(text="–")
            self.min_label.config(text="–")
            return
        prev_img = sampler.make_preview_image(CB_PREVIEW_W, CB_PREVIEW_H)
        self._preview_photo = ImageTk.PhotoImage(prev_img)
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self._preview_photo)
        self.max_label.config(text=str(sampler.first_val))
        self.min_label.config(text=str(sampler.last_val))

    # ── Cell click & value extraction ────────────────────────────────────────

    def _on_cell_click(self, canvas_x: int, canvas_y: int):
        if self.cb_sampler is None:
            return
        ox, oy = self._canvas_to_orig(canvas_x, canvas_y)
        h, w = self.orig_arr.shape[:2]
        x1 = max(0, ox - SAMPLE_RADIUS)
        x2 = min(w - 1, ox + SAMPLE_RADIUS)
        y1 = max(0, oy - SAMPLE_RADIUS)
        y2 = min(h - 1, oy + SAMPLE_RADIUS)
        patch = self.orig_arr[y1:y2 + 1, x1:x2 + 1, :3].reshape(-1, 3)
        mean_rgb = patch.mean(axis=0)
        value = self.cb_sampler.color_to_value(mean_rgb)
        self.points.append((ox, oy, value))
        self._refresh_values_list()
        self._redraw_canvas()
        r, g, b = (int(v) for v in mean_rgb)
        self._update_status(f"Pixel: ({ox}, {oy})  RGB: ({r}, {g}, {b})  →  {value:.4f}")

    # ── Canvas rendering ──────────────────────────────────────────────────────

    def _redraw_canvas(self):
        if self.display_img is None:
            return
        annotated = self.display_img.copy()
        self._draw_annotations(annotated)
        self.photo = ImageTk.PhotoImage(annotated)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def _draw_annotations(self, img: Image.Image):
        draw = ImageDraw.Draw(img)

        # Color bar rectangle
        if self.cb_rect_orig:
            cx1, cy1 = self._orig_to_canvas(*self.cb_rect_orig[:2])
            cx2, cy2 = self._orig_to_canvas(*self.cb_rect_orig[2:])
            _draw_dashed_rect(draw, cx1, cy1, cx2, cy2, BOX_COLOR, dash=6)

        # Extracted points
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        except Exception:
            font = ImageFont.load_default()

        for ox, oy, val in self.points:
            dx, dy = self._orig_to_canvas(ox, oy)
            r = DOT_RADIUS
            draw.ellipse((dx - r, dy - r, dx + r, dy + r),
                         fill=DOT_COLOR, outline="white", width=1)
            label = f"{val:.3f}"
            lx, ly = dx + r + 3, dy - 7
            bbox = draw.textbbox((lx, ly), label, font=font)
            pad = 2
            draw.rectangle((bbox[0] - pad, bbox[1] - pad,
                             bbox[2] + pad, bbox[3] + pad),
                            fill=(255, 255, 255, 180))
            draw.text((lx, ly), label, fill=FONT_COLOR, font=font)

    # ── Coordinate utilities ──────────────────────────────────────────────────

    def _scroll_coords(self, event) -> tuple[int, int]:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return int(x), int(y)

    def _canvas_to_orig(self, cx: int, cy: int) -> tuple[int, int]:
        if self.scale == 0 or self.orig_img is None:
            return 0, 0
        ox = int(cx / self.scale)
        oy = int(cy / self.scale)
        ox = max(0, min(ox, self.orig_img.width - 1))
        oy = max(0, min(oy, self.orig_img.height - 1))
        return ox, oy

    def _orig_to_canvas(self, ox: int, oy: int) -> tuple[int, int]:
        return int(ox * self.scale), int(oy * self.scale)

    # ── Values panel ──────────────────────────────────────────────────────────

    def _refresh_values_list(self):
        self.vals_text.config(state=tk.NORMAL)
        self.vals_text.delete("1.0", tk.END)
        header = f"{'#':>3}  {'x':>6}  {'y':>6}  {'value':>10}\n"
        self.vals_text.insert(tk.END, header)
        self.vals_text.insert(tk.END, "─" * 30 + "\n")
        for i, (ox, oy, val) in enumerate(self.points, 1):
            line = f"{i:>3}  {ox:>6}  {oy:>6}  {val:>10.4f}\n"
            self.vals_text.insert(tk.END, line)
        self.vals_text.config(state=tk.DISABLED)
        self.vals_text.see(tk.END)

    def copy_as_tsv(self):
        if not self.points:
            messagebox.showinfo("Nessun dato", "Nessun punto estratto.")
            return
        lines = ["x\ty\tvalue"]
        for ox, oy, val in self.points:
            lines.append(f"{ox}\t{oy}\t{val:.4f}")
        text = "\n".join(lines)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._update_status(f"{len(self.points)} valori copiati negli appunti (TSV).")

    def undo_last(self):
        if self.points:
            self.points.pop()
            self._refresh_values_list()
            self._redraw_canvas()

    def clear_all(self):
        if not self.points:
            return
        if messagebox.askyesno("Conferma", "Cancellare tutti i punti estratti?"):
            self.points.clear()
            self._refresh_values_list()
            self._redraw_canvas()

    # ── Save JSON ─────────────────────────────────────────────────────────────

    def save_json(self):
        if self.cb_sampler is None:
            messagebox.showwarning("Attenzione", "Nessuna color bar confermata.")
            return
        if not self.points:
            if not messagebox.askyesno("Nessun punto", "Nessun punto estratto. Salvare file vuoto?"):
                return

        default_name = ""
        if self.image_path:
            default_name = self.image_path.stem + "_labeled.json"

        path = filedialog.asksaveasfilename(
            title="Salva JSON",
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Tutti i file", "*.*")]
        )
        if not path:
            return

        data = {
            "chart_title": None,
            "x_axis_label": None,
            "y_axis_label": None,
            "x_axis": {"min": None, "max": None, "is_log": False},
            "y_axis": {"min": None, "max": None, "is_log": False},
            "cell_axis": {
                "min": self.cb_sampler.first_val,
                "max": self.cb_sampler.last_val,
                "is_log": False,
            },
            "data_points": [
                {"x_value": ox, "y_value": oy, "cell_value": val}
                for ox, oy, val in self.points
            ],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._update_status(f"Salvato: {path}")
        except Exception as exc:
            messagebox.showerror("Errore salvataggio", str(exc))

    # ── Status bar ────────────────────────────────────────────────────────────

    def _update_status(self, msg: str):
        self.status_var.set(msg)


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_dashed_rect(draw: ImageDraw.Draw, x1: int, y1: int,
                      x2: int, y2: int, color: str, dash: int = 6):
    sides = [(x1, y1, x2, y1), (x2, y1, x2, y2),
             (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for ax, ay, bx, by in sides:
        length = math.hypot(bx - ax, by - ay)
        if length < 1:
            continue
        n = max(1, int(length / (2 * dash)))
        for i in range(n):
            t0 = (2 * i * dash) / length
            t1 = min((2 * i + 1) * dash / length, 1.0)
            draw.line([
                (ax + t0 * (bx - ax), ay + t0 * (by - ay)),
                (ax + t1 * (bx - ax), ay + t1 * (by - ay)),
            ], fill=color, width=2)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = HeatmapLabeler(root)
    if len(sys.argv) > 1:
        app.open_image(Path(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
