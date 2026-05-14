"""
app.py — Interfaz gráfica profesional NeuroAI v2
Sistema de Análisis de Neuroimagen por IA

Ejecutar :  python app.py
Requiere :  pip install pillow matplotlib nibabel torch fpdf2
"""

import os, sys, threading, datetime
from io import BytesIO
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
try:
    _LANCZOS = Image.Resampling.LANCZOS  # Pillow 10+
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]  # Pillow 9
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analisis.predictor  import predecir_tumor
from analisis.analizador import obtener_diagnostico_clinico
from reportes.reporte    import generar_reporte_final

# ── Paleta ────────────────────────────────────────────────────────────────────
C = {
    "bg"     : "#080b11",
    "panel"  : "#0f1320",
    "panel2" : "#141928",
    "border" : "#1c2438",
    "border2": "#2a3654",
    "a1"     : "#4cc9f0",
    "a2"     : "#f72585",
    "a3"     : "#4361ee",
    "a5"     : "#3a0ca3",
    "ok"     : "#06d6a0",
    "warn"   : "#ffd166",
    "danger" : "#ef233c",
    "text"   : "#dde3f0",
    "muted"  : "#4a5568",
    "muted2" : "#718096",
}

RIESGO_C = {
    "BAJO"               : C["ok"],
    "MODERADO"           : C["warn"],
    "ALTO (CRITICO)"     : C["danger"],
    "SIN TUMOR DETECTADO": C["muted2"],
}

FM = "Consolas"   # fuente mono


# ─────────────────────────────────────────────────────────────────────────────
#  Splash
# ─────────────────────────────────────────────────────────────────────────────

class SplashScreen(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.overrideredirect(True)
        w, h = 480, 240
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.configure(bg=C["panel"])
        self.attributes("-alpha", 0.0)

        tk.Frame(self, bg=C["a3"], height=3).pack(fill="x", side="top")
        tk.Frame(self, bg=C["a3"], height=3).pack(fill="x", side="bottom")

        inner = tk.Frame(self, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=2)

        logo = tk.Frame(inner, bg=C["panel"])
        logo.pack(pady=(28, 4))
        tk.Label(logo, text="⬡ NEURO", font=(FM, 32, "bold"),
                 fg=C["a1"], bg=C["panel"]).pack(side="left")
        tk.Label(logo, text="AI", font=(FM, 32, "bold"),
                 fg=C["a2"], bg=C["panel"]).pack(side="left")

        tk.Label(inner, text="Sistema de Análisis de Neuroimagen",
                 font=(FM, 10), fg=C["muted2"], bg=C["panel"]).pack()

        style = ttk.Style(); style.theme_use("clam")
        style.configure("S.Horizontal.TProgressbar",
                         troughcolor=C["border"], background=C["a1"],
                         bordercolor=C["panel"])
        self._bar = ttk.Progressbar(inner, style="S.Horizontal.TProgressbar",
                                     mode="indeterminate", length=340)
        self._bar.pack(pady=16)
        self._bar.start(8)

        self._lbl = tk.Label(inner, text="Inicializando…",
                              font=(FM, 8), fg=C["muted"], bg=C["panel"])
        self._lbl.pack()
        self._fade()

    def _fade(self, a=0.0):
        a = min(a + 0.09, 1.0)
        self.attributes("-alpha", a)
        if a < 1.0:
            self.after(18, self._fade, a)

    def msg(self, t): self._lbl.config(text=t)
    def close(self):  self._bar.stop(); self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Figura matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def render_figure(original, mascara, datos) -> Image.Image:
    fig = plt.figure(figsize=(15, 5.2), facecolor=C["bg"])
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06,
                            left=0.01, right=0.99, top=0.88, bottom=0.04)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # Panel 1 — MRI
    ax0.imshow(original, cmap="gray", aspect="equal")
    ax0.set_facecolor(C["bg"])
    ax0.set_title("MRI  ·  T1CE  AXIAL", color=C["a1"],
                  fontsize=10, fontweight="bold", pad=10, fontfamily=FM)
    for s in ax0.spines.values(): s.set_edgecolor(C["border2"])
    ax0.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Panel 2 — Segmentación
    mask_m = np.ma.masked_where(mascara < 0.5, mascara)
    ax1.imshow(original, cmap="gray", aspect="equal")
    ax1.imshow(mask_m, cmap="cool", alpha=0.78, interpolation="none", aspect="equal")
    ax1.set_facecolor(C["bg"])
    ax1.set_title("SEGMENTACIÓN  U-NET", color=C["a2"],
                  fontsize=10, fontweight="bold", pad=10, fontfamily=FM)
    for s in ax1.spines.values(): s.set_edgecolor(C["border2"])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if mascara.max() > 0.5:
        try: ax1.contour(mascara, levels=[0.5], colors=[C["a2"]], linewidths=[1.5])
        except Exception: pass

    # Panel 3 — Métricas
    ax2.set_facecolor(C["panel"])
    for s in ax2.spines.values(): s.set_edgecolor(C["border2"])
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax2.set_title("MÉTRICAS  CLÍNICAS", color=C["a3"],
                  fontsize=10, fontweight="bold", pad=10, fontfamily=FM)

    tumor_ok = datos.get("tumor_presente", False)
    riesgo   = datos.get("riesgo", "—")
    rc       = RIESGO_C.get(riesgo, C["text"])

    ax2.text(0.5, 0.96, "─"*34, color=C["border2"], fontsize=7,
             ha="center", va="top", transform=ax2.transAxes, fontfamily=FM)

    if tumor_ok:
        rows = [
            ("LOCALIZACIÓN", datos.get("ubicacion", "—"), C["text"]),
            ("DIÁMETRO",     datos.get("diametro",   "—"), C["a1"]),
            ("VOLUMEN",      datos.get("volumen",    "—"), C["a1"]),
            ("CONFIANZA IA", datos.get("confianza",  "—"), C["ok"]),
            ("SLICES",       str(datos.get("slices_con_tumor", "—")), C["muted2"]),
        ]
        y = 0.84
        for k, v, col in rows:
            ax2.text(0.06, y, f"{k:<14}", color=C["muted2"], fontsize=9,
                     va="top", transform=ax2.transAxes, fontfamily=FM)
            ax2.text(0.56, y, v, color=col, fontsize=9, fontweight="bold",
                     va="top", transform=ax2.transAxes, fontfamily=FM)
            y -= 0.12
        ax2.add_patch(mpatches.Rectangle((0.06, 0.08), 0.88, 0.14,
                      transform=ax2.transAxes, color=rc, zorder=2))
        ax2.text(0.50, 0.155, f"NIVEL DE RIESGO:  {riesgo}",
                 color=C["bg"], fontsize=9, fontweight="bold", ha="center",
                 va="center", transform=ax2.transAxes, fontfamily=FM, zorder=3)
    else:
        ax2.text(0.5, 0.55, "SIN TUMOR\nDETECTADO",
                 color=C["ok"], fontsize=16, fontweight="bold",
                 ha="center", va="center", transform=ax2.transAxes, fontfamily=FM)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

def lbl(parent, text, size=9, color=None, bold=False, **kw):
    return tk.Label(parent, text=text,
                    font=(FM, size, "bold" if bold else "normal"),
                    fg=color or C["text"], bg=parent["bg"], **kw)

def sep(parent, pad=0):
    tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", padx=pad, pady=2)

def btn(parent, text, cmd, bg=None, fg=None, abg=None, state="normal", **kw):
    b = tk.Button(parent, text=text, command=cmd,
                  font=(FM, 10, "bold"),
                  fg=fg or C["text"], bg=bg or C["panel"],
                  activeforeground=fg or C["text"],
                  activebackground=abg or C["border2"],
                  relief="flat", bd=0, cursor="hand2",
                  padx=12, pady=8, **kw)
    # Fix: pasar state con config() evita el error de Pylance con Literal
    if state != "normal":
        b.config(state=state)  # type: ignore
    if abg:
        b.bind("<Enter>", lambda e: b.config(bg=abg))
        b.bind("<Leave>", lambda e: b.config(bg=bg or C["panel"]))
    return b


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────

class NeuroimagenApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("NeuroAI  ·  Sistema de Análisis de Neuroimagen")
        self.configure(bg=C["bg"])
        self.geometry("1380x860")
        self.minsize(1100, 680)

        self._ruta       = tk.StringVar()
        self._datos      = {}
        self._img_panel  = None
        self._tk_img     = None
        self._img_temp   = None
        self._historial  = []
        self._zoom       = 0.0    # 0 = ajustar al canvas

        self._build()
        self._splash()

    # ── Splash ────────────────────────────────────────────────────────────────

    def _splash(self):
        s = SplashScreen(self)
        steps = [
            ("Cargando módulos de análisis…",  350),
            ("Inicializando predictor U-Net…", 400),
            ("Preparando interfaz gráfica…",   300),
        ]
        def go(i=0):
            if i < len(steps):
                s.msg(steps[i][0]); self.after(steps[i][1], go, i+1)
            else:
                s.close(); self.deiconify()
                self._log("Sistema listo — selecciona un .nii para comenzar", "info")
        self.after(200, go)

    # ── Build ─────────────────────────────────────────────────────────────────

    def _build(self):
        self._topbar()
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        self._side = tk.Frame(body, bg=C["bg"], width=298)
        self._side.pack(side="left", fill="y"); self._side.pack_propagate(False)
        self._sidebar()

        tk.Frame(body, bg=C["border"], width=1).pack(side="left", fill="y")

        self._main = tk.Frame(body, bg=C["bg"])
        self._main.pack(side="left", fill="both", expand=True)
        self._tabs()

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        self._statusbar()

    # ── Topbar ────────────────────────────────────────────────────────────────

    def _topbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=54)
        bar.pack(fill="x"); bar.pack_propagate(False)

        logo = tk.Frame(bar, bg=C["panel"])
        logo.pack(side="left", padx=18)
        tk.Label(logo, text="⬡ NEURO", font=(FM, 15, "bold"),
                 fg=C["a1"], bg=C["panel"]).pack(side="left")
        tk.Label(logo, text="AI", font=(FM, 15, "bold"),
                 fg=C["a2"], bg=C["panel"]).pack(side="left")

        tk.Label(bar, text="  /  Sistema de Análisis de Neuroimagen por IA",
                 font=(FM, 9), fg=C["muted"], bg=C["panel"]).pack(side="left")

        tk.Label(bar, text=f"v2.0  ·  {datetime.datetime.now():%d %b %Y}  ",
                 font=(FM, 9), fg=C["muted"], bg=C["panel"]).pack(side="right")

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self):
        S = self._side

        # Archivo
        self._stitle(S, "ARCHIVO")
        fc = tk.Frame(S, bg=C["panel"],
                      highlightbackground=C["border"], highlightthickness=1)
        fc.pack(fill="x", padx=12, pady=4)
        self._lbl_arc = tk.Label(fc, text="Sin archivo seleccionado",
                                  font=(FM, 8), fg=C["muted"],
                                  bg=C["panel"], wraplength=252, justify="left")
        self._lbl_arc.pack(fill="x", padx=10, pady=8)

        btn(S, "  📂  Examinar archivo .nii",
            self._pick_file,
            bg=C["a3"], fg="#fff", abg=C["a5"]
            ).pack(fill="x", padx=12, pady=(0, 2))

        # Análisis
        self._stitle(S, "ANÁLISIS")
        self._btn_go = btn(S, "  🧠  Analizar imagen",
                            self._run, bg=C["a1"], fg=C["bg"],
                            abg="#38b2d8", state="disabled")
        self._btn_go.pack(fill="x", padx=12, pady=2)

        self._btn_pdf = btn(S, "  📄  Exportar reporte PDF",
                             self._export_pdf,
                             abg=C["border2"], state="disabled")
        self._btn_pdf.pack(fill="x", padx=12, pady=2)

        btn(S, "  🖼  Abrir imagen externa",
            self._open_external,
            abg=C["border"]).pack(fill="x", padx=12, pady=2)

        # Métricas rápidas
        self._stitle(S, "MÉTRICAS  RÁPIDAS")
        self._mc = tk.Frame(S, bg=C["panel"],
                             highlightbackground=C["border"],
                             highlightthickness=1)
        self._mc.pack(fill="x", padx=12, pady=4)
        self._metricas({})

        # Progreso
        self._stitle(S, "PROGRESO")
        sty = ttk.Style(); sty.theme_use("clam")
        sty.configure("P.Horizontal.TProgressbar",
                       troughcolor=C["border"], background=C["a1"],
                       bordercolor=C["panel"])
        self._pbar = ttk.Progressbar(S, style="P.Horizontal.TProgressbar",
                                      mode="indeterminate", length=274)
        self._pbar.pack(fill="x", padx=12, pady=4)
        self._lbl_step = tk.Label(S, text="—", font=(FM, 8),
                                   fg=C["muted"], bg=C["bg"])
        self._lbl_step.pack(anchor="w", padx=14)

        # Consola
        self._stitle(S, "CONSOLA")
        lw = tk.Frame(S, bg=C["panel"],
                      highlightbackground=C["border"], highlightthickness=1)
        lw.pack(fill="both", expand=True, padx=12, pady=(4, 14))
        self._log_w = tk.Text(lw, bg=C["panel"], fg=C["muted2"],
                               font=(FM, 8), relief="flat",
                               state="disabled", wrap="word")
        sb2 = tk.Scrollbar(lw, command=self._log_w.yview,
                           bg=C["border"], troughcolor=C["panel"], relief="flat")
        self._log_w.configure(yscrollcommand=sb2.set)
        sb2.pack(side="right", fill="y")
        self._log_w.pack(fill="both", expand=True, padx=4, pady=4)
        self._log_w.tag_configure("ok",   foreground=C["ok"])
        self._log_w.tag_configure("err",  foreground=C["danger"])
        self._log_w.tag_configure("info", foreground=C["a1"])
        self._log_w.tag_configure("dim",  foreground=C["muted2"])

    def _stitle(self, parent, text):
        f = tk.Frame(parent, bg=C["bg"])
        f.pack(fill="x", padx=12, pady=(12, 2))
        tk.Label(f, text=text, font=(FM, 8, "bold"),
                 fg=C["muted"], bg=C["bg"]).pack(side="left")
        tk.Frame(f, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(8, 0), pady=5)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _tabs(self):
        tabbar = tk.Frame(self._main, bg=C["panel"], height=36)
        tabbar.pack(fill="x"); tabbar.pack_propagate(False)

        self._tf  = {}
        self._tbn = {}

        for key, label in [("viewer", "  🖼  Visualizador  "),
                            ("hist",  "  📋  Historial  ")]:
            b = tk.Button(tabbar, text=label, font=(FM, 9, "bold"),
                          fg=C["muted2"], bg=C["panel"],
                          activeforeground=C["a1"],
                          activebackground=C["panel"],
                          relief="flat", bd=0, cursor="hand2",
                          padx=14, pady=8,
                          command=lambda k=key: self._tab(k))
            b.pack(side="left")
            self._tbn[key] = b
            f = tk.Frame(self._main, bg=C["bg"])
            self._tf[key] = f

        self._build_viewer()
        self._build_hist()
        self._tab("viewer")

    def _tab(self, key):
        for k, f in self._tf.items(): f.pack_forget()
        self._tf[key].pack(fill="both", expand=True)
        for k, b in self._tbn.items():
            b.config(fg=(C["a1"] if k == key else C["muted2"]),
                     bg=(C["bg"] if k == key else C["panel"]))

    # ── Viewer tab ────────────────────────────────────────────────────────────

    def _build_viewer(self):
        tab = self._tf["viewer"]

        # Toolbar zoom
        tb = tk.Frame(tab, bg=C["panel2"], height=32)
        tb.pack(fill="x"); tb.pack_propagate(False)

        tk.Label(tb, text="  Zoom:", font=(FM, 8),
                 fg=C["muted"], bg=C["panel2"]).pack(side="left", padx=(8, 0))
        for txt, fac in [("50%", .5), ("100%", 1.0), ("150%", 1.5), ("Ajustar", 0)]:
            tk.Button(tb, text=txt, font=(FM, 8),
                      fg=C["muted2"], bg=C["panel2"],
                      activeforeground=C["a1"],
                      activebackground=C["panel2"],
                      relief="flat", bd=0, cursor="hand2", padx=8,
                      command=lambda f=fac: self._zoom_set(f)
                      ).pack(side="left")

        self._lbl_ts = tk.Label(tb, text="",
                                 font=(FM, 8), fg=C["muted"], bg=C["panel2"])
        self._lbl_ts.pack(side="right", padx=12)

        # Canvas
        cw = tk.Frame(tab, bg=C["bg"])
        cw.pack(fill="both", expand=True, padx=14, pady=10)

        self._canvas = tk.Canvas(cw, bg=C["panel"],
                                  highlightbackground=C["border"],
                                  highlightthickness=1, cursor="crosshair")
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda e: self._draw())
        self._canvas.bind("<MouseWheel>",
            lambda e: self._zoom_set(self._zoom + (.1 if e.delta > 0 else -.1)))

        self._canvas.create_text(660, 310,
            text="◈  Selecciona un archivo .nii  y  presiona  Analizar imagen",
            fill=C["muted"], font=(FM, 12), tags="ph")

    # ── Historial tab ─────────────────────────────────────────────────────────

    def _build_hist(self):
        tab = self._tf["hist"]
        h = tk.Frame(tab, bg=C["bg"])
        h.pack(fill="x", padx=16, pady=(12, 4))
        lbl(h, "Historial de análisis", size=11,
            color=C["a1"], bold=True).pack(side="left")
        btn(h, "🗑 Limpiar", self._clear_hist,
            abg=C["border"]).pack(side="right")
        tk.Frame(tab, bg=C["border"], height=1).pack(fill="x", padx=16, pady=2)

        wrap = tk.Frame(tab, bg=C["bg"])
        wrap.pack(fill="both", expand=True, padx=14, pady=8)

        sb = tk.Scrollbar(wrap, bg=C["border"],
                          troughcolor=C["panel"], relief="flat")
        sb.pack(side="right", fill="y")

        self._hc = tk.Canvas(wrap, bg=C["bg"],
                              yscrollcommand=sb.set,
                              highlightthickness=0)
        self._hc.pack(fill="both", expand=True)
        sb.config(command=self._hc.yview)

        self._hi = tk.Frame(self._hc, bg=C["bg"])
        self._hc.create_window((0, 0), window=self._hi, anchor="nw", tags="win")
        self._hi.bind("<Configure>",
            lambda e: self._hc.configure(scrollregion=self._hc.bbox("all")))

        lbl(self._hi, "Sin análisis previos.", color=C["muted"]).pack(pady=30)

    # ── Statusbar ─────────────────────────────────────────────────────────────

    def _statusbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=24)
        bar.pack(fill="x"); bar.pack_propagate(False)

        self._dot = tk.Label(bar, text="●", font=(FM, 9),
                             fg=C["ok"], bg=C["panel"])
        self._dot.pack(side="left", padx=(12, 4))
        self._stlbl = tk.Label(bar, text="Listo",
                                font=(FM, 8), fg=C["muted"], bg=C["panel"])
        self._stlbl.pack(side="left")
        tk.Label(bar,
                 text="modelo: modelo_tumores.pth  ·  threshold: 0.5  ",
                 font=(FM, 8), fg=C["muted"], bg=C["panel"]).pack(side="right")

    # ── Métricas card ─────────────────────────────────────────────────────────

    def _metricas(self, datos):
        for w in self._mc.winfo_children(): w.destroy()
        if not datos:
            tk.Label(self._mc, text="Sin datos", font=(FM, 8),
                     fg=C["muted"], bg=C["panel"]).pack(pady=10)
            return
        rows = [
            ("TUMOR",    "✓ PRESENTE" if datos.get("tumor_presente")
                         else "✗ NO DETECTADO",
             C["danger"] if datos.get("tumor_presente") else C["ok"]),
            ("UBICACIÓN", datos.get("ubicacion", "—"), C["text"]),
            ("DIÁMETRO",  datos.get("diametro",   "—"), C["a1"]),
            ("VOLUMEN",   datos.get("volumen",    "—"), C["a1"]),
            ("CONFIANZA", datos.get("confianza",  "—"), C["ok"]),
        ]
        for k, v, col in rows:
            r = tk.Frame(self._mc, bg=C["panel"])
            r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=f"{k:<10}", font=(FM, 8),
                     fg=C["muted"], bg=C["panel"]).pack(side="left")
            tk.Label(r, text=v, font=(FM, 8, "bold"),
                     fg=col, bg=C["panel"]).pack(side="left")

        riesgo = datos.get("riesgo", "")
        col    = RIESGO_C.get(riesgo, C["muted2"])
        badge  = tk.Frame(self._mc, bg=col)
        badge.pack(fill="x", padx=8, pady=(6, 8))
        tk.Label(badge, text=f"  {riesgo}  ",
                 font=(FM, 8, "bold"), fg=C["bg"], bg=col).pack(pady=3)

    # ── Acciones ──────────────────────────────────────────────────────────────

    def _pick_file(self):
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo NIfTI",
            filetypes=[("NIfTI", "*.nii *.nii.gz"), ("Todos", "*.*")])
        if not ruta: return
        self._ruta.set(ruta)
        self._lbl_arc.config(text=os.path.basename(ruta), fg=C["a1"])
        self._btn_go.config(state="normal")
        self._log(f"Archivo: {os.path.basename(ruta)}", "info")

    def _run(self):
        ruta = self._ruta.get()
        if not ruta: return
        self._btn_go.config(state="disabled")
        self._btn_pdf.config(state="disabled")
        self._pbar.start(10)
        self._setstatus("Procesando…", C["warn"])
        threading.Thread(target=self._worker, args=(ruta,), daemon=True).start()

    def _worker(self, ruta):
        try:
            self._step("[1/3] Ejecutando U-Net…")
            original, mascara, mascara_prob, slices_mascara = predecir_tumor(ruta)

            self._step("[2/3] Calculando métricas…")
            datos = obtener_diagnostico_clinico(
                mascara.squeeze(), mascara_prob, slices_mascara)

            self._step("[3/3] Renderizando figura…")
            img_pil = render_figure(original, mascara.squeeze(), datos)

            os.makedirs("reportes", exist_ok=True)
            self._img_temp = "reportes/temp_panel.png"
            img_pil.save(self._img_temp, dpi=(140, 140))
            self._img_panel = img_pil
            self._datos     = datos

            self.after(0, self._ok, datos, ruta)
        except Exception as e:
            self.after(0, self._err, str(e))

    def _ok(self, datos, ruta):
        self._pbar.stop()
        self._lbl_step.config(text="Completado")
        self._metricas(datos)
        self._zoom = 0.0
        self._draw()
        self._btn_go.config(state="normal")
        self._btn_pdf.config(state="normal")
        self._setstatus("Análisis completado", C["ok"])
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._lbl_ts.config(text=f"Último análisis: {ts}")
        self._log(f"Completado — Riesgo: {datos.get('riesgo')}", "ok")
        self._add_hist(datos, ruta, ts)
        self._tab("viewer")

    def _err(self, msg):
        self._pbar.stop()
        self._lbl_step.config(text="Error")
        self._btn_go.config(state="normal")
        self._setstatus("Error en análisis", C["danger"])
        self._log(f"ERROR: {msg}", "err")
        messagebox.showerror("Error en el análisis", msg)

    def _export_pdf(self):
        if not self._datos or not self._img_temp: return
        nombre = os.path.basename(self._ruta.get())
        try:
            path = generar_reporte_final(self._datos, nombre, self._img_temp)
            self._log(f"PDF: {path}", "ok")
            if messagebox.askyesno("PDF exportado",
                                    f"Guardado en:\n{path}\n\n¿Abrir carpeta?"):
                import subprocess
                subprocess.Popen(
                    f'explorer /select,"{os.path.abspath(path)}"')
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    def _open_external(self):
        if self._img_temp and os.path.exists(self._img_temp):
            import subprocess
            subprocess.Popen(["start", "", self._img_temp], shell=True)
        else:
            messagebox.showinfo("Sin imagen", "Realiza un análisis primero.")

    # ── Historial ─────────────────────────────────────────────────────────────

    def _add_hist(self, datos, ruta, ts):
        self._historial.append({"d": datos, "r": ruta, "ts": ts})
        for w in self._hi.winfo_children(): w.destroy()

        for i, item in enumerate(reversed(self._historial)):
            d   = item["d"]
            idx = len(self._historial) - i
            row = tk.Frame(self._hi, bg=C["panel"],
                           highlightbackground=C["border"],
                           highlightthickness=1)
            row.pack(fill="x", pady=3, padx=2)

            hdr = tk.Frame(row, bg=C["panel2"])
            hdr.pack(fill="x")
            tk.Label(hdr, text=f"  #{idx}",
                     font=(FM, 9, "bold"), fg=C["a1"],
                     bg=C["panel2"]).pack(side="left", padx=8, pady=4)
            tk.Label(hdr, text=item["ts"], font=(FM, 8),
                     fg=C["muted"], bg=C["panel2"]).pack(side="left")
            riesgo = d.get("riesgo", "")
            rc     = RIESGO_C.get(riesgo, C["muted2"])
            tk.Label(hdr, text=f"  {riesgo}  ",
                     font=(FM, 8, "bold"), fg=C["bg"],
                     bg=rc).pack(side="right", padx=8, pady=4)

            det = tk.Frame(row, bg=C["panel"])
            det.pack(fill="x", padx=12, pady=6)
            for k, v in [
                ("Archivo",   os.path.basename(item["r"])),
                ("Localiz.",  d.get("ubicacion", "—")),
                ("Diámetro",  d.get("diametro",  "—")),
                ("Volumen",   d.get("volumen",   "—")),
                ("Confianza", d.get("confianza", "—")),
            ]:
                col = tk.Frame(det, bg=C["panel"])
                col.pack(side="left", padx=8)
                tk.Label(col, text=k, font=(FM, 7),
                         fg=C["muted"], bg=C["panel"]).pack()
                tk.Label(col, text=v, font=(FM, 8, "bold"),
                         fg=C["text"], bg=C["panel"]).pack()

    def _clear_hist(self):
        if not self._historial: return
        if messagebox.askyesno("Limpiar historial",
                                "¿Eliminar todo el historial?"):
            self._historial.clear()
            for w in self._hi.winfo_children(): w.destroy()
            lbl(self._hi, "Sin análisis previos.",
                color=C["muted"]).pack(pady=30)

    # ── Canvas ────────────────────────────────────────────────────────────────

    def _draw(self):
        if self._img_panel is None: return
        self._canvas.delete("all")
        cw = self._canvas.winfo_width()  or 900
        ch = self._canvas.winfo_height() or 440
        img = self._img_panel.copy()

        if self._zoom <= 0:
            img.thumbnail((cw, ch), _LANCZOS)
            self._zoom = img.width / self._img_panel.width
        else:
            nw = int(self._img_panel.width  * self._zoom)
            nh = int(self._img_panel.height * self._zoom)
            img = img.resize((nw, nh), _LANCZOS)

        self._tk_img = ImageTk.PhotoImage(img)
        self._canvas.create_image(cw // 2, ch // 2,
                                   anchor="center", image=self._tk_img)

    def _zoom_set(self, f):
        self._zoom = max(0.2, min(f, 3.0)) if f > 0 else 0.0
        self._draw()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg, tag="dim"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_w.config(state="normal")
        self._log_w.insert("end", f"[{ts}] ", "dim")
        self._log_w.insert("end", f"{msg}\n", tag)
        self._log_w.see("end")
        self._log_w.config(state="disabled")

    def _setstatus(self, msg, color=None):
        self._stlbl.config(text=msg)
        self._dot.config(fg=color or C["ok"])

    def _step(self, msg):
        self._lbl_step.config(text=msg)
        self._log(msg, "info")
        self.update_idletasks()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = NeuroimagenApp()
    app.mainloop()