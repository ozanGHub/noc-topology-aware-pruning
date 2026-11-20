"""
dnn_accelerators_chart.py
=========================
Generates the DNN Accelerator Comparison Chart as SVG and PNG.

The chart is a 3D oblique-projection scatter plot comparing four DNN
accelerator types (ASIC, FPGA, NоC-based DNN, CPU/GPU) across three axes:
  - Reconfigurability  (Y, vertical)
  - Flexibility        (X, horizontal)
  - Power Efficiency   (Z, oblique depth — High at front-right, Low at back)

Dependencies:
    pip install pymupdf

Usage:
    python dnn_accelerators_chart.py

Outputs (written to the same directory as this script):
    dnn_accelerators.svg
    dnn_accelerators.png   (2x resolution)
"""

import os
import fitz  # PyMuPDF

# ─────────────────────────────────────────────────────────────────────────────
# 1. CANVAS
# ─────────────────────────────────────────────────────────────────────────────

SVG_W     = 680
SVG_H     = 560
PNG_SCALE = 2       # PNG exported at 2x SVG resolution

# ─────────────────────────────────────────────────────────────────────────────
# 2. BOX GEOMETRY
#
#   Oblique projection:
#     sx = ORIG_X + fx * X_SPAN + d * ZX
#     sy = ORIG_Y - ry * Y_SPAN - d * ZY
#   where d = 1 - pe  (pe=1 -> High/front -> d=0,  pe=0 -> Low/back -> d=1)
#
#   8 corners:
#     Front face (d=0): FBL=(100,420)  FBR=(430,420)  FTL=(100,150)  FTR=(430,150)
#     Back  face (d=1): BBL=(256,362)  BBR=(586,362)  BTL=(256, 92)  BTR=(586, 92)
#
#   Axis label tick positions (at 20% and 80% of each axis):
#     Y: ry=0.2 -> sy=366 "At design time"
#        ry=0.5 -> sy=285 "At boost time"
#        ry=0.8 -> sy=204 "At runtime"
#     X: fx=0.2 -> sx=166 "Single application"
#        fx=0.8 -> sx=364 "General purpose"
#     Z: d=0.2(High) -> from FBR: (461,408)
#        d=0.5(Med)  -> from FBR: (508,391)
#        d=0.8(Low)  -> from FBR: (555,374)
# ─────────────────────────────────────────────────────────────────────────────

ORIG_X = 100    # front-bottom-left x
ORIG_Y = 420    # front-bottom-left y
X_SPAN = 330    # Flexibility axis pixel length
Y_SPAN = 270    # Reconfigurability axis pixel length
ZX     = 156    # Z axis horizontal component (depth vector x)
ZY     = 58     # Z axis vertical component   (depth vector y, upward)

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPHERE DATA
#    (label_lines, cx, cy, radius, gradient_id, stroke_color,
#     shadow_cx, shadow_cy, shadow_rx, shadow_ry, shadow_opacity)
# ─────────────────────────────────────────────────────────────────────────────

SPHERES = [
    (["ASIC"],              197, 354, 50, "gASIC", "#1a6650", 197, 416, 46, 12, 0.10),
    (["FPGA"],              287, 265, 42, "gFPGA", "#5e6e1e", 287, 406, 36, 10, 0.09),
    (["NоC-based", "DNN"], 401, 207, 37, "gNoC",  "#855818", 401, 398, 30,  9, 0.08),
    (["CPU/", "GPU"],       525, 151, 31, "gCPU",  "#6e4462", 554, 372, 25,  7, 0.11),
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. COLORS
# ─────────────────────────────────────────────────────────────────────────────

COL_GRID  = "#888780"   # face planes and grid lines
COL_EDGE  = "#5F5E5A"   # box edges and tick marks
COL_AXIS  = "#2C2C2A"   # main axis lines and labels
COL_TICK  = "#5F5E5A"   # tick labels
FONT      = "Arial, sans-serif"

# ─────────────────────────────────────────────────────────────────────────────
# 5. SVG ELEMENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ln(x1, y1, x2, y2, stroke, width, dasharray=None, opacity=1.0):
    dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash} '
            f'stroke-opacity="{opacity}"/>')

def poly(pts, fill, fill_op, stroke, stroke_w, dash, stroke_op):
    p = " ".join(f"{x},{y}" for x, y in pts)
    return (f'<polygon points="{p}" fill="{fill}" fill-opacity="{fill_op}" '
            f'stroke="{stroke}" stroke-width="{stroke_w}" '
            f'stroke-dasharray="{dash}" stroke-opacity="{stroke_op}"/>')

def txt(x, y, content, anchor, size, weight, fill, transform=None,
        dominant_baseline=None):
    t  = f' transform="{transform}"'         if transform          else ""
    db = f' dominant-baseline="{dominant_baseline}"' if dominant_baseline else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}"{t}{db} '
            f'font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" font-family="{FONT}">{content}</text>')

def tspan_txt(x, y, lines, anchor, size, weight, fill, lh=14):
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{ln}</tspan>'
        for i, ln in enumerate(lines)
    )
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" font-family="{FONT}">{spans}</text>')

def ell(cx, cy, rx, ry, fill, fill_op):
    return (f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
            f'fill="{fill}" fill-opacity="{fill_op}"/>')

def circ(cx, cy, r, fill, stroke=None, stroke_w=1, opacity=1.0):
    s = (f' stroke="{stroke}" stroke-width="{stroke_w}" opacity="{opacity}"'
         if stroke else "")
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{s}/>'

# ─────────────────────────────────────────────────────────────────────────────
# 6. GRADIENT & MARKER DEFS
# ─────────────────────────────────────────────────────────────────────────────

DEFS = """  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <radialGradient id="gASIC" cx="36%" cy="32%" r="60%">
      <stop offset="0%" stop-color="#7ecfb0"/>
      <stop offset="55%" stop-color="#3aaa80"/>
      <stop offset="100%" stop-color="#1a6650"/>
    </radialGradient>
    <radialGradient id="gFPGA" cx="36%" cy="32%" r="60%">
      <stop offset="0%" stop-color="#d0da82"/>
      <stop offset="55%" stop-color="#9eb842"/>
      <stop offset="100%" stop-color="#5e6e1e"/>
    </radialGradient>
    <radialGradient id="gNoC" cx="36%" cy="32%" r="60%">
      <stop offset="0%" stop-color="#eabf80"/>
      <stop offset="55%" stop-color="#c08840"/>
      <stop offset="100%" stop-color="#855818"/>
    </radialGradient>
    <radialGradient id="gCPU" cx="36%" cy="32%" r="60%">
      <stop offset="0%" stop-color="#d6afc6"/>
      <stop offset="55%" stop-color="#a87898"/>
      <stop offset="100%" stop-color="#6e4462"/>
    </radialGradient>
    <radialGradient id="gHL" cx="32%" cy="28%" r="48%">
      <stop offset="0%" stop-color="#fff" stop-opacity="0.52"/>
      <stop offset="100%" stop-color="#fff" stop-opacity="0"/>
    </radialGradient>
    <mask id="imagine-text-gaps-owau5k" maskUnits="userSpaceOnUse">
      <rect x="0" y="0" width="690" height="560" fill="white"/>
      <rect x="9.59375" y="357.3333435058594" width="83.6982650756836" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="14.458333969116211" y="276.3333435058594" width="78.83076477050781" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="29.072917938232422" y="195.33334350585938" width="64.21727752685547" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="115.84375" y="428.3333435058594" width="99.7362289428711" height="18" fill="black" rx="2"/>
      <rect x="316.828125" y="428.3333435058594" width="94.6365966796875" height="18" fill="black" rx="2"/>
      <rect x="464" y="387.3333435058594" width="31.870647430419922" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="511" y="370.3333435058594" width="50.375" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="558" y="353.3333435058594" width="29.6588134765625" height="17.33333396911621" fill="black" rx="2"/>
      <rect x="-57.40625" y="-14.666666984558105" width="115.25492858886719" height="20" fill="black" rx="2"/>
      <rect x="230.6979217529297" y="445.3333435058594" width="69.04642868041992" height="20" fill="black" rx="2"/>
      <rect x="612.8499755859375" y="329.3333435058594" width="46.26041793823242" height="20" fill="black" rx="2"/>
      <rect x="612.8499755859375" y="344.3333435058594" width="71.61397552490234" height="20" fill="black" rx="2"/>
      <rect x="175.53646850585938" y="339.3333435058594" width="42.81298828125" height="21.33333396911621" fill="black" rx="2"/>
      <rect x="263.6302185058594" y="250.33334350585938" width="47.515342712402344" height="21.33333396911621" fill="black" rx="2"/>
      <rect x="362.88543701171875" y="164" width="76.22915649414062" height="33.33332824707031" fill="black" rx="2"/>
      <rect x="504.890625" y="131" width="40.21881103515625" height="33.33332824707031" fill="black" rx="2"/>
    </mask>
  </defs>"""

# ─────────────────────────────────────────────────────────────────────────────
# 7. ASSEMBLE SVG
# ─────────────────────────────────────────────────────────────────────────────

def build_svg():
    p = []

    p.append(f'<svg width="{SVG_W}" height="{SVG_H}" '
             f'viewBox="0 0 {SVG_W} {SVG_H}" '
             f'role="img" xmlns="http://www.w3.org/2000/svg">')
    p.append('  <title>DNN Accelerator Comparison Chart</title>')
    p.append('  <desc>3D oblique scatter: ASIC, FPGA, NоC-based DNN, CPU/GPU '
             'vs Reconfigurability, Flexibility, Power Efficiency.</desc>')
    p.append(f'  <rect width="{SVG_W}" height="{SVG_H}" fill="white"/>')
    p.append(DEFS)

    # ── Face planes ──────────────────────────────────────────────────────────
    p.append("\n  <!-- FACE PLANES -->")
    p.append(poly([(100,420),(430,420),(586,362),(256,362)],  # base floor
                  COL_GRID, 0.05, COL_GRID, 0.8, "5 4", 0.32))
    p.append(poly([(100,420),(100,150),(256,92),(256,362)],   # left wall
                  COL_GRID, 0.04, COL_GRID, 0.8, "5 4", 0.32))
    p.append(poly([(256,362),(586,362),(586,92),(256,92)],    # back wall
                  COL_GRID, 0.05, COL_GRID, 0.8, "5 4", 0.32))

    # ── Label grid lines ─────────────────────────────────────────────────────
    p.append("\n  <!-- LABEL GRID LINES -->")
    G = COL_GRID
    # Base: depth lines at fx=0.2 and fx=0.8
    p.append(ln(166,420, 322,362, G, 0.6, "4 5", 0.26))
    p.append(ln(364,420, 520,362, G, 0.6, "4 5", 0.26))
    # Base: X lines at d=0.2 and d=0.8
    p.append(ln(131,408, 461,408, G, 0.6, "4 5", 0.26))
    p.append(ln(225,374, 555,374, G, 0.6, "4 5", 0.26))
    # Left wall: verticals at d=0.2 and d=0.8
    p.append(ln(131,420, 131,150, G, 0.6, "4 5", 0.22))
    p.append(ln(225,408, 225,138, G, 0.6, "4 5", 0.22))
    # Left wall: horizontals at ry=0.2 and ry=0.8
    p.append(ln(100,366, 256,308, G, 0.6, "4 5", 0.22))
    p.append(ln(100,204, 256,146, G, 0.6, "4 5", 0.22))
    # Back wall: horizontals at ry=0.2 and ry=0.8
    p.append(ln(256,308, 586,308, G, 0.6, "4 5", 0.18))
    p.append(ln(256,146, 586,146, G, 0.6, "4 5", 0.18))
    # Back wall: verticals at fx=0.2 and fx=0.8
    p.append(ln(322,362, 322, 92, G, 0.6, "4 5", 0.18))
    p.append(ln(520,362, 520, 92, G, 0.6, "4 5", 0.18))

    # ── Box edges ────────────────────────────────────────────────────────────
    p.append("\n  <!-- BOX EDGES -->")
    E = COL_EDGE
    p.append(ln(100,420, 430,420, E, 1.2, opacity=0.55))  # front bottom (solid)
    p.append(ln(100,420, 100,150, E, 1.2, opacity=0.55))  # front left   (solid)
    p.append(ln(100,150, 430,150, E, 0.8, "5 4", 0.38))   # front top
    p.append(ln(430,420, 430,150, E, 0.8, "5 4", 0.38))   # front right
    p.append(ln(100,420, 256,362, E, 0.8, "5 4", 0.38))   # bottom-left depth
    p.append(ln(430,420, 586,362, E, 0.8, "5 4", 0.38))   # bottom-right depth
    p.append(ln(100,150, 256, 92, E, 0.8, "5 4", 0.38))   # top-left depth
    p.append(ln(430,150, 586, 92, E, 0.8, "5 4", 0.38))   # top-right depth
    p.append(ln(256,362, 586,362, E, 0.8, "5 4", 0.38))   # back bottom
    p.append(ln(256, 92, 586, 92, E, 0.8, "5 4", 0.38))   # back top
    p.append(ln(256,362, 256, 92, E, 0.8, "5 4", 0.38))   # back left
    p.append(ln(586,362, 586, 92, E, 0.8, "5 4", 0.38))   # back right

    # ── Axes ─────────────────────────────────────────────────────────────────
    p.append("\n  <!-- AXES -->")
    A = COL_AXIS
    # Y: Reconfigurability — up from FBL, extends 20% beyond box
    p.append(f'<line x1="100" y1="434" x2="100" y2="134" stroke="{A}" '
             f'stroke-width="1.8" stroke-opacity="0.8" marker-end="url(#arrow)"/>')
    # X: Flexibility — right from FBL, extends 20% beyond box
    p.append(f'<line x1="86" y1="420" x2="446" y2="420" stroke="{A}" '
             f'stroke-width="1.8" stroke-opacity="0.8" marker-end="url(#arrow)"/>')
    # Z: Power Efficiency — from FBR along depth vector (156,-58), tip at 1.2x
    p.append(f'<line x1="430" y1="420" x2="617" y2="350" stroke="{A}" '
             f'stroke-width="1.8" stroke-opacity="0.8" marker-end="url(#arrow)"/>')

    # ── Y axis ticks & labels ────────────────────────────────────────────────
    p.append("\n  <!-- Y AXIS TICKS & LABELS -->")
    for sy, label in [(366, "At design time"), (285, "At boost time"), (204, "At runtime")]:
        p.append(ln(95, sy, 105, sy, COL_TICK, 1, opacity=0.6))
        p.append(txt(89, sy+4, label, "end", 11, "normal", COL_TICK))

    # ── X axis ticks & labels ────────────────────────────────────────────────
    p.append("\n  <!-- X AXIS TICKS & LABELS -->")
    for sx, label in [(166, "Single application"), (364, "General purpose")]:
        p.append(ln(sx, 415, sx, 427, COL_TICK, 1, opacity=0.6))
        p.append(txt(sx, 441, label, "middle", 11, "normal", COL_TICK))

    # ── Z axis ticks & labels (perpendicular offset from axis line) ───────────
    p.append("\n  <!-- Z AXIS TICKS & LABELS -->")
    z_ticks = [
        # tick x1,y1 → x2,y2    label x,  y,   text
        (461, 408, 465, 400,     468,  413, "High"),
        (508, 391, 512, 383,     515,  396, "Medium"),
        (555, 374, 559, 366,     562,  379, "Low"),
    ]
    for x1, y1, x2, y2, lx, ly, label in z_ticks:
        p.append(ln(x1, y1, x2, y2, COL_TICK, 1, opacity=0.6))
        p.append(txt(lx, ly, label, "start", 11, "normal", COL_TICK))

    # ── Axis main labels ─────────────────────────────────────────────────────
    p.append("\n  <!-- AXIS MAIN LABELS -->")
    # Reconfigurability: rotated vertically, shifted left to avoid overlap
    p.append(f'<text font-size="13" font-weight="500" fill="{COL_AXIS}" '
             f'font-family="{FONT}" transform="translate(10,285) rotate(-90)">'
             f'<tspan text-anchor="middle">Reconfigurability</tspan></text>')
    # Flexibility
    p.append(txt(265, 460, "Flexibility", "middle", 13, "500", COL_AXIS))
    # Power Efficiency (two lines at axis tip)
    p.append(txt(619, 344, "Power",      "start", 13, "500", COL_AXIS))
    p.append(txt(619, 359, "Efficiency", "start", 13, "500", COL_AXIS))

    # ── Floor shadow ellipses ────────────────────────────────────────────────
    p.append("\n  <!-- FLOOR SHADOW ELLIPSES -->")
    for _, _, _, _, _, _, shcx, shcy, shrx, shry, shop in SPHERES:
        p.append(ell(shcx, shcy, shrx, shry, "#000", shop))

    # ── Spheres ───────────────────────────────────────────────────────────────
    p.append("\n  <!-- SPHERES -->")
    for label_lines, cx, cy, r, grad, stroke_col, *_ in SPHERES:
        name = " ".join(label_lines)
        p.append(f"\n  <!-- {name} -->")
        p.append(circ(cx, cy, r, f"url(#{grad})"))          # base gradient
        p.append(circ(cx, cy, r, "url(#gHL)"))              # highlight
        p.append(circ(cx, cy, r, "none", stroke_col, 1, 0.4))  # outline
        # Label: single line or multi-line
        if len(label_lines) == 1:
            p.append(txt(cx, cy+4, label_lines[0], "middle", 14, "600",
                         "white", dominant_baseline="central"))
        else:
            lh = 14
            y0 = cy - ((len(label_lines) - 1) * lh) // 2 + 4
            p.append(tspan_txt(cx, y0, label_lines, "middle", 12, "600", "white", lh))

    p.append("\n</svg>")
    return "\n".join(p)

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export(svg_str, out_dir="."):
    svg_path = os.path.join(out_dir, "dnn_accelerators.svg")
    png_path = os.path.join(out_dir, "dnn_accelerators.png")

    # Write SVG
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"SVG -> {svg_path}")

    # PNG via PyMuPDF — renders SVG natively with full gradient/opacity support
    doc  = fitz.open(svg_path)
    page = doc[0]
    mat  = fitz.Matrix(PNG_SCALE, PNG_SCALE)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_path)
    print(f"PNG -> {png_path}  ({pix.width}x{pix.height} px)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    svg_str = build_svg()
    export(svg_str, out_dir)
