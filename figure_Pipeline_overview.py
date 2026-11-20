"""
pipeline_overview.py
====================
Generates the NoC-aware DNN pruning & mapping pipeline diagram as SVG, PNG, and PDF.

Dependencies:
    pip install pymupdf

Usage:
    python pipeline_overview.py

Outputs (written to the same directory as this script):
    pipeline_overview.svg
    pipeline_overview.png   (3x resolution)
    pipeline_overview.pdf
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. LAYOUT  –  all coordinates in SVG user units
# ─────────────────────────────────────────────────────────────────────────────

VIEWBOX_X  = 60 #78
VIEWBOX_Y  = 12
VIEWBOX_W  = 488 #470
VIEWBOX_H  = 846 #836
SVG_WIDTH  = 470

CX         = 310   # horizontal centre
BOX_LEFT   = 110
BOX_WIDTH  = 400

INPUT_Y    = 16;  INPUT_H   = 38
S1_Y       = 76;  S1_H     = 64
S2_Y       = 162; S2_H     = 64
S3_Y       = 248; S3_H     = 78
D1_CY      = 384
S4_Y       = 440; S4_H     = 78
D2_CY      = 576
S5A_Y      = 632; S5A_H    = 64
S5B_Y      = 720; S5B_H    = 64
OUTPUT_Y   = 820; OUTPUT_H  = 38

D_HW       = 80   # diamond horizontal half-width
D_HH       = 34   # diamond vertical half-height

PNG_SCALE  = 3    # PNG export resolution multiplier

# ─────────────────────────────────────────────────────────────────────────────
# 2. COLOURS & TYPOGRAPHY
# ─────────────────────────────────────────────────────────────────────────────

FONT = "Arial, sans-serif"

C = {
    # Standard stage (teal)
    "std_fill":    "#E1F5EE",
    "std_stroke":  "#0F6E56",
    "std_h_text":  "#085041",
    "std_s_text":  "#0F6E56",
    # Proposed heuristic (purple)
    "prop_fill":   "#EEEDFE",
    "prop_stroke": "#534AB7",
    "prop_h_text": "#26215C",
    "prop_s_text": "#3C3489",
    # Terminal pill (warm gray)
    "term_fill":   "#F1EFE8",
    "term_stroke": "#5F5E5A",
    "term_h_text": "#2C2C2A",
    # Connectors
    "arrow":       "#555555",
    "dash":        "#888888",
    # Annotation
    "label":       "#888888",
    "body":        "#444444",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT CONTENT  –  edit labels here, no need to touch drawing code
# ─────────────────────────────────────────────────────────────────────────────

INPUT_LABEL  = "Trained DNN model"
OUTPUT_LABEL = "Pruned + mapped NoC DNN"

STAGES = {
    1: {
        "title": "Stage 1 — Pre-pruning",
        "lines": ["Iterative L1 link pruning + fine-tuning · remove lonely neurons"],
        "kind": "std", "num": "1",
    },
    2: {
        "title": "Stage 2 — DNN-to-WCTG mapping",
        "lines": ["Map layers → DNN_G(N,C) · compute initial neuron groups V"],
        "kind": "std", "num": "2",
    },
    3: {
        "title": "Stage 3 — Comm. cost aware neuron pruning",
        "lines": [
            "Compute C = H (top-n%) ∩ L (bottom-m%)",
            "Select n* = argmax Δψ · remove n* · fine-tune e epochs",
        ],
        "kind": "prop", "num": "3",
    },
    4: {
        "title": "Stage 4 — Bridge-link edge pruning",
        "lines": [
            "Identify inter-group edges (bridge links) · sort by |w| ascending",
            "Zero K_e cheapest edges in B · fallback to L2 if B = ∅",
        ],
        "kind": "prop", "num": "4",
    },
    "5a": {
        "title": "Stage 5a — Neuron grouping",
        "lines": ["In-Degree Best-Fit Decreasing Neuron Grouping → DNN_AG(V,E) · minimise Ω"],
        "kind": "prop", "num": "5a",
    },
    "5b": {
        "title": "Stage 5b — NoC PE mapping",
        "lines": ["SA-based group-to-PE swap → DNN_TG(P,L) · minimise ψ = Σ w·d"],
        "kind": "std", "num": "5b",
    },
}

LOOP_ANNOTATION = ["T_pre rounds · ρ, e_pre (tunable)"]
DIAMOND_LABEL   = "β₀ − β ≤ α ?"

D1_LEFT_LABEL  = "C = ∅"
D1_RIGHT_LABEL = ["no → stop", "(budget done)"]
D2_LEFT_LABEL  = "step done"
D2_RIGHT_LABEL = ["no → restore,", "stop"]

LEGEND_STD  = "Standard stage"
LEGEND_PROP = ["Proposed heuristic", "stage"]

# ─────────────────────────────────────────────────────────────────────────────
# 4. DRAWING HELPERS  –  all styles are inline (no CSS classes)
# ─────────────────────────────────────────────────────────────────────────────

def svg_rect(x, y, w, h, rx=8, fill="#fff", stroke="#000", stroke_width=0.8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')


def svg_line(x1, y1, x2, y2, stroke=None, stroke_width=1, marker_end="url(#arr)"):
    stroke = stroke or C["arrow"]
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" fill="none" '
            f'marker-end="{marker_end}"/>')


def svg_path(d, stroke=None, stroke_width=0.8,
             dasharray="4 3", marker_end="url(#arr-dash)"):
    stroke = stroke or C["dash"]
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-dasharray="{dasharray}" fill="none" marker-end="{marker_end}"/>')


def svg_text(x, y, content, font_size=11, font_weight=400,
             fill=None, anchor="middle"):
    fill = fill or C["body"]
    return (f'<text x="{x}" y="{y}" '
            f'font-family="{FONT}" font-size="{font_size}" font-weight="{font_weight}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="central">'
            f'{content}</text>')


def stage_box(x, y, w, h, title, lines, kind):
    """Stage rectangle with title + body lines, all centred."""
    fill   = C[f"{kind}_fill"]
    stroke = C[f"{kind}_stroke"]
    h_col  = C[f"{kind}_h_text"]
    s_col  = C[f"{kind}_s_text"]
    out = [svg_rect(x, y, w, h, rx=8, fill=fill, stroke=stroke)]
    cx = x + w // 2
    n = len(lines)
    if n == 0:
        out.append(svg_text(cx, y + h // 2, title, 13, 600, h_col))
    elif n == 1:
        out.append(svg_text(cx, y + h // 2 - 10, title, 13, 600, h_col))
        out.append(svg_text(cx, y + h // 2 + 10, lines[0], 11, 400, s_col))
    else:
        ty = y + h // 2 - 18
        out.append(svg_text(cx, ty, title, 13, 600, h_col))
        for i, ln in enumerate(lines):
            out.append(svg_text(cx, ty + 18 + i * 18, ln, 11, 400, s_col))
    return "\n  ".join(out)


def diamond(cx, cy, hw, hh, label, kind="prop"):
    pts = f"{cx},{cy-hh} {cx+hw},{cy} {cx},{cy+hh} {cx-hw},{cy}"
    r = (f'<polygon points="{pts}" '
         f'fill="{C[kind+"_fill"]}" stroke="{C[kind+"_stroke"]}" stroke-width="0.8"/>')
    t = svg_text(cx, cy, label, 13, 600, C[f"{kind}_h_text"])
    return r + "\n  " + t

# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD SVG
# ─────────────────────────────────────────────────────────────────────────────

def build_svg():
    p = []

    # ── defs: arrowheads ──────────────────────────────────────────────────────
    p.append(f"""  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="{C['arrow']}"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <marker id="arr-dash" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="{C['dash']}"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>""")

    # ── legend ────────────────────────────────────────────────────────────────
    lx = 426
    p.append(f"""  <!-- LEGEND -->
  {svg_rect(lx, 16, 12, 12, rx=2, fill=C['std_fill'], stroke=C['std_stroke'])}
  {svg_text(lx+16, 22, LEGEND_STD,        11, 400, C['body'], anchor="start")}
  {svg_rect(lx, 34, 12, 12, rx=2, fill=C['prop_fill'], stroke=C['prop_stroke'])}
  {svg_text(lx+16, 40, LEGEND_PROP[0],    11, 400, C['body'], anchor="start")}
  {svg_text(lx+16, 52, LEGEND_PROP[1],    11, 400, C['body'], anchor="start")}""")

    # ── input pill ────────────────────────────────────────────────────────────
    pw, ph = 200, INPUT_H
    px = CX - pw // 2
    p.append(f"""  <!-- INPUT -->
  {svg_rect(px, INPUT_Y, pw, ph, rx=19, fill=C['term_fill'], stroke=C['term_stroke'])}
  {svg_text(CX, INPUT_Y + ph//2, INPUT_LABEL, 13, 600, C['term_h_text'])}
  {svg_line(CX, INPUT_Y+ph, CX, S1_Y)}""")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    s = STAGES[1]
    p.append(f"""  <!-- STAGE 1 -->
  {stage_box(BOX_LEFT, S1_Y, BOX_WIDTH, S1_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S1_Y+33, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_path("M110 140 Q82 140 82 151 Q82 162 110 162")}
  {svg_text(114, 157, LOOP_ANNOTATION[0], 11, 400, C['label'], anchor="start")}
  {svg_line(CX, S1_Y+S1_H, CX, S2_Y)}""")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    s = STAGES[2]
    p.append(f"""  <!-- STAGE 2 -->
  {stage_box(BOX_LEFT, S2_Y, BOX_WIDTH, S2_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S2_Y+33, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_line(CX, S2_Y+S2_H, CX, S3_Y)}""")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    s = STAGES[3]
    d1_top = D1_CY - D_HH
    p.append(f"""  <!-- STAGE 3 -->
  {stage_box(BOX_LEFT, S3_Y, BOX_WIDTH, S3_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S3_Y+42, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_line(CX, S3_Y+S3_H, CX, d1_top)}""")

    # ── Diamond 1 ─────────────────────────────────────────────────────────────
    d1_bot   = D1_CY + D_HH
    d1_left  = CX - D_HW
    d1_right = CX + D_HW
    p.append(f"""  <!-- DIAMOND 1 -->
  {diamond(CX, D1_CY, D_HW, D_HH, DIAMOND_LABEL)}
  {svg_line(CX, d1_bot, CX, S4_Y)}
  {svg_text(CX+8, d1_bot+14, "yes", 11, 400, C['body'], anchor="start")}
  {svg_path(f"M{d1_left} {D1_CY} Q158 {D1_CY} 158 450 Q158 460 110 460")}
  {svg_text(108, 415, D1_LEFT_LABEL, 11, 400, C['label'], anchor="end")}
  {svg_path(f"M{d1_right} {D1_CY} Q454 {D1_CY} 454 288 Q454 {S3_Y} 512 {S3_Y}")}
  {svg_text(458, 314, D1_RIGHT_LABEL[0], 11, 400, C['label'], anchor="start")}
  {svg_text(458, 328, D1_RIGHT_LABEL[1], 11, 400, C['label'], anchor="start")}""")

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    s = STAGES[4]
    d2_top = D2_CY - D_HH
    p.append(f"""  <!-- STAGE 4 -->
  {stage_box(BOX_LEFT, S4_Y, BOX_WIDTH, S4_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S4_Y+39, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_line(CX, S4_Y+S4_H, CX, d2_top)}""")

    # ── Diamond 2 ─────────────────────────────────────────────────────────────
    d2_bot   = D2_CY + D_HH
    d2_left  = CX - D_HW
    d2_right = CX + D_HW
    p.append(f"""  <!-- DIAMOND 2 -->
  {diamond(CX, D2_CY, D_HW, D_HH, DIAMOND_LABEL)}
  {svg_line(CX, d2_bot, CX, S5A_Y)}
  {svg_text(CX+8, d2_bot+14, "yes", 11, 400, C['body'], anchor="start")}
  {svg_path(f"M{d2_left} {D2_CY} Q158 {D2_CY} 158 480 Q158 {S4_Y} 108 {S4_Y}")}
  {svg_text(128, 567, D2_LEFT_LABEL, 11, 400, C['label'], anchor="end")}
  {svg_path(f"M{d2_right} {D2_CY} Q460 {D2_CY} 460 480 Q460 {S4_Y} 512 {S4_Y}")}
  {svg_text(464, 506, D2_RIGHT_LABEL[0], 11, 400, C['label'], anchor="start")}
  {svg_text(464, 520, D2_RIGHT_LABEL[1], 11, 400, C['label'], anchor="start")}""")

    # ── Stage 5a ──────────────────────────────────────────────────────────────
    s = STAGES["5a"]
    p.append(f"""  <!-- STAGE 5a -->
  {stage_box(BOX_LEFT, S5A_Y, BOX_WIDTH, S5A_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S5A_Y+34, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_line(CX, S5A_Y+S5A_H, CX, S5B_Y)}""")

    # ── Stage 5b ──────────────────────────────────────────────────────────────
    s = STAGES["5b"]
    p.append(f"""  <!-- STAGE 5b -->
  {stage_box(BOX_LEFT, S5B_Y, BOX_WIDTH, S5B_H, s['title'], s['lines'], s['kind'])}
  {svg_text(96, S5B_Y+34, s['num'], 11, 400, C['label'], anchor="end")}
  {svg_line(CX, S5B_Y+S5B_H, CX, OUTPUT_Y)}""")

    # ── output pill ───────────────────────────────────────────────────────────
    ow, oh = 280, OUTPUT_H
    ox = CX - ow // 2
    p.append(f"""  <!-- OUTPUT -->
  {svg_rect(ox, OUTPUT_Y, ow, oh, rx=19, fill=C['term_fill'], stroke=C['term_stroke'])}
  {svg_text(CX, OUTPUT_Y + oh//2, OUTPUT_LABEL, 13, 600, C['term_h_text'])}""")

    body = "\n\n".join(p)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_WIDTH}" height="{VIEWBOX_H}" '
        f'viewBox="{VIEWBOX_X} {VIEWBOX_Y} {VIEWBOX_W} {VIEWBOX_H}">\n'
        f'  <title>NoC-aware DNN pruning and mapping pipeline</title>\n\n'
        f'{body}\n\n'
        f'</svg>\n'
    )

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPORT  –  PyMuPDF (fitz) renders SVG natively, no Cairo needed
# ─────────────────────────────────────────────────────────────────────────────

def export(svg_str, out_dir="."):
    svg_path = os.path.join(out_dir, "pipeline_overview.svg")
    png_path = os.path.join(out_dir, "pipeline_overview.png")
    pdf_path = os.path.join(out_dir, "pipeline_overview.pdf")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"SVG → {svg_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not found. Install with:  pip install pymupdf")
        return

    doc = fitz.open(svg_path)
    page = doc[0]

    # PNG at PNG_SCALE × native resolution
    mat = fitz.Matrix(PNG_SCALE, PNG_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_path)
    print(f"PNG → {png_path}  ({PNG_SCALE}x resolution)")

    # PDF
    pdf_bytes = doc.convert_to_pdf()
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF → {pdf_path}")

    doc.close()

# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    svg_str = build_svg()
    export(svg_str, out_dir=out_dir)
