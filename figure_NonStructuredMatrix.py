"""
build_bottom_diagram.py
=======================
Generates the weight-matrix comparison diagram (bottom figure):
  Dense (regular) | Sparse irregular (edge pruning) | Sparse structured (neuron pruning)
  with consequence boxes and GPU-incompatibility warning.

Outputs (written to OUT_DIR):
  pruning_bottom.svg
  pruning_bottom.png   (PNG_SCALE × native resolution)
  pruning_bottom.pdf

Dependencies:
  pip install pymupdf
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR   = "."
PNG_SCALE = 3


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    cell_filled  = "#1D9E75",   # filled matrix cell
    cell_empty   = "#ddd",      # empty cell outline
    cell_opacity = "0.7",
    removed_fill = "#E24B4A",   # highlight for removed row/col
    removed_op   = "0.18",
    removed_text = "#A32D2D",
    title        = "#333",
    subtitle     = "#555",
    arrow        = "#888",
    # Consequence box — amber (CSR / GPU-unfriendly)
    csr_bg       = "#FAEEDA",
    csr_stroke   = "#BA7517",
    csr_title    = "#633806",
    csr_body     = "#854F0B",
    # Consequence box — green (Dense sub-matrix / GPU-friendly)
    dm_bg        = "#EAF3DE",
    dm_stroke    = "#3B6D11",
    dm_title     = "#173404",
    dm_body      = "#3B6D11",
    # Warning banner
    warn_bg      = "#FCEBEB",
    warn_stroke  = "#A32D2D",
    warn_title   = "#501313",
    warn_body    = "#791F1F",
)

# Cell geometry
CELL = 14    # cell size (px)
GAP  =  4    # gap between cells (stride = CELL + GAP = 18)
STRIDE = CELL + GAP


# ─────────────────────────────────────────────────────────────────────────────
# SVG helpers
# ─────────────────────────────────────────────────────────────────────────────
def rect(x, y, w, h, rx=2, fill="#1D9E75", stroke="none", sw=0.5, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

def text(x, y, content, size=12, weight=400, fill="#333", anchor="middle"):
    return (f'<text font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'x="{x}" y="{y}" text-anchor="{anchor}">{content}</text>')

def line(x1, y1, x2, y2, stroke="#888", sw=1.0, dasharray="", extra=""):
    dash = f'stroke-dasharray="{dasharray}"' if dasharray else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" {dash} {extra}/>')

def arrow_marker():
    return (
        '<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M2 1L8 5L2 9" fill="none" stroke="#888" '
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</marker>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Matrix drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_matrix(ox, oy, rows, cols, filled_set, label,
                removed_row=None, removed_col=None):
    """
    Draw a rows×cols matrix at (ox, oy).
    filled_set : set of (r, c) indices that are non-zero.
    removed_row / removed_col : index of highlighted row/col (structured sparsity).
    Returns a list of SVG strings.
    """
    parts = []

    # Column label
    w_total = cols * STRIDE - GAP
    cx = ox + w_total // 2
    parts.append(text(cx, oy - 6, label, size=12, fill=C["subtitle"]))

    # Draw empty outlines first, then filled cells on top
    empties, fills = [], []
    for r in range(rows):
        for c in range(cols):
            x = ox + c * STRIDE
            y = oy + r * STRIDE
            if (r, c) in filled_set:
                fills.append(rect(x, y, CELL, CELL, rx=2,
                                  fill=C["cell_filled"], stroke="none"))
            else:
                empties.append(rect(x, y, CELL, CELL, rx=2,
                                    fill="none", stroke=C["cell_empty"]))

    # Wrap filled cells in opacity group
    parts += empties
    if fills:
        inner = "\n    ".join(fills)
        parts.append(
            f'<g fill="{C["cell_filled"]}" opacity="{C["cell_opacity"]}">\n'
            f'    {inner}\n  </g>'
        )

    # Highlight removed row
    if removed_row is not None:
        y = oy + removed_row * STRIDE
        parts.append(rect(ox, y, w_total, CELL, rx=2,
                          fill=C["removed_fill"], stroke="none",
                          extra=f'opacity="{C["removed_op"]}"'))

    # Highlight removed col
    if removed_col is not None:
        x  = ox + removed_col * STRIDE
        h  = rows * STRIDE - GAP
        parts.append(rect(x, oy, CELL, h, rx=2,
                          fill=C["removed_fill"], stroke="none",
                          extra=f'opacity="{C["removed_op"]}"'))

    # "removed row / col" annotation
    if removed_row is not None or removed_col is not None:
        parts.append(text(cx, oy + rows * STRIDE + 4,
                          "removed row / col", size=10, fill=C["removed_text"]))

    return parts


def consequence_box(cx, y, w, h, title, subtitle, bg, stroke, tc, sc):
    """Rounded rectangle box with two lines of text."""
    x = cx - w // 2
    return [
        rect(x, y, w, h, rx=8, fill=bg, stroke=stroke, sw=0.8),
        text(cx, y + 20, title,    size=13, weight=600, fill=tc),
        text(cx, y + 36, subtitle, size=11, fill=sc),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Build SVG
# ─────────────────────────────────────────────────────────────────────────────
def build_svg():
    W, H = 680, 310

    # ── Matrix layout ─────────────────────────────────────────────────────────
    # Matrix top-left x origins; all start at y=50
    MAT_Y    = 50
    OX_DENSE = 24        # matrix 1 (5×5 dense)
    OX_SPARSE= 240       # matrix 2 (5×5 sparse irregular)
    OX_STRUC = 520       # matrix 3 (4×4 structured sparse)

    # Dense: all cells filled
    dense_filled = {(r, c) for r in range(5) for c in range(5)}

    # Sparse irregular: 9 non-zero cells scattered
    sparse_filled = {(0,1),(0,4),(1,0),(1,3),(2,2),(2,4),(3,0),(3,1),(4,3)}

    # Structured sparse (4×4): row 2 and col 3 are removed (zero)
    struc_filled = {(0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(3,2),(3,3)}
    REMOVED_ROW, REMOVED_COL = 2, 3

    # Matrix bottom y (for arrow origins)
    DENSE_BOT  = MAT_Y + 5 * STRIDE   # 5 rows
    SPARSE_BOT = MAT_Y + 5 * STRIDE
    STRUC_BOT  = MAT_Y + 4 * STRIDE   # 4 rows

    # Centre x of each matrix
    cx_dense  = OX_DENSE  + (5 * STRIDE - GAP) // 2
    cx_sparse = OX_SPARSE + (5 * STRIDE - GAP) // 2
    cx_struc  = OX_STRUC  + (4 * STRIDE - GAP) // 2

    # ── Consequence box positions ─────────────────────────────────────────────
    BOX_Y  = 165
    BOX_H  = 44
    BOX_W  = 180

    # ── Warning banner ────────────────────────────────────────────────────────
    WARN_Y = 228
    WARN_H =  62

    parts = []

    # ── Defs ─────────────────────────────────────────────────────────────────
    parts.append(f'<defs>\n  {arrow_marker()}\n</defs>')

    # ── Title ─────────────────────────────────────────────────────────────────
    parts.append(text(W // 2, 22, "Weight matrix comparison",
                      size=14, weight=600, fill=C["title"]))

    # ── Matrices ──────────────────────────────────────────────────────────────
    parts += draw_matrix(OX_DENSE,  MAT_Y, 5, 5, dense_filled,
                         "Dense (regular)")
    parts += draw_matrix(OX_SPARSE, MAT_Y, 5, 5, sparse_filled,
                         "Sparse, irregular (edge pruning)")
    parts += draw_matrix(OX_STRUC,  MAT_Y, 4, 4, struc_filled,
                         "Sparse, structured (neuron pruning)",
                         removed_row=REMOVED_ROW, removed_col=REMOVED_COL)

    # ── Arrows from sparse matrices down to consequence boxes ─────────────────
    arr_top_sparse = SPARSE_BOT + 2
    arr_top_struc  = STRUC_BOT  + 2
    parts += [
        line(cx_sparse, arr_top_sparse, cx_sparse, BOX_Y - 2,
             stroke=C["arrow"], dasharray="4 3",
             extra='marker-end="url(#arr)"'),
        line(cx_struc,  arr_top_struc,  cx_struc,  BOX_Y - 2,
             stroke=C["arrow"], dasharray="4 3",
             extra='marker-end="url(#arr)"'),
    ]

    # ── Consequence boxes ─────────────────────────────────────────────────────
    parts += consequence_box(
        cx_sparse, BOX_Y, BOX_W, BOX_H,
        "Compressed storage", "CSR/COO — GPU unfriendly",
        C["csr_bg"], C["csr_stroke"], C["csr_title"], C["csr_body"],
    )
    parts += consequence_box(
        cx_struc, BOX_Y, BOX_W, BOX_H,
        "Dense sub-matrix", "GPU/CPU compatible",
        C["dm_bg"], C["dm_stroke"], C["dm_title"], C["dm_body"],
    )

    # ── Warning banner ────────────────────────────────────────────────────────
    parts += [
        rect(20, WARN_Y, W - 40, WARN_H, rx=8,
             fill=C["warn_bg"], stroke=C["warn_stroke"], sw=0.8),
        text(W // 2, WARN_Y + 24, "Non-structured pruning downside",
             size=13, weight=600, fill=C["warn_title"]),
        text(W // 2, WARN_Y + 44,
             "Irregular sparse matrices break data-parallel GPU/CPU execution",
             size=11, fill=C["warn_body"]),
    ]

    body = "\n  ".join(parts)
    return (
        f'<svg width="{W}" viewBox="0 0 {W} {H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:white; font-family: Arial, sans-serif;">\n'
        f'  {body}\n'
        f'</svg>\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: SVG + PNG + PDF via PyMuPDF
# ─────────────────────────────────────────────────────────────────────────────
def export(svg_str, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, "pruning_bottom")
    svg_path = base + ".svg"
    png_path = base + ".png"
    pdf_path = base + ".pdf"

    # Write SVG
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"SVG → {svg_path}")

    # PNG + PDF via PyMuPDF
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not found.  Install with:  pip install pymupdf")
        return

    doc  = fitz.open(svg_path)
    page = doc[0]

    mat = fitz.Matrix(PNG_SCALE, PNG_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_path)
    print(f"PNG → {png_path}  ({PNG_SCALE}× resolution, {pix.width}×{pix.height} px)")

    pdf_bytes = doc.convert_to_pdf()
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF → {pdf_path}")

    doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    svg = build_svg()
    export(svg)
