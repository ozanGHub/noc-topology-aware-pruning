"""
build_top_diagram.py
====================
Generates the non-structured pruning network diagram (top figure):
  Before pruning → After edge pruning → After neuron pruning

Outputs (written to OUT_DIR):
  pruning_top.svg
  pruning_top.png   (PNG_SCALE × native resolution)
  pruning_top.pdf

Dependencies:
  pip install pymupdf
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR   = "."          # output directory
PNG_SCALE = 3            # rasterisation multiplier (3 → ~2040 px wide)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (shared across both diagram scripts)
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    node_fill   = "#5DCAA5",   # light teal  — active neuron
    node_stroke = "#0F6E56",   # dark teal
    edge_dense  = "#1D9E75",   # dense connection
    edge_sparse = "#1D9E75",   # sparse connection (same hue, lighter opacity via SVG)
    title       = "#0F6E56",
    subtitle    = "#555",
    arrow_edge  = "#534AB7",   # purple  — prune-edges arrow
    arrow_neuron= "#BA7517",   # amber   — prune-neurons arrow
    ghost_stroke= "#ccc",      # removed neuron outline
    ghost_x     = "#A32D2D",   # X mark on removed neuron
)


# ─────────────────────────────────────────────────────────────────────────────
# SVG helpers
# ─────────────────────────────────────────────────────────────────────────────
def circle(cx, cy, r, fill, stroke, sw=0.8, extra=""):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

def line(x1, y1, x2, y2, stroke, sw=0.7, opacity=1.0, dasharray="", extra=""):
    dash = f'stroke-dasharray="{dasharray}"' if dasharray else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}" {dash} {extra}/>')

def text(x, y, content, size=12, weight=400, fill="#333", anchor="middle"):
    return (f'<text font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'x="{x}" y="{y}" text-anchor="{anchor}">{content}</text>')

def arrow_marker(marker_id, stroke_color):
    return (
        f'<marker id="{marker_id}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M2 1L8 5L2 9" fill="none" stroke="{stroke_color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</marker>'
    )

def group(items, stroke="", fill="none", sw=0.7, opacity=1.0, extra=""):
    """Wrap a list of SVG strings in a <g> element."""
    attrs = []
    if stroke: attrs.append(f'stroke="{stroke}"')
    if fill:   attrs.append(f'fill="{fill}"')
    if sw != 0.7: attrs.append(f'stroke-width="{sw}"')
    if opacity != 1.0: attrs.append(f'opacity="{opacity}"')
    if extra:  attrs.append(extra)
    inner = "\n    ".join(items)
    return f'<g {" ".join(attrs)}>\n    {inner}\n  </g>'


# ─────────────────────────────────────────────────────────────────────────────
# Network-drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_nodes(centres, fill, stroke, r=12, sw=0.8):
    return [circle(cx, cy, r, fill, stroke, sw) for cx, cy in centres]

def draw_connections(pairs, stroke, sw=0.7, opacity=0.45):
    return [line(x1, y1, x2, y2, stroke, sw, opacity) for x1, y1, x2, y2 in pairs]

def draw_ghost(cx, cy, r=12, arm=10):
    """Dashed circle with an × mark — represents a pruned neuron."""
    return [
        circle(cx, cy, r, "none", C["ghost_stroke"], sw=1.0,
               extra='stroke-dasharray="4 3"'),
        line(cx - arm, cy - arm, cx + arm, cy + arm,
             C["ghost_x"], sw=1.2, opacity=0.7),
        line(cx + arm, cy - arm, cx - arm, cy + arm,
             C["ghost_x"], sw=1.2, opacity=0.7),
    ]

def draw_pruning_arrow(x1, x2, y, label1, label2, color, marker_id):
    """Horizontal arrow with two-line label above it."""
    return [
        text((x1 + x2) // 2, y - 22, label1, size=11, fill=color),
        text((x1 + x2) // 2, y - 10, label2, size=11, fill=color),
        line(x1, y, x2, y, color, sw=2,
             extra=f'marker-end="url(#{marker_id})"'),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Stage definitions
# ─────────────────────────────────────────────────────────────────────────────
# Layer y-coordinates (shared across all three stages)
Y1, Y2, Y3 = 72, 148, 224

def stage1_dense(ox):
    """4-4-3 fully-connected network."""
    l1 = [(ox+0,  Y1), (ox+32, Y1), (ox+64, Y1), (ox+96, Y1)]
    l2 = [(ox+0,  Y2), (ox+32, Y2), (ox+64, Y2), (ox+96, Y2)]
    l3 = [(ox+16, Y3), (ox+48, Y3), (ox+80, Y3)]

    conns12 = [(ax, Y1+12, bx, Y2-12) for ax,_ in l1 for bx,_ in l2]
    conns23 = [(ax, Y2+12, bx, Y3-12) for ax,_ in l2 for bx,_ in l3]

    parts = []
    parts += draw_connections(conns12, C["edge_dense"], sw=0.7, opacity=0.45)
    parts += draw_connections(conns23, C["edge_dense"], sw=0.7, opacity=0.45)
    parts += draw_nodes(l1 + l2 + l3, C["node_fill"], C["node_stroke"])
    return parts

def stage2_sparse(ox):
    """4-4-3 network with sparse irregular connections."""
    l1 = [(ox+0,  Y1), (ox+32, Y1), (ox+64, Y1), (ox+96, Y1)]
    l2 = [(ox+0,  Y2), (ox+32, Y2), (ox+64, Y2), (ox+96, Y2)]
    l3 = [(ox+16, Y3), (ox+48, Y3), (ox+80, Y3)]

    # Irregular subset of connections
    sparse12 = [
        (l1[0][0], Y1+12, l2[1][0], Y2-12),
        (l1[0][0], Y1+12, l2[3][0], Y2-12),
        (l1[1][0], Y1+12, l2[0][0], Y2-12),
        (l1[2][0], Y1+12, l2[2][0], Y2-12),
        (l1[2][0], Y1+12, l2[3][0], Y2-12),
        (l1[3][0], Y1+12, l2[0][0], Y2-12),
        (l1[3][0], Y1+12, l2[2][0], Y2-12),
    ]
    sparse23 = [
        (l2[0][0], Y2+12, l3[1][0], Y3-12),
        (l2[1][0], Y2+12, l3[0][0], Y3-12),
        (l2[1][0], Y2+12, l3[2][0], Y3-12),
        (l2[2][0], Y2+12, l3[2][0], Y3-12),
        (l2[3][0], Y2+12, l3[0][0], Y3-12),
    ]

    parts = []
    parts += draw_connections(sparse12, C["edge_sparse"], sw=0.7, opacity=0.65)
    parts += draw_connections(sparse23, C["edge_sparse"], sw=0.7, opacity=0.65)
    parts += draw_nodes(l1 + l2 + l3, C["node_fill"], C["node_stroke"])
    return parts

def stage3_pruned(ox):
    """3-3-2 network (1 neuron removed per layer) with sparse connections."""
    # Active nodes
    l1_active = [(ox+0, Y1), (ox+34, Y1), (ox+68, Y1)]
    l2_active = [(ox+0, Y2), (ox+34, Y2), (ox+102, Y2)]
    l3_active = [(ox+17, Y3), (ox+85, Y3)]

    # Ghost (pruned) node positions
    ghost1 = (ox+102, Y1)
    ghost2 = (ox+68,  Y2)
    ghost3 = (ox+51,  Y3)

    sparse12 = [
        (l1_active[0][0], Y1+12, l2_active[1][0], Y2-12),
        (l1_active[0][0], Y1+12, l2_active[2][0], Y2-12),
        (l1_active[1][0], Y1+12, l2_active[0][0], Y2-12),
        (l1_active[2][0], Y1+12, l2_active[2][0], Y2-12),
    ]
    sparse23 = [
        (l2_active[0][0], Y2+12, l3_active[0][0], Y3-12),
        (l2_active[1][0], Y2+12, l3_active[1][0], Y3-12),
        (l2_active[2][0], Y2+12, l3_active[0][0], Y3-12),
    ]

    parts = []
    parts += draw_connections(sparse12, C["edge_sparse"], sw=0.7, opacity=0.65)
    parts += draw_connections(sparse23, C["edge_sparse"], sw=0.7, opacity=0.65)
    parts += draw_nodes(l1_active + l2_active + l3_active,
                        C["node_fill"], C["node_stroke"])
    parts += draw_ghost(*ghost1)
    parts += draw_ghost(*ghost2)
    parts += draw_ghost(*ghost3)
    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Build SVG
# ─────────────────────────────────────────────────────────────────────────────
def build_svg():
    W, H = 680, 260

    # Stage x offsets (left edge of each 4-node row)
    OX1, OX2, OX3 = 36, 268, 490

    # Arrow x positions
    ARR1_X1, ARR1_X2 = 158, 208   # between stage 1 and 2
    ARR2_X1, ARR2_X2 = 386, 436   # between stage 2 and 3
    ARR_Y = 148                    # vertical midpoint

    parts = []

    # ── Markers ──────────────────────────────────────────────────────────────
    defs = (
        "<defs>\n  "
        + arrow_marker("arr_edge",   C["arrow_edge"])
        + "\n  "
        + arrow_marker("arr_neuron", C["arrow_neuron"])
        + "\n</defs>"
    )

    # ── Stage titles ──────────────────────────────────────────────────────────
    cx1 = OX1 + 48   # centre of 4-node row: 0+32+64+96 / 4 + OX1 = 48+OX1
    cx2 = OX2 + 48
    cx3 = OX3 + 51   # 3-node pruned: centre ~51
    parts += [
        text(cx1, 20, "Before pruning",          size=14, weight=600, fill=C["title"]),
        text(cx1, 36, "dense network",            size=12, fill=C["subtitle"]),
        text(cx2, 20, "After edge pruning",       size=14, weight=600, fill=C["title"]),
        text(cx2, 36, "sparse weights, all neurons remain", size=12, fill=C["subtitle"]),
        text(cx3, 20, "After neuron pruning",     size=14, weight=600, fill=C["title"]),
        text(cx3, 36, "fewer nodes, still irregular edges", size=12, fill=C["subtitle"]),
    ]

    # ── Network stages ────────────────────────────────────────────────────────
    parts += stage1_dense(OX1)
    parts += stage2_sparse(OX2)
    parts += stage3_pruned(OX3)

    # ── Arrows ────────────────────────────────────────────────────────────────
    parts += draw_pruning_arrow(ARR1_X1, ARR1_X2, ARR_Y,
                                "prune", "edges", C["arrow_edge"], "arr_edge")
    parts += draw_pruning_arrow(ARR2_X1, ARR2_X2, ARR_Y,
                                "prune", "neurons", C["arrow_neuron"], "arr_neuron")

    body = "\n  ".join(parts)
    return (
        f'<svg width="{W}" viewBox="0 0 {W} {H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:white; font-family: Arial, sans-serif;">\n'
        f'  {defs}\n\n'
        f'  {body}\n'
        f'</svg>\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: SVG + PNG + PDF via PyMuPDF
# ─────────────────────────────────────────────────────────────────────────────
def export(svg_str, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, "pruning_top")
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

    # PNG – rasterise at PNG_SCALE × native resolution
    mat = fitz.Matrix(PNG_SCALE, PNG_SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(png_path)
    print(f"PNG → {png_path}  ({PNG_SCALE}× resolution, {pix.width}×{pix.height} px)")

    # PDF – convert the SVG document to PDF
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
