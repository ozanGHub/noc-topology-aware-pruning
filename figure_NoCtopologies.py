"""
build_noc_topologies.py
=======================
Generates the NoC Topologies diagram showing:
  2-D Mesh | Torus | Condensed Mesh  (left to right)
with legend and captions.

Outputs (written to OUT_DIR):
  noc_topologies.svg
  noc_topologies.png   (PNG_SCALE × native resolution)
  noc_topologies.pdf

Dependencies:
  pip install pymupdf
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR   = "."
PNG_SCALE = 3          # 3 → 2040 px wide output


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C = dict(
    router_fill    = "#1D9E75",   # standard router node fill
    router_stroke  = "#085041",   # standard router node stroke
    switch_fill    = "#085041",   # condensed switch fill (darker)
    switch_stroke  = "#04342C",   # condensed switch stroke
    pe_fill        = "#5DCAA5",   # processing element fill (light teal)
    pe_stroke      = "#0F6E56",   # processing element stroke
    edge           = "#0F6E56",   # regular mesh link
    torus_wrap     = "#085041",   # torus wrap-around link
    title          = "#085041",   # section title text
    legend_title   = "#333",
    legend_text    = "#444",
    caption        = "#666",
    caption2       = "#999",
    separator      = "#cccccc",
)

# Node geometry
ROUTER_W  = 14    # router/switch node width & height
SWITCH_W  = 18    # condensed switch is slightly larger
PE_W      = 10    # processing element size
SPACING   = 46    # grid spacing between router centres


# ─────────────────────────────────────────────────────────────────────────────
# SVG primitive helpers  (all return strings)
# ─────────────────────────────────────────────────────────────────────────────
def R(x, y, w, h, rx=2, fill="#1D9E75", stroke="none", sw=0.8, extra=""):
    """Rectangle."""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

def L(x1, y1, x2, y2, stroke="#0F6E56", sw=0.8, dash="", extra=""):
    """Line."""
    d = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}" {d} {extra}/>')

def T(x, y, txt, size=11, weight=400, fill="#444", anchor="middle"):
    """Text."""
    return (f'<text font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'x="{x}" y="{y}" text-anchor="{anchor}">{txt}</text>')

def G(items, stroke="", fill="none", sw=0.8, opacity=1.0, extra=""):
    """Group wrapper."""
    attrs = []
    if stroke:      attrs.append(f'stroke="{stroke}"')
    if fill:        attrs.append(f'fill="{fill}"')
    if sw != 0.8:   attrs.append(f'stroke-width="{sw}"')
    if opacity < 1: attrs.append(f'opacity="{opacity}"')
    if extra:       attrs.append(extra)
    inner = "\n  ".join(items)
    return f'<g {" ".join(attrs)}>\n  {inner}\n</g>'

def path(d, stroke, sw=0.8, fill="none", dash=""):
    dattr = f'stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" {dattr}/>')


# ─────────────────────────────────────────────────────────────────────────────
# Mesh drawing helpers
# ─────────────────────────────────────────────────────────────────────────────
def router_rect(cx, cy):
    """14×14 router node centred at (cx, cy)."""
    h = ROUTER_W // 2
    return R(cx - h, cy - h, ROUTER_W, ROUTER_W, rx=2,
             fill=C["router_fill"], stroke=C["router_stroke"])

def switch_rect(cx, cy):
    """18×18 condensed switch centred at (cx, cy)."""
    h = SWITCH_W // 2
    return R(cx - h, cy - h, SWITCH_W, SWITCH_W, rx=3,
             fill=C["switch_fill"], stroke=C["switch_stroke"])

def pe_rect(x, y):
    """10×10 PE with top-left corner at (x, y)."""
    return R(x, y, PE_W, PE_W, rx=1,
             fill=C["pe_fill"], stroke=C["pe_stroke"], sw=0.7)

def mesh_router_centres(ox, oy, rows=4, cols=4, spacing=SPACING):
    """Return list of (cx, cy) for a rows×cols grid, top-left router at (ox,oy)."""
    return [(ox + c * spacing, oy + r * spacing)
            for r in range(rows) for c in range(cols)]

def draw_mesh_edges(centres, rows=4, cols=4):
    """Horizontal + vertical edges between adjacent routers (gap between node borders)."""
    gap = ROUTER_W // 2 + 1   # small gap so line starts outside node
    lines = []
    for r in range(rows):
        for c in range(cols):
            cx, cy = centres[r * cols + c]
            # horizontal →
            if c < cols - 1:
                nx, ny = centres[r * cols + c + 1]
                lines.append(L(cx + gap, cy, nx - gap, ny,
                               stroke=C["edge"], sw=0.8))
            # vertical ↓
            if r < rows - 1:
                nx, ny = centres[(r + 1) * cols + c]
                lines.append(L(cx, cy + gap, nx, ny - gap,
                               stroke=C["edge"], sw=0.8))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# 1. 2-D MESH
# ─────────────────────────────────────────────────────────────────────────────
def draw_2d_mesh():
    """
    4×4 router grid. Routers at col*46+42, row*46+50.
    PEs: row-0 above, row-3 below, col-0 left, col-3 right.
    Interior (row1-2, col1-2) PEs placed at 45° diagonal outward.
    """
    OX, OY = 42, 50
    centres = mesh_router_centres(OX, OY)

    parts = []

    # ── edges ────────────────────────────────────────────────────────────────
    parts += draw_mesh_edges(centres)

    # ── routers ──────────────────────────────────────────────────────────────
    for cx, cy in centres:
        parts.append(router_rect(cx, cy))

    # ── PEs and their stub links ──────────────────────────────────────────────
    # row 0: PE above each router
    for c in range(4):
        cx, cy = centres[c]
        parts.append(pe_rect(cx - 5, cy - 31))           # PE
        parts.append(L(cx, cy - 7, cx, cy - 21,
                       stroke=C["pe_stroke"], sw=0.6))    # stub

    # row 3: PE below each router
    for c in range(4):
        cx, cy = centres[12 + c]
        parts.append(pe_rect(cx - 5, cy + 21))
        parts.append(L(cx, cy + 7, cx, cy + 21,
                       stroke=C["pe_stroke"], sw=0.6))

    # col 0: PE left of row-1 and row-2 routers
    for r in [1, 2]:
        cx, cy = centres[r * 4]
        parts.append(pe_rect(cx - 24, cy - 5))
        parts.append(L(cx - 7, cy, cx - 14, cy,
                       stroke=C["pe_stroke"], sw=0.6))

    # col 3: PE right of row-1 and row-2 routers
    for r in [1, 2]:
        cx, cy = centres[r * 4 + 3]
        parts.append(pe_rect(cx + 15, cy - 5))
        parts.append(L(cx + 7, cy, cx + 15, cy,
                       stroke=C["pe_stroke"], sw=0.6))

    # Interior (row1,col1), (row1,col2), (row2,col1), (row2,col2)
    # 45° outward directions: NW, NE, SW, SE
    interior = [
        (1, 1, -14, -14),   # NW
        (1, 2, +14, -14),   # NE
        (2, 1, -14, +14),   # SW
        (2, 2, +14, +14),   # SE
    ]
    for r, c, dx, dy in interior:
        cx, cy = centres[r * 4 + c]
        px = cx + dx - 5
        py = cy + dy - 5
        parts.append(pe_rect(px, py))
        # stub from router corner toward PE centre
        sx = cx + (7 if dx > 0 else -7)
        sy = cy + (7 if dy > 0 else -7)
        parts.append(L(sx, sy, cx + dx, cy + dy,
                       stroke=C["pe_stroke"], sw=0.6))

    return parts


# ─────────────────────────────────────────────────────────────────────────────
# 2. TORUS  (same grid + wrap-around arcs, offset x+230)
# ─────────────────────────────────────────────────────────────────────────────
def draw_torus():
    """
    Identical 4×4 grid to 2-D Mesh but shifted right by 230px,
    plus dashed wrap-around arcs on all four sides.
    """
    OX, OY = 272, 50
    centres = mesh_router_centres(OX, OY)

    parts = []

    # ── wrap-around arcs (dashed) ─────────────────────────────────────────────
    # Vertical wraps: top↔bottom for each column
    col_xs = [OX + c * SPACING for c in range(4)]
    top_y, bot_y = OY, OY + 3 * SPACING
    h7 = ROUTER_W // 2

    # outer columns (0 and 3) use wide arcs; inner columns use narrow arcs
    wrap_arcs = [
        # col 0: wide arc to the left
        f"M {col_xs[0]},{top_y - h7} C {col_xs[0]-20},{top_y-40} "
        f"{col_xs[0]-20},{bot_y+40} {col_xs[0]},{bot_y + h7}",
        # col 1: narrow arc
        f"M {col_xs[1]},{top_y - h7} C {col_xs[1]-5},{top_y-18} "
        f"{col_xs[1]-5},{bot_y+18} {col_xs[1]},{bot_y + h7}",
        # col 2: narrow arc (right side)
        f"M {col_xs[2]},{top_y - h7} C {col_xs[2]+5},{top_y-18} "
        f"{col_xs[2]+5},{bot_y+18} {col_xs[2]},{bot_y + h7}",
        # col 3: wide arc to the right
        f"M {col_xs[3]},{top_y - h7} C {col_xs[3]+20},{top_y-40} "
        f"{col_xs[3]+20},{bot_y+40} {col_xs[3]},{bot_y + h7}",
    ]
    for d in wrap_arcs:
        parts.append(path(d, C["torus_wrap"], sw=0.8, dash="3 2"))

    # Horizontal wraps: left↔right for each row
    row_ys = [OY + r * SPACING for r in range(4)]
    left_x, right_x = OX, OX + 3 * SPACING
    wrap_h_arcs = [
        # row 0: arc above
        f"M {left_x - h7},{row_ys[0]} C {left_x-40},{row_ys[0]-18} "
        f"{right_x+40},{row_ys[0]-18} {right_x + h7},{row_ys[0]}",
        # row 1: arc slightly above
        f"M {left_x - h7},{row_ys[1]} C {left_x-18},{row_ys[1]-6} "
        f"{right_x+18},{row_ys[1]-6} {right_x + h7},{row_ys[1]}",
        # row 2: arc slightly below
        f"M {left_x - h7},{row_ys[2]} C {left_x-18},{row_ys[2]+6} "
        f"{right_x+18},{row_ys[2]+6} {right_x + h7},{row_ys[2]}",
        # row 3: arc below
        f"M {left_x - h7},{row_ys[3]} C {left_x-40},{row_ys[3]+18} "
        f"{right_x+40},{row_ys[3]+18} {right_x + h7},{row_ys[3]}",
    ]
    for d in wrap_h_arcs:
        parts.append(path(d, C["torus_wrap"], sw=0.8, dash="3 2"))

    # ── regular mesh edges ────────────────────────────────────────────────────
    parts += draw_mesh_edges(centres)

    # ── routers ──────────────────────────────────────────────────────────────
    for cx, cy in centres:
        parts.append(router_rect(cx, cy))

    # ── PEs (same layout as 2-D mesh, offset by 230) ─────────────────────────
    # row 0: above
    for c in range(4):
        cx, cy = centres[c]
        parts.append(pe_rect(cx - 5, cy - 31))
        parts.append(L(cx, cy - 7, cx, cy - 21, stroke=C["pe_stroke"], sw=0.6))
    # row 3: below
    for c in range(4):
        cx, cy = centres[12 + c]
        parts.append(pe_rect(cx - 5, cy + 21))
        parts.append(L(cx, cy + 7, cx, cy + 21, stroke=C["pe_stroke"], sw=0.6))
    # col 0: left
    for r in [1, 2]:
        cx, cy = centres[r * 4]
        parts.append(pe_rect(cx - 24, cy - 5))
        parts.append(L(cx - 7, cy, cx - 14, cy, stroke=C["pe_stroke"], sw=0.6))
    # col 3: right
    for r in [1, 2]:
        cx, cy = centres[r * 4 + 3]
        parts.append(pe_rect(cx + 15, cy - 5))
        parts.append(L(cx + 7, cy, cx + 15, cy, stroke=C["pe_stroke"], sw=0.6))
    # interior: 45° diagonal
    interior = [(1,1,-14,-14),(1,2,+14,-14),(2,1,-14,+14),(2,2,+14,+14)]
    for r, c, dx, dy in interior:
        cx, cy = centres[r * 4 + c]
        parts.append(pe_rect(cx + dx - 5, cy + dy - 5))
        sx = cx + (7 if dx > 0 else -7)
        sy = cy + (7 if dy > 0 else -7)
        parts.append(L(sx, sy, cx + dx, cy + dy,
                       stroke=C["pe_stroke"], sw=0.6))

    return parts


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONDENSED MESH  (offset x+460)
# ─────────────────────────────────────────────────────────────────────────────
def draw_condensed_mesh():
    """
    8 condensed switch nodes: 4 corners + 4 edge-midpoints.
    Inter-switch links on perimeter + two diagonal cross-links.
    PEs placed outside the structure at varied angles.
    """
    # Switch centres
    TL = (502, 50);  TM = (548, 50);  TR = (594, 50)
    ML = (502, 119);                   MR = (594, 119)
    BL = (502, 188); BM = (548, 188); BR = (594, 188)

    h = SWITCH_W // 2
    parts = []

    # ── inter-switch links ────────────────────────────────────────────────────
    sw_links = [
        # top row
        (TL[0]+h, TL[1], TM[0]-h, TM[1]),
        (TM[0]+h, TM[1], TR[0]-h, TR[1]),
        # bottom row
        (BL[0]+h, BL[1], BM[0]-h, BM[1]),
        (BM[0]+h, BM[1], BR[0]-h, BR[1]),
        # left col
        (TL[0], TL[1]+h, ML[0], ML[1]-h),
        (ML[0], ML[1]+h, BL[0], BL[1]-h),
        # right col
        (TR[0], TR[1]+h, MR[0], MR[1]-h),
        (MR[0], MR[1]+h, BR[0], BR[1]-h),
        # diagonals TL↔BR and TR↔BL
        (TL[0]+h, TL[1]+h, BR[0]-h, BR[1]-h),
        (TR[0]-h, TR[1]+h, BL[0]+h, BL[1]-h),
    ]
    for x1, y1, x2, y2 in sw_links:
        parts.append(L(x1, y1, x2, y2, stroke=C["edge"], sw=1.2))

    # ── switch nodes ──────────────────────────────────────────────────────────
    for cx, cy in [TL, TM, TR, ML, MR, BL, BM, BR]:
        parts.append(switch_rect(cx, cy))

    # ── PEs and stubs ─────────────────────────────────────────────────────────
    # All (pe_x, pe_y, stub_x1, stub_y1, stub_x2, stub_y2) tuples
    pe_data = [
        # TL: two PEs upper-left
        (478, 24,  TL[0]-h, TL[1]-3, 483, 29),
        (464, 38,  TL[0]-h, TL[1]+5, 469, 43),
        # TM: one PE above
        (543, 24,  TM[0], TM[1]-h, TM[0], 34),
        # TR: two PEs upper-right
        (614, 24,  TR[0]+h, TR[1]-3, 619, 29),
        (628, 38,  TR[0]+h, TR[1]+5, 628, 43),
        # ML: one PE left
        (476, 114, ML[0]-h, ML[1], 486, 119),
        # MR: one PE right
        (610, 114, MR[0]+h, MR[1], 610, 119),
        # BL: two PEs lower-left
        (464, 160, BL[0]-h, BL[1]-5, 469, 165),
        (476, 196, BL[0]-h, BL[1]+6, 481, 201),
        # BM: one PE below
        (543, 208, BM[0], BM[1]+h, BM[0], 208),
        # BR: two PEs lower-right
        (628, 160, BR[0]+h, BR[1]-5, 628, 165),
        (626, 196, BR[0]+h, BR[1]+6, 621, 201),
        # Interior PEs: 30° outward from each corner
        (485, 19,  TL[0]-h, TL[1]-h+1, 490, 29),   # TL outward-up
        (601, 19,  TR[0]+h, TR[1]-h+1, 606, 29),    # TR outward-up
        (485, 209, BL[0]-h, BL[1]+h-1, 490, 209),   # BL outward-down
        (601, 209, BR[0]+h, BR[1]+h-1, 606, 209),   # BR outward-down
    ]
    for px, py, sx1, sy1, sx2, sy2 in pe_data:
        parts.append(pe_rect(px, py))
        parts.append(L(sx1, sy1, sx2, sy2, stroke=C["pe_stroke"], sw=0.6))

    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
def draw_legend():
    parts = []
    SEP_Y = 240
    parts.append(L(30, SEP_Y, 240, SEP_Y, stroke=C["separator"], sw=0.5))

    # Legend title
    parts.append(T(340, 256, "Legend", size=11, weight=600, fill=C["legend_title"]))

    # Row 1: Router node | Condensed switch | Processing element
    parts.append(R(110, 264, 14, 14, rx=2,
                   fill=C["router_fill"], stroke=C["router_stroke"]))
    parts.append(T(130, 275, "Router node", size=11, fill=C["legend_text"],
                   anchor="start"))

    parts.append(R(220, 262, 18, 18, rx=3,
                   fill=C["switch_fill"], stroke=C["switch_stroke"]))
    parts.append(T(244, 275, "Condensed switch", size=11, fill=C["legend_text"],
                   anchor="start"))

    parts.append(R(370, 266, 10, 10, rx=1,
                   fill=C["pe_fill"], stroke=C["pe_stroke"], sw=0.7))
    parts.append(T(386, 275, "Processing element (PE)", size=11,
                   fill=C["legend_text"], anchor="start"))

    # Row 2: Wrap-around link | Regular link
    parts.append(L(110, 298, 145, 298,
                   stroke=C["torus_wrap"], sw=0.8, dash="3 2"))
    parts.append(T(152, 302, "Wrap-around link (torus)", size=11,
                   fill=C["legend_text"], anchor="start"))

    parts.append(L(340, 298, 375, 298, stroke=C["edge"], sw=0.8))
    parts.append(T(382, 302, "Regular mesh link", size=11,
                   fill=C["legend_text"], anchor="start"))

    # Captions
    parts.append(T(340, 330,
                   "NoC topologies: 2-D Mesh, Torus, and Condensed Mesh (left to right)",
                   size=11, fill=C["caption"]))
    parts.append(T(340, 346,
                   "Dashed arcs in Torus indicate wrap-around links connecting opposite edges",
                   size=10, fill=C["caption2"]))
    parts.append(T(340, 360,
                   "Condensed Mesh uses fewer, high-bandwidth switch nodes (dark) to reduce hop count",
                   size=10, fill=C["caption2"]))

    return parts


# ─────────────────────────────────────────────────────────────────────────────
# Assemble full SVG
# ─────────────────────────────────────────────────────────────────────────────
def build_svg():
    W, H = 680, 380

    parts = []

    # ── Section titles ────────────────────────────────────────────────────────
    parts += [
        T(110, 2, "2-D Mesh",       size=13, weight=600, fill=C["title"]),
        T(340, 2, "Torus",          size=13, weight=600, fill=C["title"]),
        T(572, 2, "Condensed Mesh", size=13, weight=600, fill=C["title"]),
    ]

    # ── Topology diagrams ─────────────────────────────────────────────────────
    parts += draw_2d_mesh()
    parts += draw_torus()
    parts += draw_condensed_mesh()

    # ── Legend + captions ─────────────────────────────────────────────────────
    parts += draw_legend()

    body = "\n  ".join(parts)
    return (
        f'<svg width="{W}" viewBox="0 0 {W} {H}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background:white; font-family: Arial, sans-serif;">\n\n'
        f'  {body}\n\n'
        f'</svg>\n'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Export: SVG + PNG + PDF via PyMuPDF
# ─────────────────────────────────────────────────────────────────────────────
def export(svg_str, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    base     = os.path.join(out_dir, "noc_topologies")
    svg_path = base + ".svg"
    png_path = base + ".png"
    pdf_path = base + ".pdf"

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"SVG → {svg_path}")

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
