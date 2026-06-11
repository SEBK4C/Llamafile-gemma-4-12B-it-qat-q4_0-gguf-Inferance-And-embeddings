"""Visualize what the gemma4uv (unified Gemma 4 vision) preprocessing does to
an image BEFORE it reaches the model — with this repo's exact settings
(patch 0010: budget-fill resize to 280 soft tokens, 48px patches, bicubic).

    uv run --with pillow python3 tests/viz_preprocess.py photo.png
    open photo.viz_pipeline.png photo.viz_tokens.png

Outputs, next to the input (or --out-dir):
  <name>.viz_pipeline.png  source | old pipeline | new pipeline, each with the
                           48px token grid + row-major reading order overlaid
  <name>.viz_tokens.png    contact sheet: each tile is exactly ONE soft
                           token's field of view (48x48 of the resized image),
                           magnified — the densest rows are picked
                           automatically, or use --rows A-B

Flags mirror the C++ (tools/mtmd/mtmd-image.cpp):
  --max-tokens 280   soft-token budget (img_tool::calc_size_fill_budget)
  --align 48         effective patch size (16px base x3 merge, folded)
  --old              ALSO simulate the pre-patch-0010 smart_resize path
"""

import argparse
import math
import os

from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Helvetica.ttc"


def budget_fill_size(w, h, align, max_tokens):
    """img_tool::calc_size_fill_budget — HF get_aspect_ratio_preserving_size."""
    target_px = max_tokens * align * align
    f = math.sqrt(target_px / (w * h))
    flr = lambda x: max(align, int(x) // align * align)
    return flr(f * w), flr(f * h)


def smart_resize_size(w, h, align, min_tokens, max_tokens):
    """calc_size_preserved_ratio — the pre-patch-0010 path, for comparison."""
    rb = lambda x: max(align, round(x / align) * align)
    wb, hb = rb(w), rb(h)
    if wb * hb > max_tokens * align * align:
        b = math.sqrt((w * h) / (max_tokens * align * align))
        flr = lambda x: max(align, math.floor(x / align) * align)
        wb, hb = flr(w / b), flr(h / b)
    elif wb * hb < min_tokens * align * align:
        b = math.sqrt(min_tokens * align * align / (w * h))
        ceil = lambda x: math.ceil(x / align) * align
        wb, hb = ceil(w * b), ceil(h * b)
    return wb, hb


def gridded(img, label, align, font_s, font_t):
    g = img.copy()
    d = ImageDraw.Draw(g)
    cols, rows = g.width // align, g.height // align
    for i in range(cols * rows):
        x, y = (i % cols) * align, (i // cols) * align
        d.rectangle([x, y, x + align, y + align], outline=(255, 0, 0), width=1)
        d.text((x + 2, y + 1), str(i), fill=(0, 100, 255), font=font_s)
    pad = 26
    out = Image.new("RGB", (g.width, g.height + pad), "white")
    out.paste(g, (0, pad))
    ImageDraw.Draw(out).text((4, 4), f"{label}  ({cols}x{rows} = {cols * rows} tokens)",
                             fill="black", font=font_t)
    return out


def ink_per_row(img, align):
    """Mean darkness per patch row — to auto-pick interesting contact-sheet rows."""
    gray = img.convert("L")
    rows = img.height // align
    px = gray.load()
    scores = []
    for r in range(rows):
        s = 0
        for y in range(r * align, (r + 1) * align, 4):
            for x in range(0, img.width, 4):
                s += 255 - px[x, y]
        scores.append(s)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--max-tokens", type=int, default=280)
    ap.add_argument("--min-tokens", type=int, default=40, help="old-path floor")
    ap.add_argument("--align", type=int, default=48)
    ap.add_argument("--old", action="store_true", help="include pre-fix smart_resize panel")
    ap.add_argument("--rows", default=None, help="contact-sheet patch rows, e.g. 2-4")
    ap.add_argument("--mag", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = Image.open(args.image).convert("RGB")
    font_s = ImageFont.truetype(FONT, 11)
    font_t = ImageFont.truetype(FONT, 16)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.image))

    new_w, new_h = budget_fill_size(*src.size, args.align, args.max_tokens)
    new_img = src.resize((new_w, new_h), Image.BICUBIC)

    views = [gridded(src, f"source {src.width}x{src.height}", args.align, font_s, font_t)]
    if args.old:
        ow, oh = smart_resize_size(*src.size, args.align, args.min_tokens, args.max_tokens)
        views.append(gridded(src.resize((ow, oh), Image.BILINEAR),
                             f"OLD smart_resize -> {ow}x{oh} BILINEAR", args.align, font_s, font_t))
    views.append(gridded(new_img, f"model input (patch 0010): budget-fill -> {new_w}x{new_h} BICUBIC",
                         args.align, font_s, font_t))

    W = sum(v.width for v in views) + 20 * (len(views) + 1)
    H = max(v.height for v in views) + 20
    canvas = Image.new("RGB", (W, H), (235, 235, 235))
    x = 20
    for v in views:
        canvas.paste(v, (x, 10))
        x += v.width + 20
    p1 = os.path.join(out_dir, f"{base}.viz_pipeline.png")
    canvas.save(p1)

    # contact sheet
    cols = new_w // args.align
    n_rows = new_h // args.align
    if args.rows:
        a, _, b = args.rows.partition("-")
        sel = range(int(a), min(int(b or a) + 1, n_rows))
    else:
        scores = ink_per_row(new_img, args.align)
        best = max(range(n_rows), key=lambda r: scores[r] + (scores[r + 1] if r + 1 < n_rows else 0))
        sel = range(best, min(best + 2, n_rows))
    tile = args.align * args.mag
    tiles = []
    for r in sel:
        for c in range(cols):
            i = r * cols + c
            t = new_img.crop((c * args.align, r * args.align,
                              (c + 1) * args.align, (r + 1) * args.align))
            t = t.resize((tile, tile), Image.NEAREST)
            d = ImageDraw.Draw(t)
            d.rectangle([0, 0, tile - 1, tile - 1], outline=(255, 0, 0), width=2)
            d.text((4, 2), f"tok {i}", fill=(0, 100, 255), font=font_t)
            tiles.append(t)
    sheet_cols = min(cols, 8)
    rows_n = math.ceil(len(tiles) / sheet_cols)
    sheet = Image.new("RGB", (sheet_cols * (tile + 8) + 8, rows_n * (tile + 8) + 40), "white")
    ImageDraw.Draw(sheet).text(
        (8, 6),
        f"each tile = ONE soft token ({args.align}px of the {new_w}x{new_h} model input, "
        f"{args.mag}x mag), patch rows {sel.start}-{sel.stop - 1}",
        fill="black", font=font_t)
    for n, t in enumerate(tiles):
        sheet.paste(t, (8 + (n % sheet_cols) * (tile + 8), 40 + (n // sheet_cols) * (tile + 8)))
    p2 = os.path.join(out_dir, f"{base}.viz_tokens.png")
    sheet.save(p2)

    n_tok = (new_w // args.align) * (new_h // args.align)
    print(f"model input: {new_w}x{new_h} = {n_tok} soft tokens "
          f"(scale {new_w / src.width:.2f}x{', UPSCALED — detail cannot be recovered' if new_w > src.width else ''})")
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
