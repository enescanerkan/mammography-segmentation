"""Draw the architecture figures of docs/figures/architectures (light + dark SVG per model).

    python tools/figures/architectures.py            # writes docs/figures/architectures/*.svg
    python tools/figures/architectures.py unet       # one figure

Every shape and parameter count below was read from the models themselves: forward hooks on a 640 x 640 input
for the YOLO checkpoints, SegFormer (``nvidia/mit-b3`` config of the trained model), the nnU-Net plans of the
trained model, U-Net (``breast_seg.training.nets``) and the MedSAM2 checkpoint. Shapes are channels x height x
width for one image.

Layout: input on the left, then the bands (Backbone / Neck / Head, Encoder / Decoder / Head, ...) from left to
right, output on the right. Each row of blocks is one stride (/4, /8, ...); the row gets shorter as the
resolution drops, so the feature pyramid is visible at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svgdiagram import Diagram, save  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "architectures"

M, GAP, PAD = 24, 22, 16  # outer margin, gap between columns, padding inside a band
TOP, HEAD = 84, 46  # top of the bands, height of a band header
PYRAMID = [(4, 108), (8, 94), (16, 80), (32, 66)]  # stride -> row height of the four-level figures


class Rows:
    """One row per stride. ``spec``: (stride, height) or (stride, height, extra gap above)."""

    def __init__(self, top: float, gap: float, spec):
        self._y, self._h = {}, {}
        y = top
        for key, h, *extra in spec:
            y += extra[0] if extra else 0
            self._y[key], self._h[key] = y, h
            y += h + gap

    def top(self, k):
        return self._y[k]

    def h(self, k):
        return self._h[k]

    def bottom(self, k):
        return self._y[k] + self._h[k]

    def mid(self, k):
        return self._y[k] + self._h[k] / 2


def _band(d: Diagram, x, w, bottom, title, accent, sub="", top=TOP):
    return d.band(x, top, w, bottom + PAD - top, title, accent, sub)


def _io(d: Diagram, x, w, cy, h, title, shape="", lines=()):
    return d.block(x, cy - h / 2, w, h, title, shape, lines, accent="io")


# ── YOLO26 (pose and seg share backbone and neck) ──────────────────────────────
def yolo(title: str, subtitle: str, n: int, head: str, head_detail: str, input_detail: str, out_shape: str,
         out_lines, proto: bool) -> Diagram:
    W = 1360
    d = Diagram(W, title, subtitle)
    R = Rows(TOP + HEAD, 28, PYRAMID)
    bottom = R.bottom(32)
    w_in, bb_w, col_w, hd_w = 128, 282, 184, 242
    nk_w = 2 * PAD + 2 * col_w + 28
    x_bb = M + w_in + GAP
    x_nk = x_bb + bb_w + GAP
    x_hd = x_nk + nk_w + GAP
    x_out = x_hd + hd_w + GAP
    _band(d, x_bb, bb_w, bottom, "Backbone", "blue")
    _band(d, x_nk, nk_w, bottom, "Neck", "teal", "FPN top-down · PAN bottom-up")
    _band(d, x_hd, hd_w, bottom, "Head", "amber")

    inp = _io(d, M, w_in, R.mid(4), R.h(4) + 44, "Input", "3 × 640 × 640", [input_detail])

    bx, bw = x_bb + PAD, bb_w - 2 * PAD
    stem = d.block(bx, R.top(4), bw, R.h(4), f"Stem · Conv s2 ×2, C3k2 ×{n}", "256 × 160 × 160", tag="/4")
    p3 = d.block(bx, R.top(8), bw, R.h(8), f"Conv s2 · C3k2 ×{n}", "512 × 80 × 80", tag="P3 · /8")
    p4 = d.block(bx, R.top(16), bw, R.h(16), f"Conv s2 · C3k2 ×{n} (C3k)", "512 × 40 × 40", tag="P4 · /16")
    p5 = d.block(bx, R.top(32), bw, R.h(32), f"Conv s2 · C3k2 ×{n} (C3k)", "512 × 20 × 20",
                 [f"SPPF · C2PSA ×{n} (self-attention)"], tag="P5 · /32")

    ax = x_nk + PAD
    cx = ax + col_w + 28
    td4 = d.block(ax, R.top(16), col_w, R.h(16), f"concat · C3k2 ×{n}", "512 × 40 × 40", accent="teal")
    td3 = d.block(ax, R.top(8), col_w, R.h(8), f"concat · C3k2 ×{n}", "256 × 80 × 80", accent="teal", tag="P3′")
    bu4 = d.block(cx, R.top(16), col_w, R.h(16), f"concat · C3k2 ×{n}", "512 × 40 × 40", accent="teal", tag="P4′")
    bu5 = d.block(cx, R.top(32), col_w, R.h(32), "concat · C3k2 (PSA)", "512 × 20 × 20", accent="teal", tag="P5′")

    hx, hw = x_hd + PAD, hd_w - 2 * PAD
    heads = {s: d.block(hx, R.top(s), hw, R.h(s), head, f"{c} × {c} cells", [head_detail], accent="amber",
                        tag=f"/{s}")
             for s, c in ((8, 80), (16, 40), (32, 20))}
    top = R.top(8)
    if proto:
        pr = d.block(hx, R.top(4), hw, R.h(4), "Mask prototypes", "32 × 160 × 160", ["from P3′"], accent="amber",
                     tag="/4")
        top = R.top(4)
    out = d.block(x_out, top, W - M - x_out, bottom - top, "Output", out_shape, out_lines, accent="io")

    # backbone
    d.wire(inp.r, stem.l)
    for a, b in ((stem, p3), (p3, p4), (p4, p5)):
        d.wire(a.b, b.t)
    # neck: laterals, top-down, bottom-up
    d.wire(p3.r, td3.l)
    d.wire(p4.r, td4.l)
    d.wire(p5.r, bu5.l)
    d.wire((td4.cx, p5.cy), td4.b, label="upsample ×2")
    d.dot(td4.cx, p5.cy)
    d.wire(td4.t, td3.b, label="upsample ×2")
    d.wire(td4.r, bu4.l)
    d.wire((bu4.cx, td3.cy), bu4.t, label="conv s2")
    d.dot(bu4.cx, td3.cy)
    d.wire(bu4.b, bu5.t, label="conv s2")
    # heads and output
    d.wire(td3.r, heads[8].l)
    d.wire(bu4.r, heads[16].l)
    d.wire(bu5.r, heads[32].l)
    if proto:
        xp = x_nk + nk_w + GAP / 2
        d.wire((xp, td3.cy), (xp, pr.cy), pr.l)
        d.dot(xp, td3.cy)
        d.wire(pr.r, (out.x, pr.cy))
    for h in heads.values():
        d.wire(h.r, (out.x, h.cy))
    return d


def fig_pose() -> Diagram:
    return yolo("Landmark models · YOLO26l-pose (MLO and CC)",
                "3 × 640 × 640 wavelet canvas  →  breast box + keypoints · 27.9 M parameters",
                2, "Pose26 head", "box 4 · score 1 · keypoints K×3", "wavelet canvas", "300 × (6 + 3K)",
                ["top 300 · no NMS", "best 'Breast' box", "",
                 ("MLO · K = 3", "bold"), "nipple", "pectoral top", "pectoral bottom", "",
                 ("CC · K = 1", "bold"), "nipple"], proto=False)


def fig_seg() -> Diagram:
    return yolo("Segmentation model · YOLO26m-seg",
                "3 × 640 × 640 canvas  →  instances of pectoral, breast tissue, nipple · 27.0 M parameters",
                1, "Segment26 head", "box 4 · class 3 · coef 32", "canvas", "300 × (6 + 32)",
                ["top 300 · no NMS", "",
                 ("Masks", "bold"), "coef · prototypes", ("→ 640 × 640", "mono"), "",
                 ("Classes", "bold"), "pectoral", "breast tissue", "nipple"], proto=True)


# ── SegFormer ──────────────────────────────────────────────────────────────────
def fig_segformer() -> Diagram:
    W = 1360
    d = Diagram(W, "Semantic segmentation · SegFormer-B3",
                "3 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 47.2 M parameters")
    R = Rows(TOP + HEAD, 28, PYRAMID)
    bottom = R.bottom(32)
    w_in, en_w, mlp_w, fuse_w, hd_w = 128, 290, 200, 172, 212
    de_w = 2 * PAD + mlp_w + 28 + fuse_w
    x_en = M + w_in + GAP
    x_de = x_en + en_w + GAP
    x_hd = x_de + de_w + GAP
    x_out = x_hd + hd_w + GAP
    _band(d, x_en, en_w, bottom, "Encoder", "blue", "MiT-B3 · 44.1 M")
    _band(d, x_de, de_w, bottom, "Decoder", "teal", "all-MLP · 3.2 M")
    _band(d, x_hd, hd_w, bottom, "Head", "amber")

    inp = _io(d, M, w_in, R.mid(4), R.h(4) + 44, "Input", "3 × 640 × 640", ["grey ×3", "ImageNet norm."])
    stages = [  # stride, channels, blocks, heads, patch embedding, reduction ratio
        (4, "64 × 160 × 160", 3, "1 head", "7×7, s4", 8),
        (8, "128 × 80 × 80", 4, "2 heads", "3×3, s2", 4),
        (16, "320 × 40 × 40", 18, "5 heads", "3×3, s2", 2),
        (32, "512 × 20 × 20", 3, "8 heads", "3×3, s2", 1),
    ]
    ex = x_en + PAD
    mx = x_de + PAD
    fuse = d.block(mx + mlp_w + 28, R.top(4), fuse_w, bottom - R.top(4), "Concat + fuse", "3072 → 768",
                   ["1×1 conv, BN, ReLU", "dropout 0.1"], accent="teal")
    prev = inp
    for i, (s, shape, blocks, heads, patch, r) in enumerate(stages):
        st = d.block(ex, R.top(s), en_w - 2 * PAD, R.h(s), f"Stage {i + 1} · {blocks} blocks", shape,
                     [f"patch {patch} · {heads} · R = {r}"], tag=f"/{s}")
        mlp = d.block(mx, R.top(s), mlp_w, R.h(s), f"Linear {shape.split(' ')[0]} → 768",
                      ("" if s == 4 else "↑ ") + "768 × 160 × 160", accent="teal")
        d.wire(prev.r if prev is inp else prev.b, st.l if prev is inp else st.t)
        d.wire(st.r, mlp.l)
        d.wire(mlp.r, (fuse.x, mlp.cy))
        prev = st
    cls = d.block(x_hd + PAD, fuse.cy - 45, hd_w - 2 * PAD, 90, "Classifier", "4 × 160 × 160", ["1×1 conv → 4"],
                  accent="amber", tag="/4")
    out = _io(d, x_out, W - M - x_out, fuse.cy, 176, "Output", "4 × 640 × 640",
              ["bilinear ×4", "", ("Classes", "bold"), "background", "pectoral", "breast tissue", "nipple"])
    d.wire((fuse.right, cls.cy), cls.l)
    d.wire(cls.r, out.l)
    return d


# ── U-shaped networks ──────────────────────────────────────────────────────────
def u_net(title: str, subtitle: str, levels, bottleneck, gap: float, input_lines, head, down: str, up: str,
          deep_supervision=None) -> Diagram:
    """``levels``: (stride, row height, encoder title, encoder shape, decoder title, decoder shape), top to bottom;
    ``bottleneck``: (stride, row height, title, shape)."""
    W = 1320
    d = Diagram(W, title, subtitle)
    spec = [lv[:2] for lv in levels] + [(bottleneck[0], bottleneck[1], 12)]
    if deep_supervision:  # room for the title of the deep-supervision group
        spec[1] = (*spec[1], 18)
    R = Rows(TOP + HEAD, gap, spec)
    first, last, low = levels[0][0], levels[-1][0], bottleneck[0]
    w_in, en_w, hd_w, w_out = 128, 252, 212, 128
    x_en = M + w_in + GAP
    x_out = W - M - w_out
    x_hd = x_out - GAP - hd_w
    x_de = x_hd - GAP - en_w
    x_bn = x_en + en_w + 12
    bn_w = x_de - 12 - x_bn
    _band(d, x_en, en_w, R.bottom(last), "Encoder", "blue")
    _band(d, x_de, en_w, R.bottom(last), "Decoder", "teal")
    _band(d, x_bn, bn_w, R.bottom(low), "Bottleneck", "violet", top=R.top(low) - HEAD)
    _band(d, x_hd, hd_w, R.bottom(deep_supervision[-1][0] if deep_supervision else first), "Head", "amber")

    inp = _io(d, M, w_in, R.mid(first), R.h(first), "Input", *input_lines)
    bw = en_w - 2 * PAD
    enc, dec = [], []
    for s, h, et, es, dt, ds in levels:
        tag = "" if s == 1 else f"/{s}"
        enc.append(d.block(x_en + PAD, R.top(s), bw, h, et, es, tag=tag))
        dec.append(d.block(x_de + PAD, R.top(s), bw, h, dt, ds, accent="teal", tag=tag))
    bn = d.block(x_bn + PAD, R.top(low), bn_w - 2 * PAD, R.h(low), bottleneck[2], bottleneck[3], accent="violet",
                 tag=f"/{low}")
    hd = d.block(x_hd + PAD, R.top(first), hd_w - 2 * PAD, R.h(first), *head, accent="amber")
    out = _io(d, x_out, w_out, R.mid(first), max(R.h(first), 124), "Output", "",
              [("4 classes", "bold"), "background", "pectoral", "breast tissue", "nipple"])

    d.wire(inp.r, enc[0].l)
    for a, b in zip(enc, enc[1:]):
        d.wire(a.b, b.t, label=down)
    for a, b in zip(dec[1:], dec):
        d.wire(a.t, b.b, label=up, side="left")
    d.wire(enc[-1].b, (enc[-1].cx, bn.cy), bn.l, label=down, seg=0)
    d.wire(bn.r, (dec[-1].cx, bn.cy), dec[-1].b, label=up, seg=1, side="left")
    for i, (e, dd) in enumerate(zip(enc, dec)):
        d.wire(e.r, dd.l, dashed=True, label="skip · concat" if i == 0 else "", side="above")
    d.wire(dec[0].r, hd.l)
    d.wire(hd.r, out.l)
    if deep_supervision:
        g = d.group(hd.x, R.bottom(first) + 10, hd.w, R.bottom(deep_supervision[-1][0]) - R.bottom(first) - 10,
                    "Deep supervision", "amber")
        d.text(g.x + 12, g.y + 38, "training only")
        for (s, shape), dd in zip(deep_supervision, dec[1:]):
            d.wire(dd.r, (g.x, R.mid(s)), dashed=True)
            d.text(g.x + 14, R.mid(s) + 4, shape, mono=True, tone="text")
    return d


def fig_unet() -> Diagram:
    shapes = ["64 × 640 × 640", "128 × 320 × 320", "256 × 160 × 160", "512 × 80 × 80"]
    levels = [(s, h, "DoubleConv", sh, "DoubleConv", sh) for (s, h), sh in zip([(1, 94), (2, 82), (4, 70), (8, 58)],
                                                                             shapes)]
    return u_net("Semantic baseline · U-Net", "1 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 31.0 M parameters",
                 levels, (16, 46, "DoubleConv", "1024 × 40 × 40"), 24, ("1 × 640 × 640", ["grey, 0–1"]),
                 ("1×1 conv", "4 × 640 × 640"), down="max-pool 2×2", up="up-conv 2×2")


def fig_nnunet() -> Diagram:
    enc = ["Stem + 1 residual block", "3 residual blocks", "4 residual blocks", "6 residual blocks",
           "6 residual blocks", "6 residual blocks"]
    shapes = ["32 × 640 × 640", "64 × 320 × 320", "128 × 160 × 160", "256 × 80 × 80", "512 × 40 × 40",
              "512 × 20 × 20"]
    rows = [(1, 80), (2, 74), (4, 68), (8, 62), (16, 56), (32, 50)]
    levels = [(s, h, et, sh, "Conv 3×3, IN, LReLU", sh) for (s, h), et, sh in zip(rows, enc, shapes)]
    ds = [(2, "4 × 320 × 320"), (4, "4 × 160 × 160"), (8, "4 × 80 × 80"), (16, "4 × 40 × 40"), (32, "4 × 20 × 20")]
    return u_net("Semantic benchmark · nnU-Net ResEnc-M (2D)",
                 "1 × 640 × 640 canvas  →  4 × 640 × 640 class logits · 105.9 M parameters",
                 levels, (64, 44, "6 residual blocks", "512 × 10 × 10"), 22, ("1 × 640 × 640", ["z-score"]),
                 ("1×1 conv", "4 × 640 × 640"), down="stride 2", up="ConvT 2×2", deep_supervision=ds)


# ── MedSAM2 ────────────────────────────────────────────────────────────────────
def fig_medsam2() -> Diagram:
    W = 1400
    d = Diagram(W, "Prompted segmentation (annotation toolkit) · MedSAM2 (SAM2 Hiera-tiny)",
                "3 × 1024 × 1024 image + box  →  1 × 256 × 256 mask logits · 31.4 M parameters used "
                "(38.9 M in the checkpoint)")
    R = Rows(TOP + HEAD, 28, PYRAMID)
    w_in, st_w, fpn_w, z_w, a_w, b_w, w_out = 154, 196, 176, 136, 196, 176, 126
    im_w = 2 * PAD + st_w + 24 + fpn_w
    x_im = M + w_in + GAP
    x_md = x_im + im_w + GAP
    md_w = 2 * PAD + z_w + a_w + b_w + 2 * 24
    x_out = x_md + md_w + GAP
    im = _band(d, x_im, im_w, R.bottom(32), "Image encoder", "blue", "Hiera-T 26.8 M · FPN 0.4 M")
    pr_top = im.bottom + GAP
    dense_y = pr_top + HEAD
    sparse_y = dense_y + 58
    _band(d, x_im, im_w, sparse_y + 55, "Prompt encoder", "violet", "frozen", top=pr_top)
    _band(d, x_md, md_w, sparse_y + 55, "Mask decoder", "teal", "4.2 M")

    # image encoder: Hiera trunk (left column), FPN neck (right column)
    sx, fx = x_im + PAD, x_im + PAD + st_w + 24
    img = _io(d, M, w_in, R.mid(4), R.h(4) + 44, "Input image", "3 × 1024 × 1024", ["raw1024"])
    stages = [(4, "Stage 1 · 1 block", "96 × 256 × 256", ["patch embed 7×7, s4", "window 8"]),
              (8, "Stage 2 · 2 blocks", "192 × 128 × 128", ["window 4"]),
              (16, "Stage 3 · 7 blocks", "384 × 64 × 64", ["window 14 · 3 global"]),
              (32, "Stage 4 · 2 blocks", "768 × 32 × 32", ["window 7"])]
    st = {s: d.block(sx, R.top(s), st_w, R.h(s), t, sh, ln, tag=f"/{s}") for s, t, sh, ln in stages}
    f4 = d.block(fx, R.top(4), fpn_w, R.h(4), "1×1 lateral", "256 × 256 × 256")
    f8 = d.block(fx, R.top(8), fpn_w, R.h(8), "1×1 lateral", "256 × 128 × 128")
    emb = d.block(fx, R.top(16), fpn_w, R.h(16), "Image embedding", "256 × 64 × 64", ["+ no-memory embedding"])
    d.wire(img.r, st[4].l)
    for a, b in ((4, 8), (8, 16), (16, 32)):
        d.wire(st[a].b, st[b].t, label="Q-pool ×2")
    d.wire(st[4].r, f4.l)
    d.wire(st[8].r, f8.l)
    d.wire(st[16].r, emb.l)
    d.wire(st[32].r, (emb.cx, st[32].cy), emb.b, label="top-down", seg=1)

    # prompt encoder
    box = _io(d, M, w_in, sparse_y + 24, 72, "Box prompt", "(x0, y0, x1, y1)", ["2 corners, labels 2 / 3"])
    dense = d.block(sx, dense_y, 300, 46, "Dense prompt + pos. enc.", "256 × 64 × 64 each", accent="violet")
    sparse = d.block(sx, sparse_y, 300, 48, "Sparse tokens", "3 × 256 (2 corners + pad)", accent="violet")
    d.wire(box.r, sparse.l)

    # mask decoder: high-res features (Z), transformer and upscaling (A), hypernetwork (B)
    zx = x_md + PAD
    ax = zx + z_w + 24
    bx = ax + a_w + 24
    s0 = d.block(zx, R.top(4), z_w, R.h(4), "High-res s0", "32 × 256 × 256", accent="teal")
    s1 = d.block(zx, R.top(8), z_w, R.h(8), "High-res s1", "64 × 128 × 128", accent="teal")
    add = d.op(zx + z_w / 2, R.mid(16), "+", "teal")
    up2 = d.block(ax, R.top(4), a_w, R.h(4), "ConvT ×2 + s0", "32 × 256 × 256", accent="teal", tag="/4")
    up1 = d.block(ax, R.top(8), a_w, R.h(8), "ConvT ×2 + s1", "64 × 128 × 128", accent="teal", tag="/8")
    tf = d.block(ax, R.top(16), a_w, R.h(16), "Two-way transformer ×2", "",
                 ["256-d · 8 heads · MLP 2048", "tokens ↔ image attention"], accent="teal")
    tok = d.block(ax, sparse.cy - 31, a_w, 62, "Output tokens", "", ["obj-score · IoU · 4 mask", "+ prompt tokens"],
                  accent="teal")
    hyp = d.block(bx, R.top(16), b_w, R.h(16), "Hypernetwork MLP", "",
                  ["mask token → 32 weights", "IoU + object-score heads"], accent="teal")
    out = d.block(x_out, R.top(4), w_out, R.h(4), "Mask logits", "1 × 256 × 256", ["single mask"], accent="io")

    d.wire(f4.r, s0.l)
    d.wire(f8.r, s1.l)
    d.wire(emb.r, add.l)
    d.wire(dense.r, (add.cx, dense.cy), add.b)
    d.wire(sparse.r, tok.l)
    d.wire(add.r, tf.l)
    d.wire(tok.t, tf.b)
    d.wire(s1.r, up1.l)
    d.wire(s0.r, up2.l)
    d.wire(tf.t, up1.b)
    d.wire(up1.t, up2.b)
    d.wire(tf.r, hyp.l)
    d.wire(up2.r, out.l)
    d.wire(hyp.r, (out.cx, hyp.cy), out.b)
    return d


FIGURES = {
    "yolo26l_pose": fig_pose,
    "yolo26m_seg": fig_seg,
    "segformer_b3": fig_segformer,
    "nnunet_resenc_m": fig_nnunet,
    "unet": fig_unet,
    "medsam2": fig_medsam2,
}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    status = 0
    for name in names:
        diagram = FIGURES[name]()
        for problem in diagram.problems():
            print(f"{name}: {problem}", file=sys.stderr)
            status = 1
        for path in save(diagram, OUT, name):
            print(path)
    return status


if __name__ == "__main__":
    sys.exit(main())
