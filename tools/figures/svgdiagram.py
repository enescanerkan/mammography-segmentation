"""A small SVG block-diagram writer for the architecture figures (light and dark variants from one drawing).

A figure is laid out once in pixel coordinates and rendered per theme, so ``<name>.svg`` and ``<name>_dark.svg``
always show the same diagram. The visual language is shared by every figure:

* bands: rounded, lightly tinted columns with a title (Backbone, Neck, Head, ...), one accent colour each;
* blocks: feature maps or layers inside a band, a bold title, the tensor shape in monospace and an optional
  stride pill; the caller scales block heights with the spatial resolution so the pyramid is visible;
* wires: orthogonal polylines with an arrow head, optional dashes (skip connections) and a short label.

``Diagram.problems()`` estimates text widths and reports any label that would not fit its block or band.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, Union
from xml.sax.saxutils import escape

FONT = "Inter, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Text sizes (px at the native width of the figure).
TITLE, SUBTITLE, BAND, BLOCK, SMALL = 20, 13.5, 14, 13, 12

PALETTES = {
    "light": {
        "bg": "#FFFFFF", "ink": "#0F172A", "text": "#334155", "muted": "#64748B", "wire": "#6B7686",
        "io": {"fill": "#F1F5F9", "stroke": "#94A3B8", "label": "#475569", "pill": "#E2E8F0"},
        "blue": {"band": "#F5F8FF", "line": "#DCE5FB", "fill": "#E6EDFD", "stroke": "#4A6FD8", "label": "#2F4FB0",
                 "pill": "#D3DFFB"},
        "teal": {"band": "#F2FAF9", "line": "#D3ECEA", "fill": "#DDF3F1", "stroke": "#1E9B96", "label": "#0F7470",
                 "pill": "#C4EAE6"},
        "amber": {"band": "#FFF9F0", "line": "#F4E3C6", "fill": "#FCEFD9", "stroke": "#D08A1E", "label": "#965C05",
                  "pill": "#F7DEB4"},
        "violet": {"band": "#F8F6FE", "line": "#E4DEFA", "fill": "#ECE8FC", "stroke": "#7B63D6", "label": "#5B43B5",
                   "pill": "#DCD4FA"},
    },
    "dark": {
        "bg": "#0D1117", "ink": "#E6EDF3", "text": "#C3CCD8", "muted": "#8D99AA", "wire": "#8591A3",
        "io": {"fill": "#161B22", "stroke": "#6E7781", "label": "#C3CCD8", "pill": "#262D37"},
        "blue": {"band": "#0F1626", "line": "#1F2E4D", "fill": "#16264A", "stroke": "#7393EE", "label": "#A3B8F6",
                 "pill": "#22396B"},
        "teal": {"band": "#0C1A1B", "line": "#1A3A3A", "fill": "#113233", "stroke": "#3FBDB6", "label": "#74D5CF",
                 "pill": "#184A47"},
        "amber": {"band": "#1A150D", "line": "#3E3018", "fill": "#33270F", "stroke": "#E3A646", "label": "#F2C273",
                  "pill": "#4D3A16"},
        "violet": {"band": "#14111F", "line": "#2E2650", "fill": "#231C40", "stroke": "#9F8AEF", "label": "#BEAEF7",
                   "pill": "#372C66"},
    },
}

Point = Tuple[float, float]
Line = Union[str, Tuple[str, str]]  # "text" (detail) or (text, style), style in mono / bold / detail / gap

_ROW = {"title": (BLOCK, 18), "mono": (SMALL, 17), "detail": (SMALL, 16), "bold": (SMALL, 17), "gap": (0, 6)}


def text_width(s: str, size: float, bold: bool = False, mono: bool = False) -> float:
    """A generous estimate of the rendered width (Segoe UI / Helvetica / Arial; monospace at 0.6 em)."""
    if mono:
        return 0.61 * size * len(s)
    em = 0.0
    for ch in s:
        if ch in "iljtfrI.,:;'|!()[] ·′":
            em += 0.31
        elif ch in "mwMW":
            em += 0.86
        elif ch.isupper():
            em += 0.68
        elif ch.isdigit():
            em += 0.57
        elif ch in "→↑↓←":
            em += 1.0
        else:
            em += 0.55
    return em * size * (1.07 if bold else 1.0) * 1.04


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def l(self) -> Point:  # noqa: E743
        return (self.x, self.y + self.h / 2)

    @property
    def r(self) -> Point:
        return (self.x + self.w, self.y + self.h / 2)

    @property
    def t(self) -> Point:
        return (self.x + self.w / 2, self.y)

    @property
    def b(self) -> Point:
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class Diagram:
    width: float
    title: str
    subtitle: str = ""
    _bands: list = field(default_factory=list)
    _wires: list = field(default_factory=list)
    _items: list = field(default_factory=list)
    _front: list = field(default_factory=list)
    _bottom: float = 70.0

    # ── primitives ─────────────────────────────────────────────────────────
    def band(self, x, y, w, h, title: str, accent: str, sub: str = "") -> Box:
        """A titled column; ``accent`` is one of blue / teal / amber / violet."""
        self._bands.append(("band", x, y, w, h, title, accent, sub))
        self._grow(y + h)
        return Box(x, y, w, h)

    def block(self, x, y, w, h, title: str, shape: str = "", lines: Sequence[Line] = (), accent: str = "blue",
              tag: str = "") -> Box:
        """A layer or feature map: bold title, monospace shape, extra lines; ``tag`` is a pill at the top right.
        ``accent="io"`` draws a neutral input / output block."""
        rows = [(title, "title")] + ([(shape, "mono")] if shape else [])
        rows += [(ln, "detail") if isinstance(ln, str) else ln for ln in lines]
        rows = [(t, "gap") if s != "gap" and t == "" else (t, s) for t, s in rows]
        self._items.append(("block", x, y, w, h, tuple(rows), accent, tag))
        self._grow(y + h)
        return Box(x, y, w, h)

    def group(self, x, y, w, h, title: str, accent: str) -> Box:
        """A dashed container (e.g. the deep-supervision outputs)."""
        self._items.append(("group", x, y, w, h, title, accent))
        self._grow(y + h)
        return Box(x, y, w, h)

    def op(self, cx, cy, symbol: str, accent: str, r: float = 12) -> Box:
        """A small circular operator (add, concat) centred at (cx, cy)."""
        self._items.append(("op", cx, cy, r, symbol, accent))
        return Box(cx - r, cy - r, 2 * r, 2 * r)

    def wire(self, *points: Point, dashed: bool = False, head: bool = True, label: str = "", seg: int = -1,
             side: str = "right"):
        """Orthogonal polyline; ``label`` sits beside segment ``seg`` (``side``: right / left / above / below)."""
        self._wires.append(("wire", tuple(points), dashed, head))
        if label:
            i = seg % (len(points) - 1)
            (x0, y0), (x1, y1) = points[i], points[i + 1]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if side == "right":
                self.text(mx + 8, my + 4, label, anchor="start")
            elif side == "left":
                self.text(mx - 8, my + 4, label, anchor="end")
            elif side == "above":
                self.text(mx, my - 7, label, anchor="middle")
            else:
                self.text(mx, my + 16, label, anchor="middle")

    def dot(self, x, y):
        """A junction dot where a wire branches."""
        self._front.append(("dot", x, y))

    def text(self, x, y, s: str, size: float = SMALL, anchor: str = "start", tone: str = "muted", bold=False,
             mono=False):
        self._front.append(("text", x, y, s, size, anchor, tone, bold, mono))
        self._grow(y + 6)

    def _grow(self, y: float):
        self._bottom = max(self._bottom, y)

    @property
    def height(self) -> float:
        return self._bottom + 22

    # ── checks ─────────────────────────────────────────────────────────────
    def problems(self) -> list:
        """Labels whose estimated width does not fit their block, group or band."""
        out = []
        for item in self._items:
            if item[0] != "block":
                continue
            _, x, y, w, h, rows, accent, tag = item
            room = w - 24
            if sum(_ROW[s][1] for _, s in rows) - 4 > h - 8:
                out.append(f"block {rows[0][0]!r}: {len(rows)} lines do not fit {h:.0f} px")
            pill_row = _pill_row(rows)
            for i, (t, s) in enumerate(rows):
                if s == "gap":
                    continue
                size = _ROW[s][0]
                width = text_width(t, size, bold=s in ("title", "bold"), mono=s == "mono")
                limit = room - (_pill_width(tag) + 8 if tag and i == pill_row else 0)
                if width > limit:
                    out.append(f"block {rows[0][0]!r}: {t!r} is {width:.0f} px, room {limit:.0f} px")
        for band in self._bands:
            w, title, sub = band[3], band[5], band[7]
            need = 30 + text_width(title, BAND, bold=True) + (16 + text_width(sub, SMALL) if sub else 0)
            if need > w - 16:
                out.append(f"band {title!r}: header needs {need:.0f} px, band is {w:.0f} px")
        return out

    # ── rendering ──────────────────────────────────────────────────────────
    def render(self, theme: str = "light") -> str:
        p = PALETTES[theme]
        w, h = self.width, self.height
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}" '
               f'font-family="{FONT}" role="img" aria-label="{escape(self.title)}">',
               f"<title>{escape(self.title)}</title>",
               "<defs>"
               f'<marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" '
               f'markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0.5 L10,5 L0,9.5 z" fill="{p["wire"]}"/></marker>'
               "</defs>",
               f'<rect width="{w:.0f}" height="{h:.0f}" rx="12" fill="{p["bg"]}"/>',
               _t(24, 38, self.title, TITLE, "start", p["ink"], bold=True)]
        if self.subtitle:
            out.append(_t(24, 62, self.subtitle, SUBTITLE, "start", p["muted"]))
        for item in self._bands:
            out.append(self._band(item, p))
        for item in self._wires:
            out.append(self._wire(item, p))
        for item in self._items:
            out.append(self._item(item, p))
        for item in self._front:
            out.append(self._front_item(item, p))
        out.append("</svg>")
        return "\n".join(s for s in out if s)

    @staticmethod
    def _band(item, p) -> str:
        _, x, y, w, h, title, accent, sub = item
        a = p[accent]
        s = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="14" fill="{a["band"]}" '
             f'stroke="{a["line"]}"/>',
             f'<rect x="{x + 16:.1f}" y="{y + 15:.1f}" width="4" height="16" rx="2" fill="{a["stroke"]}"/>',
             _t(x + 28, y + 28, title, BAND, "start", a["label"], bold=True)]
        if sub:
            s.append(_t(x + w - 16, y + 28, sub, SMALL, "end", p["muted"]))
        return "\n".join(s)

    @staticmethod
    def _wire(item, p) -> str:
        _, pts, dashed, head = item
        pts = list(pts)
        if head:  # stop the line inside the arrow head so its tip stays sharp
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            dx, dy = x1 - x0, y1 - y0
            n = max(abs(dx), abs(dy)) or 1.0
            pts[-1] = (x1 - 4 * dx / n, y1 - 4 * dy / n)
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        mk = ' marker-end="url(#arrow)"' if head else ""
        return (f'<path d="{d}" fill="none" stroke="{p["wire"]}" stroke-width="1.5" stroke-linejoin="round"'
                f"{dash}{mk}/>")

    @staticmethod
    def _item(item, p) -> str:
        kind = item[0]
        if kind == "block":
            _, x, y, w, h, rows, accent, tag = item
            a = p[accent]
            s = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="{a["fill"]}" '
                 f'stroke="{a["stroke"]}" stroke-width="1.25"/>']
            total = sum(_ROW[r][1] for _, r in rows) - 4
            cy = y + (h - total) / 2
            pill_row, pill_base = _pill_row(rows), None
            for i, (text, role) in enumerate(rows):
                size, adv = _ROW[role]
                if role != "gap":
                    base = cy + size * 0.8
                    if i == pill_row:
                        pill_base = base
                    if role == "title":
                        s.append(_t(x + 12, base, text, size, "start", p["ink"], bold=True))
                    elif role == "mono":
                        s.append(_t(x + 12, base, text, size, "start", p["text"], mono=True))
                    elif role == "bold":
                        s.append(_t(x + 12, base, text, size, "start", p["text"], bold=True))
                    else:
                        s.append(_t(x + 12, base, text, size, "start", p["muted"]))
                cy += adv
            if tag:  # a pill on the shape line, right-aligned
                pw = _pill_width(tag)
                px, py = x + w - 10 - pw, pill_base - 13.5
                s.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" height="19" rx="9.5" fill="{a["pill"]}"/>')
                s.append(_t(px + pw / 2, pill_base, tag, SMALL, "middle", a["label"], bold=True))
            return "\n".join(s)
        if kind == "group":
            _, x, y, w, h, title, accent = item
            a = p[accent]
            return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" fill="none" '
                    f'stroke="{a["stroke"]}" stroke-width="1.25" stroke-dasharray="5 4"/>\n'
                    + _t(x + 12, y + 21, title, BLOCK, "start", p["ink"], bold=True))
        if kind == "op":
            _, cx, cy, r, symbol, accent = item
            a = p[accent]
            return (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{a["fill"]}" stroke="{a["stroke"]}" '
                    f'stroke-width="1.25"/>\n' + _t(cx, cy + 5.5, symbol, 16, "middle", p["ink"], bold=True))
        raise ValueError(kind)

    @staticmethod
    def _front_item(item, p) -> str:
        if item[0] == "dot":
            _, x, y = item
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{p["wire"]}"/>'
        _, x, y, s, size, anchor, tone, bold, mono = item
        return _t(x, y, s, size, anchor, p[tone], bold=bold, mono=mono)


def _pill_row(rows) -> int:
    """The pill sits on the shape (monospace) line, or on the title when there is none."""
    return next((i for i, (_, s) in enumerate(rows) if s == "mono"), 0)


def _pill_width(tag: str) -> float:
    return text_width(tag, SMALL, bold=True) + 14


def _t(x, y, s, size, anchor, color, bold=False, mono=False) -> str:
    weight = ' font-weight="600"' if bold else ""
    family = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}"'
            f"{weight}{family}>{escape(s)}</text>")


def save(diagram: Diagram, out_dir, name: str) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for theme, suffix in (("light", ""), ("dark", "_dark")):
        path = out_dir / f"{name}{suffix}.svg"
        path.write_text(diagram.render(theme) + "\n", encoding="utf-8")
        written.append(str(path))
    return written
