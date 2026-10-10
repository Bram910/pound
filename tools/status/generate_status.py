#!/usr/bin/env python3
"""Generate the README status treemap from a Ballistic source checkout.

Usage: python tools/status/generate_status.py --ballistic <path-to-ballistic>

Reads Ballistic's generated ARM64 decoder table and its x86 tier-1 compiler,
classifies every ARM64 encoding, and writes docs/status/*.svg plus the
status block in README.md (between the POUND-STATUS markers).
"""

import argparse
import html
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "status"
README = ROOT / "README.md"
MARK_BEGIN = "<!-- POUND-STATUS:BEGIN -->"
MARK_END = "<!-- POUND-STATUS:END -->"

RUNS, DECODED, TODO = "runs", "decoded", "todo"
COLORS = {RUNS: "#2ea043", DECODED: "#bf8700", TODO: "#6e7681"}
BORDER = "#0d1117"
REG = {"BAL_OPERAND_TYPE_REGISTER_32", "BAL_OPERAND_TYPE_REGISTER_64"}
IMM = "BAL_OPERAND_TYPE_IMMEDIATE"

# Names the tier-1 compiler hands to the host before its opcode switch.
HOST_TRAPPED = ("SVC", "SMC", "HVC", "WFI")

ENTRY_RE = re.compile(
    r'\{\s*"(?P<name>[^"]+)",\s*0x(?P<mask>[0-9A-Fa-f]+),\s*0x(?P<expected>[0-9A-Fa-f]+),'
    r"\s*(?P<opcode>OPCODE_\w+),\s*\{(?P<operands>(?:\s*\{[^}]*\},?)+)\s*\}",
)
OPERAND_RE = re.compile(r"\{\s*(\w+),\s*(\d+),\s*(\d+)\s*\}")


def parse_decoder_table(path):
    entries = []
    for m in ENTRY_RE.finditer(path.read_text(encoding="utf-8")):
        operands = [(t, int(p), int(w)) for t, p, w in OPERAND_RE.findall(m["operands"])]
        entries.append(
            {
                "name": m["name"],
                "expected": int(m["expected"], 16),
                "opcode": m["opcode"],
                "operands": operands,
            }
        )
    return entries


def encoding_class(expected):
    """Top-level A64 encoding group from op0 (bits 28:25)."""
    op0 = (expected >> 25) & 0xF
    if op0 == 0b0000:
        return "SME" if expected >> 31 else "Reserved"
    if op0 == 0b0010:
        return "SVE"
    if op0 in (0b1000, 0b1001):
        return "DP imm"
    if op0 in (0b1010, 0b1011):
        return "Branch"
    if op0 & 0b0101 == 0b0100:
        return "Loads & stores"
    if op0 & 0b0111 == 0b0101:
        return "DP reg"
    if op0 & 0b0111 == 0b0111:
        return "SIMD & FP"
    return "Other"


def tier1_status(entry):
    """Mirror the opcode switch in bal_x86_tier1_compiler.c."""
    name, op, ops = entry["name"], entry["opcode"], entry["operands"]
    kind = [t for t, _, _ in ops]
    if name.startswith(HOST_TRAPPED):
        return RUNS
    if op == "OPCODE_TRAP":
        return TODO
    if op == "OPCODE_MOV":
        if name[3:4] in ("N", "K", "Z") or kind[1] in REG:
            return RUNS
        return DECODED
    if op in ("OPCODE_ADD", "OPCODE_SUB", "OPCODE_CMP"):
        if kind[3] in REG:
            return RUNS
        if kind[2] == IMM and ops[2][2] == 3 and kind[4] in REG:
            return DECODED  # extended register: explicitly unsupported
        return RUNS if kind[2] == IMM else DECODED
    if op == "OPCODE_AND":
        return RUNS if kind[3] in REG else DECODED
    if op in (
        "OPCODE_JUMP",
        "OPCODE_BRANCH_ZERO",
        "OPCODE_BRANCH_NOT_ZERO",
        "OPCODE_BRANCH_CONDITIONAL",
        "OPCODE_RETURN",
    ):
        return RUNS
    return DECODED


def parse_ir_opcodes(types_h, compiler_c):
    enum = re.search(r"typedef enum\s*\{(.*?)\}\s*bal_opcode_t;", types_h.read_text(), re.S)
    names = [n for n in re.findall(r"(OPCODE_\w+)", enum.group(1)) if n != "OPCODE_ENUM_END"]
    lowered = set(re.findall(r"case (OPCODE_\w+):", compiler_c.read_text()))
    return [(n.removeprefix("OPCODE_"), RUNS if n in lowered else TODO) for n in names]


# --- squarified treemap -------------------------------------------------------


def squarify(items, x, y, w, h):
    """items: list of (key, weight) sorted descending. Returns {key: rect}."""
    total = sum(v for _, v in items)
    if not items or total <= 0:
        return {}
    scale = w * h / total
    sized = [(k, v * scale) for k, v in items]
    out = {}

    def worst(row, side):
        s = sum(a for _, a in row)
        return max(max(side * side * a / (s * s), (s * s) / (side * side * a)) for _, a in row)

    while sized:
        side = min(w, h)
        row = [sized.pop(0)]
        while sized and worst(row + [sized[0]], side) <= worst(row, side):
            row.append(sized.pop(0))
        s = sum(a for _, a in row)
        if w >= h:
            cw = s / h
            cy = y
            for k, a in row:
                ch = a / cw
                out[k] = (x, cy, cw, ch)
                cy += ch
            x, w = x + cw, w - cw
        else:
            rh = s / w
            cx = x
            for k, a in row:
                cwid = a / rh
                out[k] = (cx, y, cwid, rh)
                cx += cwid
            y, h = y + rh, h - rh
    return out


def cells(paths, statuses, x, y, w, h):
    """Fill a rect with a grid of one square per item, finished items first."""
    n = len(statuses)
    cols = max(1, round(math.sqrt(n * w / h))) if h > 0 else n
    rows = math.ceil(n / cols)
    cw, ch = w / cols, h / rows
    gap = min(0.5, cw / 5, ch / 5)
    order = {RUNS: 0, DECODED: 1, TODO: 2}
    for i, st in enumerate(sorted(statuses, key=order.get)):
        cx, cy = x + (i % cols) * cw, y + (i // cols) * ch
        paths[st].append(
            f"M{cx + gap:.1f} {cy + gap:.1f}h{cw - 2 * gap:.1f}v{ch - 2 * gap:.1f}h{2 * gap - cw:.1f}z"
        )


def label(svg, text, x, y, size=10, weight="normal"):
    svg.append(
        f'<text x="{x + 3:.1f}" y="{y + size + 1:.1f}" font-size="{size}" font-weight="{weight}" '
        f'class="lbl">{html.escape(text)}</text>'
    )


def pct(done, total):
    return f"{100 * done / total:.2f}%" if total else "0%"


def render(entries, ir_ops, commit):
    W, H = 900, 330
    lw, gap, top = 560, 14, 26
    rw = W - lw - gap
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        "<style>.lbl{fill:#fff;paint-order:stroke;stroke:#000;stroke-width:2.5px;"
        "stroke-linejoin:round}.ttl{fill:#8b949e;font-size:13px;font-weight:600}</style>",
    ]

    # Left panel: every ARM64 encoding in Ballistic's decoder table.
    st = Counter(e["status"] for e in entries)
    svg.append(
        f'<text x="0" y="16" class="ttl">ARM64 instructions on x86: {pct(st[RUNS], len(entries))} '
        f"({st[RUNS]}/{len(entries)})</text>"
    )
    by_class = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_class[e["class"]][e["name"]].append(e["status"])
    class_items = sorted(
        ((c, sum(len(v) for v in m.values())) for c, m in by_class.items()), key=lambda t: -t[1]
    )
    ph = H - top - 22
    paths = defaultdict(list)
    frames = []
    for cls, (cx, cy, cw, ch) in squarify(class_items, 0, top, lw, ph).items():
        mnems = sorted(((k, len(v)) for k, v in by_class[cls].items()), key=lambda t: -t[1])
        hdr = 14 if ch > 40 else 0
        for mn, (mx, my, mw, mh) in squarify(mnems, cx, cy + hdr, cw, ch - hdr).items():
            cells(paths, by_class[cls][mn], mx, my, mw, mh)
            frames.append(
                f'<rect x="{mx:.2f}" y="{my:.2f}" width="{mw:.2f}" height="{mh:.2f}" '
                f'fill="none" stroke="{BORDER}" stroke-width="1"/>'
            )
            if mw > 30 and mh > 15:
                label(frames, mn[: max(2, int(mw / 6.5))], mx, my, 9)
        frames.append(
            f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{ch:.2f}" '
            f'fill="none" stroke="{BORDER}" stroke-width="2.5"/>'
        )
        if hdr:
            n_cls = sum(len(v) for v in by_class[cls].values())
            n_run = sum(s == RUNS for v in by_class[cls].values() for s in v)
            frames.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" width="{cw:.2f}" height="{hdr}" fill="#161b22"/>'
            )
            text = f"{cls}  {n_run}/{n_cls}"
            size = 10 if len(text) * 6.2 < cw else 8.5
            label(frames, text[: int(cw / (size * 0.58))], cx, cy, size, "600")
    svg.append(f'<rect x="0" y="{top}" width="{lw}" height="{ph}" fill="{BORDER}"/>')
    for st, d in paths.items():
        svg.append(f'<path fill="{COLORS[st]}" d="{"".join(d)}"/>')
    svg.extend(frames)

    # Right panel: IR opcodes the x86 backend can lower.
    done = sum(s == RUNS for _, s in ir_ops)
    rx = lw + gap
    svg.append(
        f'<text x="{rx}" y="16" class="ttl">IR opcodes lowered to x86: {pct(done, len(ir_ops))} '
        f"({done}/{len(ir_ops)})</text>"
    )
    cols = 3
    rows = math.ceil(len(ir_ops) / cols)
    cw, ch = rw / cols, ph / rows
    order = {RUNS: 0, TODO: 1}
    for i, (name, s) in enumerate(sorted(ir_ops, key=lambda t: order[t[1]])):
        x, y = rx + (i % cols) * cw, top + (i // cols) * ch
        svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{cw:.2f}" height="{ch:.2f}" '
            f'fill="{COLORS[s]}" stroke="{BORDER}" stroke-width="1.5"/>'
        )
        label(svg, name.replace("_", " ")[: int(cw / 5.4)], x, y + 1, 9)

    # Legend.
    ly = H - 8
    lx = 0
    for key, text in (
        (RUNS, "runs on x86"),
        (DECODED, "decoded to IR, no x86 lowering yet"),
        (TODO, "not started"),
    ):
        svg.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" fill="{COLORS[key]}"/>')
        svg.append(f'<text x="{lx + 14}" y="{ly}" font-size="11" fill="#8b949e">{text}</text>')
        lx += 14 + len(text) * 6 + 18
    svg.append(
        f'<text x="{W}" y="{ly}" font-size="10" fill="#8b949e" text-anchor="end">'
        f"pound-emu/ballistic @ {commit}</text>"
    )
    svg.append("</svg>")
    return "\n".join(svg)


def badge(left, right, color):
    lw, rw = 7 * len(left) + 12, 7 * len(right) + 12
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + rw}" height="20" '
        f'font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">'
        f'<rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{rw}" height="20" fill="{color}"/>'
        f'<text x="{lw / 2}" y="14" fill="#fff" text-anchor="middle">{left}</text>'
        f'<text x="{lw + rw / 2}" y="14" fill="#fff" text-anchor="middle">{right}</text></svg>'
    )


def badge_color(fraction):
    for limit, color in ((0.9, "#4c1"), (0.6, "#97ca00"), (0.3, "#dfb317"), (0.1, "#fe7d37")):
        if fraction >= limit:
            return color
    return "#e05d44"


def update_readme(arm_line):
    block = f"""{MARK_BEGIN}
## Status

![cpu](docs/status/badge_cpu.svg) ![ir](docs/status/badge_ir.svg) ![gpu](docs/status/badge_gpu.svg)

![Status](docs/status/status.svg)

{arm_line}

GPU shader translation (SM86 to SPIR-V) and Horizon OS services are not started yet.
A [GPU command frontend and inspection tool](docs/GPU_DEVELOPMENT.md) are now available;
command decoding is separate from shader translation and rendering.
Regenerate with `python tools/status/generate_status.py --ballistic <path-to-ballistic-checkout>`.
{MARK_END}"""
    text = README.read_text(encoding="utf-8")
    if MARK_BEGIN in text:
        text = re.sub(re.escape(MARK_BEGIN) + ".*?" + re.escape(MARK_END), lambda _: block, text, flags=re.S)
    else:
        text = text.replace("\n## Development", "\n" + block + "\n\n## Development", 1)
    README.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ballistic", required=True, type=Path, help="Ballistic source checkout")
    ap.add_argument("--no-readme", action="store_true", help="only write the SVGs")
    args = ap.parse_args()
    b = args.ballistic

    entries = parse_decoder_table(b / "src/generated/decoder_table.c")
    if not entries:
        raise SystemExit("No decoder entries parsed; did the table format change?")
    for e in entries:
        e["class"] = encoding_class(e["expected"])
        e["status"] = tier1_status(e)
    ir_ops = parse_ir_opcodes(
        b / "include/bal_types.h", b / "src/backend/x86/bal_x86_tier1_compiler.c"
    )
    commit = subprocess.run(
        ["git", "-C", str(b), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "status.svg").write_text(render(entries, ir_ops, commit), encoding="utf-8")

    st = Counter(e["status"] for e in entries)
    n = len(entries)
    ir_done = sum(s == RUNS for _, s in ir_ops)
    for fname, left, frac in (
        ("badge_cpu.svg", "ARM64", st[RUNS] / n),
        ("badge_ir.svg", "IR to x86", ir_done / len(ir_ops)),
        ("badge_gpu.svg", "shaders", 0.0),
    ):
        (OUT_DIR / fname).write_text(badge(left, f"{100 * frac:.2f}%", badge_color(frac)), encoding="utf-8")

    arm_line = (
        f"* ARM64: share of the {n} A64 encodings in Ballistic's decoder table that its x86 tier-1 "
        f"compiler can run ({st[RUNS]} run, {st[DECODED]} decoded only, {st[TODO]} not started). "
        f"Ballistic commit `{commit}`."
    )
    if not args.no_readme:
        update_readme(arm_line)
    print(f"ARM64: {st[RUNS]} run / {st[DECODED]} decoded / {st[TODO]} todo of {n}")
    print(f"IR: {ir_done}/{len(ir_ops)} lowered")
    for cls, c in sorted(Counter(e["class"] for e in entries).items(), key=lambda t: -t[1]):
        print(f"  {cls}: {c}")


if __name__ == "__main__":
    main()
