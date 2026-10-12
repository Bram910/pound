#!/usr/bin/env python3
"""Render the fork's GPU checkpoint without measuring or changing Ballistic."""

import argparse
import html
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "status"
MARK_BEGIN = "<!-- POUND-GPU-STATUS:BEGIN -->"
MARK_END = "<!-- POUND-GPU-STATUS:END -->"


def load_status():
    status = json.loads((ROOT / "tools/status/gpu_status.json").read_text(encoding="utf-8"))
    date.fromisoformat(status["updated"])
    if not re.fullmatch(r"[0-9a-f]{40}", status["checkpoint"]):
        raise ValueError("GPU checkpoint must be a full Git commit hash")
    header = (ROOT / "src/core/gpu/gpu_shader.h").read_text(encoding="utf-8")
    operations = re.search(r"typedef enum\s*\{(.*?)\}\s*gpu_shader_operation_t;", header, re.S)
    families = re.findall(r"GPU_SHADER_(\w+)", operations.group(1))
    if status["families"] != families:
        raise ValueError("GPU checkpoint families differ from the shader operation enum")
    bounded_only = status["bounded_only"]
    if len(set(bounded_only)) != len(bounded_only) or not set(bounded_only) <= set(families):
        raise ValueError("Invalid bounded-only families")
    for check in status["validation"]:
        expected = check["programs"] * check["lanes_per_program"] * max(1, len(check["budgets"]))
        if check["lane_states"] != expected or check["mismatches"] < 0 or not check["backends"]:
            raise ValueError("Inconsistent GPU validation counts")
    return status


def text(svg, content, position_x, position_y, size=14, color="#e6edf3", weight="400"):
    svg.append(
        f'<text x="{position_x}" y="{position_y}" font-size="{size}" '
        f'fill="{color}" font-weight="{weight}">{html.escape(str(content))}</text>'
    )


def rectangle(svg, position_x, position_y, width, height, color="#161b22", border="#30363d"):
    svg.append(
        f'<rect x="{position_x}" y="{position_y}" width="{width}" height="{height}" '
        f'rx="8" fill="{color}" stroke="{border}"/>'
    )


def render_svg(status):
    families = status["families"]
    fast_count = len(families) - len(status["bounded_only"])
    straight, bounded = status["validation"]
    fixtures = status["control_fixtures"]
    mismatches = sum(check["mismatches"] for check in status["validation"]) + fixtures["mismatches"]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="714" viewBox="0 0 900 714" '
        'role="img" aria-labelledby="title description" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
        '<title id="title">Pound GPU development checkpoint</title>',
        '<desc id="description">Limited SM86 integer compute prototype. '
        'Family counts refer only to the decoded subset, not all GPU instructions. '
        'Game rendering is not implemented.</desc>',
    ]
    rectangle(svg, 1, 1, 898, 712, "#0d1117")
    text(svg, "POUND / GPU PROGRESS", 24, 37, 22, weight="700")
    text(svg, f'{status["updated"]}  |  {status["checkpoint"][:8]}', 635, 35, 13, "#8b949e")
    text(svg, "SM86 integer compute prototype - not game rendering", 24, 65, 16, "#d29922")
    for index, (value, label) in enumerate((
        (len(families), "Decoded families"),
        (f"{fast_count}/{len(families)}", "Straight-line SPIR-V"),
        (f"{len(families)}/{len(families)}", "Bounded SPIR-V"),
        (mismatches, "Recorded mismatches"),
    )):
        position_x = 24 + index * 216
        rectangle(svg, position_x, 84, 204, 84)
        text(svg, value, position_x + 14, 123, 28, "#3fb950", "700")
        text(svg, label, position_x + 14, 149, 13, "#8b949e")
    text(svg, "CURRENTLY DECODED SUBSET", 24, 200, 13, "#8b949e", "700")
    text(svg, "Green: both paths   /   Amber: bounded path only   /   Not full SM86 coverage", 24, 225, 13)
    for index, family in enumerate(families):
        position_x = 24 + index % 11 * 78
        position_y = 242 + index // 11 * 40
        bounded_only = family in status["bounded_only"]
        rectangle(svg, position_x, position_y, 72, 32,
                  "#3b2e12" if bounded_only else "#12301d",
                  "#d29922" if bounded_only else "#238636")
        text(svg, family, position_x + 8, position_y + 21, 12,
             "#e3b341" if bounded_only else "#7ee787", "600")
    text(svg, "VALIDATION AT THIS CHECKPOINT", 24, 346, 13, "#8b949e", "700")
    for index, (check, subtitle) in enumerate((
        (straight, "RTX 3080 / straight-line"),
        (bounded, "Each: RTX 3080 + software Vulkan"),
    )):
        position_x = 24 + index * 432
        rectangle(svg, position_x, 360, 420, 101)
        text(svg, f'{check["lane_states"]:,} full lane states', position_x + 14, 390, 21, weight="600")
        text(svg, subtitle, position_x + 14, 414, 13)
        detail = f'{check["programs"]} programs / {check["words_per_lane"]} words per lane'
        if check["budgets"]:
            detail += f' / {len(check["budgets"])} budgets'
        text(svg, detail, position_x + 14, 440, 12, "#8b949e")
    text(svg, "spirv-val before every dispatch; complete state compared against CPU reference.", 24, 483, 13)
    text(svg, f'{fixtures["programs"]} independent control fixtures / {fixtures["lane_states"]} states per backend; '
         f'resumption and budget {fixtures["maximum_budget"]:,}.', 24, 505, 13)
    text(svg, "INTEGRATION MILESTONES", 24, 538, 13, "#8b949e", "700")
    for position_y, label, state, color in (
        (561, "Mappings, DMA, tiled transfers, uploads, queues, semaphores", "Software prototype", "#58a6ff"),
        (587, "Integer decode, prepared programs, GPU branch/loop emission", "Verified subset", "#3fb950"),
        (613, "Guest OS submission, graphics draws, textures and presentation", "Not implemented", "#8b949e"),
    ):
        svg.append(f'<circle cx="29" cy="{position_y - 4}" r="4" fill="{color}"/>')
        text(svg, label, 42, position_y, 13)
        text(svg, state, 706, position_y, 13, color)
    text(svg, "Still missing: FP/SIMD, shader headers, graphics stages, warp/shared-memory synchronization.",
         24, 645, 13, "#8b949e")
    text(svg, "Synthetic independent-lane checks. Counts are not cumulative, game FPS or overall GPU completion.",
         24, 677, 13, "#d29922")
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def readme_section(status):
    straight, bounded = status["validation"]
    return f"""{MARK_BEGIN}
### GPU progress

[![GPU prototype progress](docs/status/gpu.svg)](docs/status/GPU_STATUS.md)

**[GPU status details](docs/status/GPU_STATUS.md)** - checkpoint `{status['checkpoint'][:8]}` ({status['updated']}).
The decoded integer prototype covers {len(status['families'])} SM86 families: {len(status['families']) - len(status['bounded_only'])} in the straight-line SPIR-V path
and all {len(status['families'])} in the bounded branch/loop path. At this checkpoint,
{straight['lane_states']:,} complete straight-line lane states pass on RTX 3080; {bounded['lane_states']:,} bounded states pass
**on each of RTX 3080 and software Vulkan**. These synthetic checks are not an
overall GPU completion percentage or evidence that games render.

[GPU development details](docs/GPU_DEVELOPMENT.md) cover the software DMA,
tiled transfers, inline uploads, mappings, queues, cooperative scheduling and
semaphore prototypes. Guest OS submission and production rendering remain
unimplemented. CPU status above is a separate Ballistic snapshot and is not
refreshed by this GPU update.

Regenerate GPU visuals with `python tools/status/generate_gpu_status.py`.
{MARK_END}"""


def render_markdown(status):
    family_rows = "\n".join(
        f"| `{family}` | {'Not supported' if family in status['bounded_only'] else 'Prototype'} | Prototype |"
        for family in status["families"]
    )
    validation_rows = "\n".join(
        f"| {check['mode']} | {check['programs']} | {', '.join(map(str, check['budgets'])) or 'N/A'} | "
        f"{check['lane_states']:,} | {check['words_per_lane']} | {', '.join(check['backends'])} | {check['mismatches']} |"
        for check in status["validation"]
    )
    fixtures = status["control_fixtures"]
    return f"""# GPU development status

![GPU prototype progress](gpu.svg)

Updated **{status['updated']}** for [GPU checkpoint `{status['checkpoint'][:8]}`](https://github.com/Bram910/pound/commit/{status['checkpoint']}).
[Back to the fork overview](../../README.md#status) | [GPU architecture and tools](../GPU_DEVELOPMENT.md)

## Scope

This is a **limited SM86 integer compute prototype**, not a working Switch 2
graphics backend. The counts below cover only the {len(status['families'])} families currently
decoded by Pound, not the entire SM86 instruction set or every encoding and
modifier within a family. There is no defensible overall GPU completion
percentage yet. Ballistic CPU progress is separate and unchanged here.

## Instruction paths

| Family | Straight-line SPIR-V | Bounded branch/loop SPIR-V |
| --- | --- | --- |
{family_rows}

The straight-line path requires a final unpredicated EXIT. The bounded path
adds BRA, guarded/early exits, loops, faults and resumable instruction budgets.
Both use independent lane states; neither models warp reconvergence or
inter-lane/shared-memory synchronization. See the [buffer ABIs and limits](../GPU_DEVELOPMENT.md#experimental-integer-spir-v-emission)
before using emitted modules.

## Recorded validation

These are developer-checkpoint results, not freshly rerun measurements on
every status regeneration. Generated modules passed Khronos `spirv-val`
before dispatch. Full output states were compared with Pound's CPU reference.

| Path | Programs | Budgets | Lane states per backend | Words per state | Backends checked | Mismatches |
| --- | ---: | --- | ---: | ---: | --- | ---: |
{validation_rows}

The bounded counts are **per backend**, not a combined or cumulative total.
Inputs cover all 128 predicate masks, register aliases, integer boundaries,
mixed arithmetic, branch graphs, interior PCs and fault-state preservation.
The straight-line check covers arithmetic/bit/compare modes, including all
256 LOP3 truth tables and 65,536 indexed PRMT selectors.

Additionally, **{fixtures['programs']} independent control fixtures / {fixtures['lane_states']} lane states per backend**
pass on RTX 3080 and software Vulkan, including chained resumption, invalid
state, overflow, falloff and the maximum instruction budget of {fixtures['maximum_budget']:,}.
The fast path's software Vulkan validation was recorded at the preceding
21-family checkpoint; the table lists only backends checked at this checkpoint.
These are synthetic checks, not game compatibility, frame rates or benchmarks.

## Integration

| Layer | Status |
| --- | --- |
| GPU address mappings and permission checks | Software prototype |
| DMA copies, tiled/3D transfers, remapping and fills | Software prototype |
| Inline uploads, pushbuffer decoding and bounded GPFIFO queues | Software prototype |
| Cooperative multi-channel scheduling, barriers and semaphores | Software prototype |
| Integer decoder, CPU reference, prepared programs and CFG analysis | Implemented for the limited subset |
| Integer SPIR-V, bounded branches/loops and resumable state | Verified independent-lane subset |
| Guest OS GPU services/submission and shader headers | Not implemented |
| Floating-point/SIMD shader coverage, textures and graphics stages | Not implemented |
| Production Vulkan resources, graphics draws and presentation | Not implemented |
| Warp reconvergence, shared memory and inter-lane synchronization | Not implemented |

## Reproduction and maintenance

[`gpu_status.json`](../../tools/status/gpu_status.json) records the checkpoint
and counts; the generator cross-checks its family list against the source enum.
Private Vulkan differential harnesses produced the bulk measurements above.
Public synthetic fixtures can be generated with:

```sh
python tools/gpu/make_shader_spirv_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_control_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
```

The demo scripts emit modules and expected state buffers and optionally run
`spirv-val`; **they do not create a Vulkan device or dispatch GPU work**.
Dispatching the fixtures requires an external compute harness respecting the
documented 256-word straight-line or 261-word bounded buffer ABI.

After validating new GPU work, update the ledger, regenerate and check:

```sh
python tools/status/generate_gpu_status.py
python tools/status/generate_gpu_status.py --check
```

GPU regeneration updates this page, `gpu.svg` and only the GPU subsection of
the README. It does not read Ballistic or overwrite the CPU treemap/badges.
"""


def update_readme(contents, status):
    pattern = re.escape(MARK_BEGIN) + ".*?" + re.escape(MARK_END)
    if len(re.findall(pattern, contents, re.S)) != 1:
        raise ValueError("README must contain exactly one GPU status block")
    return re.sub(pattern, lambda match: readme_section(status), contents, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--check", action="store_true", help="fail if generated status is stale; do not write")
    args = parser.parse_args()
    try:
        status = load_status()
        outputs = {
            args.output_dir / "gpu.svg": render_svg(status),
            args.output_dir / "GPU_STATUS.md": render_markdown(status),
        }
        if not args.no_readme:
            readme = ROOT / "README.md"
            outputs[readme] = update_readme(readme.read_text(encoding="utf-8"), status)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    stale = []
    for path, contents in outputs.items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != contents:
                stale.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
    if stale:
        raise SystemExit("Stale GPU status:\n" + "\n".join(stale))
    print(f"GPU status {'checked' if args.check else 'generated'} at {status['checkpoint'][:8]}; CPU tracker untouched.")


if __name__ == "__main__":
    main()
