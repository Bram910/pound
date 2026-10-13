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
    runner = status["runner_validation"]
    if (runner["lane_states_per_backend"] <= 0 or runner["dispatches_per_backend"] <= 0
            or runner["mismatches"] < 0 or not runner["backends"]):
        raise ValueError("Inconsistent Vulkan runner validation counts")
    session = status["session_validation"]
    if (session["lane_states_per_backend"] <= 0 or session["dispatches_per_backend"] <= 0
            or session["kernels_per_session"] <= 0 or session["mismatches"] < 0):
        raise ValueError("Inconsistent reusable-session validation counts")
    for name in ("float_validation", "arithmetic_validation", "fused_validation",
                 "conversion_validation", "rounding_validation", "fast_conversion_validation",
                 "half_conversion_validation", "widening_validation"):
        floating = status[name]
        if (floating["hardware_results"] != floating["hardware_variants"] * floating["hardware_cases_per_variant"]
                or floating["lane_states_per_backend"] != floating["lanes_per_dispatch"] * floating["dispatches_per_backend"]
                or floating["hardware_variants"] <= 0 or floating["hardware_cases_per_variant"] <= 0
                or floating["dispatches_per_backend"] <= 0 or floating["lanes_per_dispatch"] <= 0
                or floating["independent_lanes_per_path"] <= 0 or floating["independent_paths"] != 2
                or floating["mismatches"] < 0 or not floating["backends"]):
            raise ValueError("Inconsistent floating-point validation counts")
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
    runner = status["runner_validation"]
    session = status["session_validation"]
    floating = status["float_validation"]
    arithmetic = status["arithmetic_validation"]
    fused = status["fused_validation"]
    conversion = status["conversion_validation"]
    rounding = status["rounding_validation"]
    fast_conversion = status["fast_conversion_validation"]
    half_conversion = status["half_conversion_validation"]
    mismatches = (sum(check["mismatches"] for check in status["validation"])
                  + fixtures["mismatches"] + runner["mismatches"] + session["mismatches"]
                  + floating["mismatches"] + arithmetic["mismatches"] + fused["mismatches"]
                  + conversion["mismatches"] + rounding["mismatches"] + fast_conversion["mismatches"]
                  + half_conversion["mismatches"] + status["widening_validation"]["mismatches"])
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1516" viewBox="0 0 900 1516" '
        'role="img" aria-labelledby="title description" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
        '<title id="title">Pound GPU development checkpoint</title>',
        '<desc id="description">Limited SM86 compute prototype with FP32 arithmetic, integral rounding and scalar conversions. '
        'Family counts refer only to the decoded subset, not all GPU instructions. '
        'Game rendering is not implemented.</desc>',
    ]
    rectangle(svg, 1, 1, 898, 1514, "#0d1117")
    text(svg, "POUND / GPU PROGRESS", 24, 37, 22, weight="700")
    text(svg, f'{status["updated"]}  |  integer baseline {status["checkpoint"][:8]}', 565, 35, 13, "#8b949e")
    text(svg, "SM86 compute prototype / FP32 arithmetic, rounding and scalar conversions", 24, 65, 16, "#d29922")
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
    text(svg, f'INTEGER EMITTER CHECKPOINT @ {status["checkpoint"][:8]} (22-FAMILY SCOPE)',
         24, 386, 13, "#8b949e", "700")
    for index, (check, subtitle) in enumerate((
        (straight, "RTX 3080 / straight-line"),
        (bounded, "Each: RTX 3080 + software Vulkan"),
    )):
        position_x = 24 + index * 432
        rectangle(svg, position_x, 400, 420, 101)
        text(svg, f'{check["lane_states"]:,} full lane states', position_x + 14, 430, 21, weight="600")
        text(svg, subtitle, position_x + 14, 454, 13)
        detail = f'{check["programs"]} programs / {check["words_per_lane"]} words per lane'
        if check["budgets"]:
            detail += f' / {len(check["budgets"])} budgets'
        text(svg, detail, position_x + 14, 480, 12, "#8b949e")
    text(svg, "spirv-val before every dispatch; complete state compared against CPU reference.", 24, 523, 13)
    text(svg, f'{fixtures["programs"]} independent control fixtures / {fixtures["lane_states"]} states per backend; '
         f'resumption and budget {fixtures["maximum_budget"]:,}.', 24, 545, 13)
    text(svg, f'Public runner: {runner["lane_states_per_backend"]:,} states per backend / '
         f'{runner["dispatches_per_backend"]} dispatches / {runner["simulated_vulkan_failures"]} simulated host failures.',
         24, 571, 13, "#58a6ff")
    text(svg, f'Reused sessions: {session["lane_states_per_backend"]:,} states per backend / '
         f'{session["dispatches_per_backend"]} dispatches / {session["kernels_per_session"]} alternating kernels.',
         24, 596, 13, "#58a6ff")
    text(svg, "FP32 SHADER CHECKS", 24, 632, 13, "#8b949e", "700")
    text(svg, "Vulkan states: per backend (RTX 3080 + software)", 456, 632, 13, "#8b949e")
    for index, (check, label, independent) in enumerate((
        (floating, "COMPARISON + MIN/MAX", "independent edge states"),
        (arithmetic, "ADDITION + MULTIPLICATION", "exact edge states"),
        (fused, "FUSED MULTIPLY-ADD", "exact edge states"),
        (conversion, "FP32 / INTEGER CONVERSIONS", "exact edge states"),
        (rounding, "FP32 INTEGRAL ROUNDING", "exact edge states"),
        (fast_conversion, "I2FP / 32-BIT TO FP32", "exact edge states"),
        (half_conversion, "FP32 TO FP16 CONVERSION", "exact edge states"),
        (status["widening_validation"], "FP16 TO FP32 WIDENING", "binary16 oracle states"),
    )):
        position_x = 24 + index % 2 * 432
        position_y = 646 + index // 2 * 166
        rectangle(svg, position_x, position_y, 420, 154)
        text(svg, label, position_x + 14, position_y + 22, 13, "#8b949e", "700")
        text(svg, f'{check["hardware_results"]:,}', position_x + 14, position_y + 54, 24, "#7ee787", "600")
        text(svg, "native SM86 results", position_x + 14, position_y + 76, 12)
        text(svg, f'{check["lane_states_per_backend"]:,} full states / '
             f'{check["dispatches_per_backend"]} dispatches', position_x + 14, position_y + 99, 12)
        text(svg, f'{check["independent_lanes_per_path"]:,} {independent} per path',
             position_x + 14, position_y + 125, 12, "#8b949e")
    text(svg, "INTEGRATION MILESTONES", 24, 1322, 13, "#8b949e", "700")
    for position_y, label, state, color in (
        (1345, "Mappings, DMA, tiled transfers, uploads, queues, semaphores", "Software prototype", "#58a6ff"),
        (1371, "Integer + FP32 arithmetic/conversions, GPU branches/loops", "Verified subset", "#3fb950"),
        (1397, "Vulkan sessions, compiled kernels and resumable lane buffers", "Verified prototype", "#3fb950"),
        (1423, "Guest OS submission, graphics draws, textures and presentation", "Not implemented", "#8b949e"),
    ):
        svg.append(f'<circle cx="29" cy="{position_y - 4}" r="4" fill="{color}"/>')
        text(svg, label, 42, position_y, 13)
        text(svg, state, 706, position_y, 13, color)
    text(svg, "Still missing: FP16 arithmetic, FP64, other conversions, headers, graphics and warp/shared-memory sync.",
         24, 1457, 13, "#8b949e")
    text(svg, "Synthetic independent-lane checks. Counts are not cumulative, game FPS or overall GPU completion.",
         24, 1493, 13, "#d29922")
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def readme_section(status):
    straight, bounded = status["validation"]
    runner = status["runner_validation"]
    session = status["session_validation"]
    floating = status["float_validation"]
    arithmetic = status["arithmetic_validation"]
    fused = status["fused_validation"]
    conversion = status["conversion_validation"]
    rounding = status["rounding_validation"]
    fast_conversion = status["fast_conversion_validation"]
    half_conversion = status["half_conversion_validation"]
    return f"""{MARK_BEGIN}
### GPU progress

[![GPU prototype progress](docs/status/gpu.svg)](docs/status/GPU_STATUS.md)

**[GPU status details](docs/status/GPU_STATUS.md)** - updated {status['updated']}.
The decoded prototype covers {len(status['families'])} SM86 families: {len(status['families']) - len(status['bounded_only'])} in the straight-line SPIR-V path
and all {len(status['families'])} in the bounded branch/loop path. At the earlier integer
emitter checkpoint `{status['checkpoint'][:8]}` (22-family scope),
{straight['lane_states']:,} complete straight-line lane states pass on RTX 3080; {bounded['lane_states']:,} bounded states pass
**on each of RTX 3080 and software Vulkan**. These synthetic checks are not an
overall GPU completion percentage or evidence that games render.

**New FP32 comparison/min-max support:** FSETP and FMNMX pass
{floating['hardware_results']:,} native SM86 result comparisons and
{floating['lane_states_per_backend']:,} full Vulkan lane states per backend.

**FP32 addition/multiplication:** FADD and FMUL pass
{arithmetic['hardware_results']:,} native result checks and
{arithmetic['lane_states_per_backend']:,} complete Vulkan states per backend,
including all four rounding modes, FTZ and saturation.

**Fused FP32 multiply-add:** FFMA passes {fused['hardware_results']:,} native result checks
and {fused['lane_states_per_backend']:,} complete Vulkan states per backend. The new checks also
correct a rounding-dependent underflow boundary shared by FMUL and FFMA.
**FP32 / 32-bit integer conversions:** F2I and I2F pass
{conversion['hardware_results']:,} native result checks and
{conversion['lane_states_per_backend']:,} complete Vulkan states per backend.
Signed/unsigned results, four rounding modes and F2I FTZ/NaN handling are covered.

**FP32 integral rounding:** FRND passes {rounding['hardware_results']:,} native results
and {rounding['lane_states_per_backend']:,} complete Vulkan states per backend, including
nearest-even, floor, ceil, truncation, FTZ, abs/neg and signed-zero behavior.

**I2FP conversion encoding:** signed/unsigned 32-bit to FP32 with RN/RZ passes
{fast_conversion['hardware_results']:,} native results and {fast_conversion['lane_states_per_backend']:,}
complete Vulkan states per backend. Other I2FP sizes/modes reject explicitly.

**Scalar FP32 to FP16 conversion:** F2F.F16.F32 passes {half_conversion['hardware_results']:,}
native results and {half_conversion['lane_states_per_backend']:,} complete Vulkan states per backend.
Four rounding modes, FTZ, subnormals, overflow and NaNs are covered.
**Scalar FP16 to FP32 widening:** F2F.F32.F16 passes {status['widening_validation']['hardware_results']:,}
native comparisons covering every binary16 bit pattern and {status['widening_validation']['lane_states_per_backend']:,}
complete Vulkan states per backend. Register sources, FTZ and abs/neg are verified.
FP16 arithmetic, packed conversions and graphics rendering remain unimplemented.

The optional **PoundGpuCompute** tool now runs the supported subset through
Vulkan, including resumable bounded state. Its separate runner checks pass
{runner['lane_states_per_backend']:,} full states across {runner['dispatches_per_backend']} dispatches **per backend**.
Reusable sessions and compiled kernels avoid rebuilding the device/pipeline
for each dispatch; another {session['lane_states_per_backend']:,} states pass per backend while alternating
{session['kernels_per_session']} kernels in one session. These remain compute-only development tools.

[GPU development details](docs/GPU_DEVELOPMENT.md) cover the software DMA,
tiled transfers, inline uploads, mappings, queues, cooperative scheduling and
semaphore prototypes. Guest OS submission and production rendering remain
unimplemented. CPU status above is a separate Ballistic snapshot and is not
refreshed by this GPU update.

Regenerate GPU visuals with `python tools/status/generate_gpu_status.py`.
{MARK_END}"""


def render_markdown(status):
    runner = status["runner_validation"]
    session = status["session_validation"]
    floating = status["float_validation"]
    arithmetic = status["arithmetic_validation"]
    fused = status["fused_validation"]
    conversion = status["conversion_validation"]
    rounding = status["rounding_validation"]
    fast_conversion = status["fast_conversion_validation"]
    half_conversion = status["half_conversion_validation"]
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

Updated **{status['updated']}**. Earlier integer-emitter validation checkpoint (22-family scope):
[`{status['checkpoint'][:8]}`](https://github.com/Bram910/pound/commit/{status['checkpoint']}).
[Back to the fork overview](../../README.md#status) | [GPU architecture and tools](../GPU_DEVELOPMENT.md)

## Scope

This is a **limited SM86 compute prototype with FP32 arithmetic, rounding and scalar conversions**, not a working Switch 2
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

The table below records the earlier **22-family integer checkpoint**, before
FSETP/FMNMX were added. These are developer-checkpoint results, not freshly rerun measurements on
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

### FP32 comparison and min/max validation

FSETP supports ordered/unordered comparisons, NUM/NAN, F/T, AND/OR/XOR,
dual predicate outputs and register/immediate sources. FMNMX selects min/max
with a predicate. Both support FTZ and register abs/neg modifiers. Signed
zero, infinities, quiet/signaling NaNs and subnormal values are handled using
integer bit operations in the CPU reference and SPIR-V, without relying on
the host floating-point mode.

**{floating['hardware_results']:,} native SM86 results** on {floating['hardware_device']}
match the CPU reference across {floating['hardware_variants']:,} encodings and
{floating['hardware_cases_per_variant']:,} inputs each. Separately,
**{floating['lane_states_per_backend']:,} complete lane states / {floating['dispatches_per_backend']} Vulkan dispatches per backend**
pass across {floating['programs']} programs, including branch graphs, predicate/register aliases,
interior PCs, faults and instruction-budget exhaustion. Both compiler outputs
pass Khronos `spirv-val`. These checks use RTX 3080 and software Vulkan.

The public floating-point demo checks **{floating['independent_lanes_per_path']} independently specified edge states
for each of the {floating['independent_paths']} paths on each backend**, plus {floating['rejections']} unsupported-emission
rejections with output preservation. **Recorded mismatches: {floating['mismatches']}.**
These measurements cover FP32 comparison/min-max only. Arithmetic checks are
recorded separately below.

### FP32 addition and multiplication validation

FADD and FMUL now support all four rounding modes, FTZ, saturation,
register/immediate inputs, abs/neg modifiers, execution guards and register aliases.
Integer significands and paired multiplication preserve subnormal behavior
without requiring shaderInt64 or host floating-point controls. Nondefault
FMUL scaling and DNZ are rejected. The corrected, rounding-dependent SM86
underflow boundary is included in the edge checks.

**{arithmetic['hardware_results']:,} native SM86 results** match across
{arithmetic['hardware_variants']:,} encodings and {arithmetic['hardware_cases_per_variant']:,} inputs each.
Separately, **{arithmetic['lane_states_per_backend']:,} complete states / {arithmetic['dispatches_per_backend']} dispatches per backend**
pass across {arithmetic['programs']} programs with budgets 1/3/16/97, aliases,
all predicate masks, branch graphs, interior PCs and fault preservation.
Both backends run Khronos-validated modules. Separate host API checks pass
maximum-size emission and ASan/UBSan. **Recorded mismatches: {arithmetic['mismatches']}.**

The public arithmetic demo checks **{arithmetic['independent_lanes_per_path']:,} exact rational edge states
per path per backend**, {arithmetic['rejections']} output-preserving rejection cases and a
{arithmetic['persistent_chain_lanes']}-lane loop resumed through {arithmetic['persistent_chain_dispatches']} dispatches using one compiled kernel.
Other conversion sizes, FP16 arithmetic, FP64 and texture/graphics operations remain missing.

### Fused FP32 multiply-add validation

FFMA preserves the product through the addition and rounds once to FP32.
Register, immediate-multiplicand and immediate-addend forms support all four
rounding modes, FTZ, saturation, abs/neg modifiers and execution guards.
The CPU reference uses a wide integer significand; SPIR-V uses paired
32-bit arithmetic without shaderInt64. DNZ, constant/uniform sources and
unknown semantic fields are rejected explicitly.

**{fused['hardware_results']:,} native SM86 results** match across
{fused['hardware_variants']:,} encodings and {fused['hardware_cases_per_variant']:,} inputs each.
Separately, **{fused['lane_states_per_backend']:,} complete states / {fused['dispatches_per_backend']} dispatches per backend**
pass across {fused['programs']} stratified programs with budgets {', '.join(map(str, fused['budgets']))},
aliases, all predicate masks, FSETP-driven branches, interior PCs and fault
preservation. Modules pass Khronos validation on RTX 3080 and software Vulkan;
maximum-size emission and API preservation pass ASan/UBSan.

The public fused demo checks **{fused['independent_lanes_per_path']} exact rational states per path
per backend** and {fused['rejections']} output-preserving rejections, including cancellation
that differs from separate multiply/add, intermediate product overflow,
signed zeros, NaNs and rounding-boundary sweeps. These checks also fix the
FMUL/FFMA FTZ cutoff near the smallest normal result. **Recorded mismatches: {fused['mismatches']}.**
The demo also checks a {fused['persistent_chain_lanes']}-lane fused-accumulation loop through
{fused['persistent_chain_dispatches']} dispatches with one compiled kernel and an aliased accumulator.

### FP32 / 32-bit integer conversion validation

F2I and I2F support signed/unsigned 32-bit integer types, all four rounding
modes, register/immediate sources, execution guards and register aliases.
F2I also supports FTZ and register abs/neg modifiers. Finite and infinite
float-to-integer results clamp to the destination range. NTZ converts NaNs
to zero; without NTZ, raw SM86 returns `0x80000000` for either integer type.
The CPU and SPIR-V use integer bit operations, not undefined host C casts
or Vulkan's implementation-dependent out-of-range float-to-integer conversion.
Other widths, constant/uniform sources and I2F source modifiers are rejected.

**{conversion['hardware_results']:,} native SM86 results** match across
{conversion['hardware_variants']:,} encodings and {conversion['hardware_cases_per_variant']:,} inputs each.
Separately, **{conversion['lane_states_per_backend']:,} complete states / {conversion['dispatches_per_backend']} dispatches per backend**
pass across {conversion['programs']} programs with budgets {', '.join(map(str, conversion['budgets']))},
aliases, RZ/PT, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public conversion demo checks **{conversion['independent_lanes_per_path']} exact rational states per path
per backend**, {conversion['rejections']} output-preserving rejections, and a
{conversion['persistent_chain_lanes']}-lane conversion/arithmetic loop through
{conversion['persistent_chain_dispatches']} dispatches with one compiled kernel.
**Recorded mismatches: {conversion['mismatches']}.** These counts cover FP32/32-bit conversions only.

### FP32 integral-rounding validation

FRND rounds an FP32 input to an integral value stored in FP32, without the
32-bit integer range clamp of F2I. Nearest-even, floor, ceil and truncation
support register/immediate sources, FTZ, abs/neg, guards, RZ/PT and aliases.
Signed zeros and infinities are preserved; NaNs produce the native SM86
canonical `0x7fffffff`. Large finite FP32 integers remain unchanged.
The CPU reference and SPIR-V use integer bit operations and bounded shifts,
without host FP controls, shaderInt64 or float-rounding capabilities.
Other widths, constant/uniform sources and unchecked fields are rejected.

**{rounding['hardware_results']:,} native SM86 results** match across
{rounding['hardware_variants']:,} encodings and {rounding['hardware_cases_per_variant']:,} inputs each.
Separately, **{rounding['lane_states_per_backend']:,} complete states / {rounding['dispatches_per_backend']} dispatches per backend**
pass across {rounding['programs']} programs with budgets {', '.join(map(str, rounding['budgets']))},
aliases, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public rounding demo checks **{rounding['independent_lanes_per_path']} exact rational states per path
per backend**, {rounding['rejections']} output-preserving rejections and a
{rounding['persistent_chain_lanes']}-lane aliased rounding/arithmetic loop through
{rounding['persistent_chain_dispatches']} dispatches with one compiled kernel.
Maximum-sized modules, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: {rounding['mismatches']}.** These counts cover FP32 FRND only.

### I2FP FP32/32-bit conversion validation

The separate I2FP encoding now supports signed/unsigned 32-bit integer inputs
and FP32 results with nearest-even (RN) and toward-zero (RZ) rounding.
Register/immediate sources, guards, RZ/PT and source/destination aliases
use the existing integer-only I2F reference and SPIR-V implementation.
This includes the conversion encoding emitted by the native SM86 PTX fixture
for `cvt.rn.f32.s32` and `cvt.rz.f32.u32`.
Other widths, rounding modes, source modifiers and constant/uniform sources
are rejected. In particular, RM/RP are not silently coerced to RN/RZ.
Supporting this encoding is a coverage improvement, not a measured speedup.

The tests also correct a software Vulkan predicated-zero-write issue shared
by MOV and I2F: selecting zero over `0xbf800000` could yield `0x80000000`
on llvmpipe/LLVM 15. Register commits now use a bit-exact mask/XOR merge,
including wide destinations. Public fixtures check zero overwrites and
inactive-write preservation for MOV, I2F and I2FP across predicate masks.

**{fast_conversion['hardware_results']:,} native SM86 results** match across
{fast_conversion['hardware_variants']:,} encodings and {fast_conversion['hardware_cases_per_variant']:,} inputs each,
including register aliases and RZ. Separately,
**{fast_conversion['lane_states_per_backend']:,} complete states / {fast_conversion['dispatches_per_backend']} dispatches per backend**
pass across {fast_conversion['programs']} programs with budgets {', '.join(map(str, fast_conversion['budgets']))},
all predicate masks, aliases, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public I2FP demo checks **{fast_conversion['independent_lanes_per_path']} exact rational states per path
per backend**, including paired I2F/I2FP encodings, {fast_conversion['rejections']}
output-preserving rejections, and a {fast_conversion['persistent_chain_lanes']}-lane
conversion/arithmetic loop through {fast_conversion['persistent_chain_dispatches']} dispatches using one kernel.
Maximum-size emission, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: {fast_conversion['mismatches']}.** This covers only the FP32/32-bit RN/RZ I2FP subset.

### Scalar FP32 to FP16 conversion validation

The F16.F32 direction converts an FP32 source into
an FP16 bit pattern in the low 16 bits of a zero-extended 32-bit register.
Nearest-even, down, up and toward-zero rounding support register/immediate
sources, abs/neg, guards, RZ/PT and aliases. NaNs produce `0x7fff`;
infinities and signed zeros retain their sign. Finite overflow follows the
rounding direction. The measured raw SM86 FTZ flag flushes FP32 subnormal
inputs and half-subnormal results; results rounded into half-normal remain normal.

CPU and SPIR-V use integer significands, bounded shifts and exact rounding,
without host FP modes, shaderInt64 or Vulkan shaderFloat16 support.
Half arithmetic, packed conversions, FP64, constant/uniform sources
and unchecked fields remain unsupported.

**{half_conversion['hardware_results']:,} native SM86 results** match across
{half_conversion['hardware_variants']:,} encodings and {half_conversion['hardware_cases_per_variant']:,} inputs each.
Separately, **{half_conversion['lane_states_per_backend']:,} complete states / {half_conversion['dispatches_per_backend']} dispatches per backend**
pass across {half_conversion['programs']} programs with budgets {', '.join(map(str, half_conversion['budgets']))},
aliases, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public half-conversion demo checks **{half_conversion['independent_lanes_per_path']} exact rational states per path
per backend**, {half_conversion['rejections']} output-preserving rejections and a
{half_conversion['persistent_chain_lanes']}-lane conversion/arithmetic loop through
{half_conversion['persistent_chain_dispatches']} dispatches with one compiled kernel.
Maximum-size emission, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: {half_conversion['mismatches']}.** This is scalar FP32-to-FP16 conversion only, not full FP16 support.

### Scalar FP16 to FP32 widening validation

F2F.F32.F16 expands the low half of a register to FP32. Upper source bits are
ignored. Finite values widen exactly; rounding flags do not change the result.
Abs/neg operate on the half sign, FTZ flushes half subnormal inputs to signed
zero, and NaNs become canonical `0x7fffffff`. Immediate widening, high-half
selectors, FP64 and unchecked fields remain explicitly unsupported.

**{status['widening_validation']['hardware_results']:,} native SM86 comparisons** pass across
{status['widening_validation']['hardware_variants']} encodings, each supplied all 65,536 binary16 bit patterns
with randomized upper halves and execution guards. Unconditional variants cover
every bit pattern; conditional variants also test inactive destination preservation.
**{status['widening_validation']['lane_states_per_backend']:,} complete states / {status['widening_validation']['dispatches_per_backend']} dispatches per backend**
pass across 32 programs with aliases, branches and budgets 1/37. The public
binary16-oracle demo checks {status['widening_validation']['independent_lanes_per_path']:,} independent states per path
on each backend and {status['widening_validation']['rejections']} output-preserving rejections.
Maximum-size emission and 49,152 raw/prepared state pairs pass ASan/UBSan.
Khronos validation passes before dispatch. This extends F2F, not the family count
or an overall GPU-completion percentage.

```powershell
python tools/gpu/make_shader_spirv_widen_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe --compute-tool build/gcc/bin/windows/PoundGpuCompute.exe
```

### Public Vulkan runner checks

The optional **PoundGpuCompute** API/tool now dispatches the supported compute
subset, including bounded branches/loops and resumable lane buffers, without
depending on a private harness. Separate runner checks pass
**{runner['lane_states_per_backend']:,} complete lane states across {runner['dispatches_per_backend']} dispatches per backend**
on RTX 3080 and software Vulkan. All 13 independent control fixtures also pass
on both backends, including actual chained resumption.
Additional developer checks cover **{runner['cli_rejections']} CLI rejections** with output preservation
and **{runner['simulated_vulkan_failures']} simulated device-lost/OOM failures** with cleanup and unchanged state.
These host-failure simulations are not GPU instruction coverage measurements.
See [build instructions and API limits](../GPU_DEVELOPMENT.md#optional-headless-vulkan-compute-runner).

Reusable sessions additionally pass **{session['lane_states_per_backend']:,} lane states across
{session['dispatches_per_backend']} dispatches per backend**, alternating {session['kernels_per_session']} immutable compiled kernels
and varying batch sizes in one pooled buffer. The control demo also checks
{session['persistent_chain_fixtures']} persistent chains, including dispatch caps and terminal step overflow.
See [session ownership and automatic resumption](../GPU_DEVELOPMENT.md#reusable-sessions-and-compiled-kernels).

## Integration

| Layer | Status |
| --- | --- |
| GPU address mappings and permission checks | Software prototype |
| DMA copies, tiled/3D transfers, remapping and fills | Software prototype |
| Inline uploads, pushbuffer decoding and bounded GPFIFO queues | Software prototype |
| Cooperative multi-channel scheduling, barriers and semaphores | Software prototype |
| Integer and FP32 arithmetic/rounding/compare/conversion decoder, CPU reference and prepared programs | Implemented for the limited subset |
| SPIR-V, bounded branches/loops and resumable state | Verified independent-lane subset |
| Vulkan sessions, reusable compiled kernels and bounded chains | Verified development prototype; not guest submission |
| Guest OS GPU services/submission and shader headers | Not implemented |
| Other conversion sizes, FP16 arithmetic, FP64, textures and graphics stages | Not implemented |
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
python tools/gpu/make_shader_spirv_float_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_arithmetic_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_fused_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_conversion_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_round_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_i2fp_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_half_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val
```

Without `--compute-tool`, the demo scripts emit modules and expected state
buffers and optionally run `spirv-val`; **no GPU dispatch is performed**.
Build the opt-in runner, then add `--compute-tool /path/to/PoundGpuCompute`
to a demo command to dispatch and compare complete lane outputs.
The documented 256-word straight-line and 261-word bounded buffer ABIs still apply.

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
