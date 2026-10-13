<h1 align="center">
  <img src="/resources/pound.png" height="150px">
  <br><br>
  <img src="https://img.shields.io/github/stars/pound-emu/pound" width="100">
  <a href="https://github.com/pound-emu/pound/actions?query=branch%3Amain">
    <img src="https://img.shields.io/badge/Latest Builds-Here-aa00aa.svg" width="150">
  </a>
  <br><br>
  Pound
</h1>

<p align="center"><em>“i think of getting pounded when i see that [name]” – Satisfied Customer</em></p>

## Overview

**IMPORTANT: IN ORDER TO SQUEEZE AS MUCH PERFORMANCE FOR SWITCH 1 AND 2 EMULATORS, DEVELOPMENT HAS FULLY SHIFTED TO
CREATING A [NEW ARM RECOMPILER](https://github.com/pound-emu/ballistic) FROM THE GROUND UP. IF YOU ARE A COMPILER
DEVELOPER PLEASE GIVE US YOUR SUPPORT**

- [ ] Translate SM86 to SPIR-V to Vulkan.
- [X] Add `mimalloc` for host allocator.
- [ ] Create a custom pool / slab allocator for Horizon OS.
- [ ] Create a JIT code cache memory allocator for Ballistic.
- [ ] Create a JIT metadata manager.
- [ ] Integrate Ballistic into Pound.

<!-- POUND-STATUS:BEGIN -->
## Status

![cpu](docs/status/badge_cpu.svg) ![ir](docs/status/badge_ir.svg) ![gpu](docs/status/badge_gpu.svg)

![Status](docs/status/status.svg)

* ARM64: share of the 3856 A64 encodings in Ballistic's decoder table that its x86 tier-1 compiler can run (54 run, 201 decoded only, 3601 not started). Ballistic commit `9db2adab`.

<!-- POUND-GPU-STATUS:BEGIN -->
### GPU progress

[![GPU prototype progress](docs/status/gpu.svg)](docs/status/GPU_STATUS.md)

**[GPU status details](docs/status/GPU_STATUS.md)** - updated 2026-10-13.
The decoded prototype covers 32 SM86 families: 31 in the straight-line SPIR-V path
and all 32 in the bounded branch/loop path. At the earlier integer
emitter checkpoint `e6e19f3f` (22-family scope),
301,056 complete straight-line lane states pass on RTX 3080; 112,640 bounded states pass
**on each of RTX 3080 and software Vulkan**. These synthetic checks are not an
overall GPU completion percentage or evidence that games render.

**New FP32 comparison/min-max support:** FSETP and FMNMX pass
4,718,592 native SM86 result comparisons and
62,464 full Vulkan lane states per backend.

**FP32 addition/multiplication:** FADD and FMUL pass
12,582,912 native result checks and
77,824 complete Vulkan states per backend,
including all four rounding modes, FTZ and saturation.

**Fused FP32 multiply-add:** FFMA passes 29,360,128 native result checks
and 45,056 complete Vulkan states per backend. The new checks also
correct a rounding-dependent underflow boundary shared by FMUL and FFMA.
**FP32 / 32-bit integer conversions:** F2I and I2F pass
10,289,152 native result checks and
45,056 complete Vulkan states per backend.
Signed/unsigned results, four rounding modes and F2I FTZ/NaN handling are covered.

**FP32 integral rounding:** FRND passes 2,752,512 native results
and 45,056 complete Vulkan states per backend, including
nearest-even, floor, ceil, truncation, FTZ, abs/neg and signed-zero behavior.

**I2FP conversion encoding:** signed/unsigned 32-bit to FP32 with RN/RZ passes
1,474,560 native results and 45,056
complete Vulkan states per backend. Other I2FP sizes/modes reject explicitly.

**Scalar FP32 to FP16 conversion:** F2F.F16.F32 passes 2,752,512
native results and 45,056 complete Vulkan states per backend.
Four rounding modes, FTZ, subnormals, overflow and NaNs are covered.
**Scalar FP16 to FP32 widening:** F2F.F32.F16 passes 12,582,912
native comparisons covering every binary16 bit pattern and 45,056
complete Vulkan states per backend. Register sources, FTZ and abs/neg are verified.
FP16 arithmetic, packed conversions and graphics rendering remain unimplemented.

The optional **PoundGpuCompute** tool now runs the supported subset through
Vulkan, including resumable bounded state. Its separate runner checks pass
37,888 full states across 58 dispatches **per backend**.
Reusable sessions and compiled kernels avoid rebuilding the device/pipeline
for each dispatch; another 63,168 states pass per backend while alternating
3 kernels in one session. These remain compute-only development tools.

[GPU development details](docs/GPU_DEVELOPMENT.md) cover the software DMA,
tiled transfers, inline uploads, mappings, queues, cooperative scheduling and
semaphore prototypes. Guest OS submission and production rendering remain
unimplemented. CPU status above is a separate Ballistic snapshot and is not
refreshed by this GPU update.

Regenerate GPU visuals with `python tools/status/generate_gpu_status.py`.
<!-- POUND-GPU-STATUS:END -->

Regenerate the CPU snapshot with `python tools/status/generate_status.py --ballistic <path-to-ballistic-checkout>`.
<!-- POUND-STATUS:END -->

## Development

Pound actively supports Linux and Windows with the Clang compiler only. I cannot run Pound on Windows, so the Windows
builds will fall behind Linux builds. I would appreciate any Windows specific contributions greatly.
