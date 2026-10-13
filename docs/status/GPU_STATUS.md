# GPU development status

![GPU prototype progress](gpu.svg)

Updated **2026-10-13**. Earlier integer-emitter validation checkpoint (22-family scope):
[`e6e19f3f`](https://github.com/Bram910/pound/commit/e6e19f3fee229b8f093a96a87bcdd960365dd774).
[Back to the fork overview](../../README.md#status) | [GPU architecture and tools](../GPU_DEVELOPMENT.md)

## Scope

This is a **limited SM86 compute prototype with FP32 arithmetic, rounding and scalar conversions**, not a working Switch 2
graphics backend. The counts below cover only the 32 families currently
decoded by Pound, not the entire SM86 instruction set or every encoding and
modifier within a family. There is no defensible overall GPU completion
percentage yet. Ballistic CPU progress is separate and unchanged here.

## Instruction paths

| Family | Straight-line SPIR-V | Bounded branch/loop SPIR-V |
| --- | --- | --- |
| `MOV` | Prototype | Prototype |
| `IADD3` | Prototype | Prototype |
| `IMAD` | Prototype | Prototype |
| `LOP3` | Prototype | Prototype |
| `POPC` | Prototype | Prototype |
| `SEL` | Prototype | Prototype |
| `NOP` | Prototype | Prototype |
| `BRA` | Not supported | Prototype |
| `EXIT` | Prototype | Prototype |
| `ISETP` | Prototype | Prototype |
| `PLOP3` | Prototype | Prototype |
| `IMNMX` | Prototype | Prototype |
| `BREV` | Prototype | Prototype |
| `IABS` | Prototype | Prototype |
| `SHF` | Prototype | Prototype |
| `FLO` | Prototype | Prototype |
| `PRMT` | Prototype | Prototype |
| `SGXT` | Prototype | Prototype |
| `BMSK` | Prototype | Prototype |
| `LEA` | Prototype | Prototype |
| `IDP4` | Prototype | Prototype |
| `IDP2` | Prototype | Prototype |
| `FSETP` | Prototype | Prototype |
| `FMNMX` | Prototype | Prototype |
| `FADD` | Prototype | Prototype |
| `FMUL` | Prototype | Prototype |
| `FFMA` | Prototype | Prototype |
| `F2I` | Prototype | Prototype |
| `I2F` | Prototype | Prototype |
| `FRND` | Prototype | Prototype |
| `I2FP` | Prototype | Prototype |
| `F2F` | Prototype | Prototype |

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
| Straight-line | 147 | N/A | 301,056 | 256 | NVIDIA GeForce RTX 3080 | 0 |
| Bounded branches / loops | 55 | 1, 3, 16, 97 | 112,640 | 261 | NVIDIA GeForce RTX 3080, Mesa software Vulkan (llvmpipe) | 0 |

The bounded counts are **per backend**, not a combined or cumulative total.
Inputs cover all 128 predicate masks, register aliases, integer boundaries,
mixed arithmetic, branch graphs, interior PCs and fault-state preservation.
The straight-line check covers arithmetic/bit/compare modes, including all
256 LOP3 truth tables and 65,536 indexed PRMT selectors.

Additionally, **13 independent control fixtures / 16 lane states per backend**
pass on RTX 3080 and software Vulkan, including chained resumption, invalid
state, overflow, falloff and the maximum instruction budget of 4,096.
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

**4,718,592 native SM86 results** on NVIDIA GeForce RTX 3080
match the CPU reference across 1,152 encodings and
4,096 inputs each. Separately,
**62,464 complete lane states / 122 Vulkan dispatches per backend**
pass across 26 programs, including branch graphs, predicate/register aliases,
interior PCs, faults and instruction-budget exhaustion. Both compiler outputs
pass Khronos `spirv-val`. These checks use RTX 3080 and software Vulkan.

The public floating-point demo checks **648 independently specified edge states
for each of the 2 paths on each backend**, plus 12 unsupported-emission
rejections with output preservation. **Recorded mismatches: 0.**
These measurements cover FP32 comparison/min-max only. Arithmetic checks are
recorded separately below.

### FP32 addition and multiplication validation

FADD and FMUL now support all four rounding modes, FTZ, saturation,
register/immediate inputs, abs/neg modifiers, execution guards and register aliases.
Integer significands and paired multiplication preserve subnormal behavior
without requiring shaderInt64 or host floating-point controls. Nondefault
FMUL scaling and DNZ are rejected. The corrected, rounding-dependent SM86
underflow boundary is included in the edge checks.

**12,582,912 native SM86 results** match across
1,536 encodings and 8,192 inputs each.
Separately, **77,824 complete states / 152 dispatches per backend**
pass across 32 programs with budgets 1/3/16/97, aliases,
all predicate masks, branch graphs, interior PCs and fault preservation.
Both backends run Khronos-validated modules. Separate host API checks pass
maximum-size emission and ASan/UBSan. **Recorded mismatches: 0.**

The public arithmetic demo checks **1,800 exact rational edge states
per path per backend**, 12 output-preserving rejection cases and a
128-lane loop resumed through 6 dispatches using one compiled kernel.
Other conversion sizes, FP16 arithmetic, FP64 and texture/graphics operations remain missing.

### Fused FP32 multiply-add validation

FFMA preserves the product through the addition and rounds once to FP32.
Register, immediate-multiplicand and immediate-addend forms support all four
rounding modes, FTZ, saturation, abs/neg modifiers and execution guards.
The CPU reference uses a wide integer significand; SPIR-V uses paired
32-bit arithmetic without shaderInt64. DNZ, constant/uniform sources and
unknown semantic fields are rejected explicitly.

**29,360,128 native SM86 results** match across
3,584 encodings and 8,192 inputs each.
Separately, **45,056 complete states / 88 dispatches per backend**
pass across 32 stratified programs with budgets 1, 37,
aliases, all predicate masks, FSETP-driven branches, interior PCs and fault
preservation. Modules pass Khronos validation on RTX 3080 and software Vulkan;
maximum-size emission and API preservation pass ASan/UBSan.

The public fused demo checks **538 exact rational states per path
per backend** and 8 output-preserving rejections, including cancellation
that differs from separate multiply/add, intermediate product overflow,
signed zeros, NaNs and rounding-boundary sweeps. These checks also fix the
FMUL/FFMA FTZ cutoff near the smallest normal result. **Recorded mismatches: 0.**
The demo also checks a 128-lane fused-accumulation loop through
7 dispatches with one compiled kernel and an aliased accumulator.

### FP32 / 32-bit integer conversion validation

F2I and I2F support signed/unsigned 32-bit integer types, all four rounding
modes, register/immediate sources, execution guards and register aliases.
F2I also supports FTZ and register abs/neg modifiers. Finite and infinite
float-to-integer results clamp to the destination range. NTZ converts NaNs
to zero; without NTZ, raw SM86 returns `0x80000000` for either integer type.
The CPU and SPIR-V use integer bit operations, not undefined host C casts
or Vulkan's implementation-dependent out-of-range float-to-integer conversion.
Other widths, constant/uniform sources and I2F source modifiers are rejected.

**10,289,152 native SM86 results** match across
1,256 encodings and 8,192 inputs each.
Separately, **45,056 complete states / 88 dispatches per backend**
pass across 32 programs with budgets 1, 37,
aliases, RZ/PT, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public conversion demo checks **512 exact rational states per path
per backend**, 24 output-preserving rejections, and a
128-lane conversion/arithmetic loop through
7 dispatches with one compiled kernel.
**Recorded mismatches: 0.** These counts cover FP32/32-bit conversions only.

### FP32 integral-rounding validation

FRND rounds an FP32 input to an integral value stored in FP32, without the
32-bit integer range clamp of F2I. Nearest-even, floor, ceil and truncation
support register/immediate sources, FTZ, abs/neg, guards, RZ/PT and aliases.
Signed zeros and infinities are preserved; NaNs produce the native SM86
canonical `0x7fffffff`. Large finite FP32 integers remain unchanged.
The CPU reference and SPIR-V use integer bit operations and bounded shifts,
without host FP controls, shaderInt64 or float-rounding capabilities.
Other widths, constant/uniform sources and unchecked fields are rejected.

**2,752,512 native SM86 results** match across
336 encodings and 8,192 inputs each.
Separately, **45,056 complete states / 88 dispatches per backend**
pass across 32 programs with budgets 1, 37,
aliases, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public rounding demo checks **512 exact rational states per path
per backend**, 24 output-preserving rejections and a
128-lane aliased rounding/arithmetic loop through
7 dispatches with one compiled kernel.
Maximum-sized modules, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: 0.** These counts cover FP32 FRND only.

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

**1,474,560 native SM86 results** match across
180 encodings and 8,192 inputs each,
including register aliases and RZ. Separately,
**45,056 complete states / 88 dispatches per backend**
pass across 32 programs with budgets 1, 37,
all predicate masks, aliases, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public I2FP demo checks **512 exact rational states per path
per backend**, including paired I2F/I2FP encodings, 28
output-preserving rejections, and a 128-lane
conversion/arithmetic loop through 7 dispatches using one kernel.
Maximum-size emission, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: 0.** This covers only the FP32/32-bit RN/RZ I2FP subset.

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

**2,752,512 native SM86 results** match across
336 encodings and 8,192 inputs each.
Separately, **45,056 complete states / 88 dispatches per backend**
pass across 32 programs with budgets 1, 37,
aliases, all predicate masks, branches, interior PCs and fault preservation.
Modules pass Khronos validation on RTX 3080 and software Vulkan.

The public half-conversion demo checks **512 exact rational states per path
per backend**, 24 output-preserving rejections and a
128-lane conversion/arithmetic loop through
6 dispatches with one compiled kernel.
Maximum-size emission, constant capacity and output preservation pass ASan/UBSan.
**Recorded mismatches: 0.** This is scalar FP32-to-FP16 conversion only, not full FP16 support.

### Scalar FP16 to FP32 widening validation

F2F.F32.F16 expands the low half of a register to FP32. Upper source bits are
ignored. Finite values widen exactly; rounding flags do not change the result.
Abs/neg operate on the half sign, FTZ flushes half subnormal inputs to signed
zero, and NaNs become canonical `0x7fffffff`. Immediate widening, high-half
selectors, FP64 and unchecked fields remain explicitly unsupported.

**12,582,912 native SM86 comparisons** pass across
192 encodings, each supplied all 65,536 binary16 bit patterns
with randomized upper halves and execution guards. Unconditional variants cover
every bit pattern; conditional variants also test inactive destination preservation.
**45,056 complete states / 88 dispatches per backend**
pass across 32 programs with aliases, branches and budgets 1/37. The public
binary16-oracle demo checks 4,096 independent states per path
on each backend and 8 output-preserving rejections.
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
**37,888 complete lane states across 58 dispatches per backend**
on RTX 3080 and software Vulkan. All 13 independent control fixtures also pass
on both backends, including actual chained resumption.
Additional developer checks cover **26 CLI rejections** with output preservation
and **50 simulated device-lost/OOM failures** with cleanup and unchanged state.
These host-failure simulations are not GPU instruction coverage measurements.
See [build instructions and API limits](../GPU_DEVELOPMENT.md#optional-headless-vulkan-compute-runner).

Reusable sessions additionally pass **63,168 lane states across
384 dispatches per backend**, alternating 3 immutable compiled kernels
and varying batch sizes in one pooled buffer. The control demo also checks
3 persistent chains, including dispatch caps and terminal step overflow.
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
