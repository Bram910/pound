# GPU development status

![GPU prototype progress](gpu.svg)

Updated **2026-10-12** for [GPU checkpoint `e6e19f3f`](https://github.com/Bram910/pound/commit/e6e19f3fee229b8f093a96a87bcdd960365dd774).
[Back to the fork overview](../../README.md#status) | [GPU architecture and tools](../GPU_DEVELOPMENT.md)

## Scope

This is a **limited SM86 integer compute prototype**, not a working Switch 2
graphics backend. The counts below cover only the 22 families currently
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
