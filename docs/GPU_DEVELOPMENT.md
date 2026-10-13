# GPU development

`PoundCore` now includes GPU address mappings, command channels, host semaphore
operations, bounded GPFIFO queues, cooperative multi-channel scheduling,
inline uploads and a software implementation
of the Ampere DMA copy engine. Commands
can read and write mapped GPU memory. Production rendering is still
unimplemented; no guest OS GPU submission service is connected yet. A separate
experimental SM86 shader decoder and single-lane reference executor are
available for integer operations, FP32 comparisons/min-max, addition/multiplication/fused multiply-add
and FP32/32-bit integer conversions, integral rounding and scalar FP32-to-FP16 conversion. The SPIR-V
emitter provides straight-line and budget-limited control-flow modes.

## Supported command formats

| Packet | Behavior |
| --- | --- |
| Incrementing | Advance the method by four bytes per payload |
| Non-incrementing | Send every payload to the same method |
| Increment-once | Advance after the first payload only |
| Immediate | Decode the 13-bit payload directly from the header |
| NOP / zero-count packet | Consume without dispatch |
| End of segment | Ignore the rest of the supplied segment |

`gpu_pushbuffer_decode` accepts little-endian bytes, including unaligned buffers.
Keep one initialized decoder per channel. A packet may continue into the next
segment; call `gpu_pushbuffer_finish` only at the end of the entire stream.
Each decode call supplies one segment, or a chunk of it. `segment_ended` signals
an end-of-segment word; discard the rest of that segment before supplying another
chunk. The next decode call clears this flag and starts the next segment.

`bytes_consumed` identifies the first unconsumed word on failure. A failed
callback leaves its word and decoder state ready to retry. The handler must not
perform irreversible work before returning an error, and must not mutate the
decoder or its input. Earlier successfully dispatched methods are not rolled back.
Headers with reserved bit 12 or method sequences beyond the supported 12-bit
method address range are rejected. Subdevice-mask controls and reserved opcodes
return an unsupported-command error instead of being silently skipped.

`gpu_gpfifo_decode` extracts a 40-bit address, word count and conditional-fetch,
subroutine-level and synchronization flags from an eight-byte entry. Only the
all-zero control NOP is supported. The caller must implement scheduling semantics
for these flags before submitting referenced memory to the decoder.
GPU addresses are not CPU addresses; no guest-memory translation is assumed here.

## GPU address space

`gpu_memory_init`, `gpu_memory_map` and `gpu_memory_unmap` manage up to 128 sorted
host-backed ranges in a 49-bit GPU address space. GPU virtual ranges cannot
overlap, but separate ranges can refer to the same host storage. Unmapping
requires an exact range match. This is a software mapping table, not a hardware
page-table walker.

Reads and writes span adjacent mappings and enforce read/write permissions.
The entire access is checked before copying any bytes, so a hole, permission
failure or overflowing address cannot produce a partial write.
Mapping storage and host backing remain caller-owned. Host read/write buffers
must not overlap mapped storage or the mapper itself. Mapping changes and GPU
execution must be serialized by the caller.
Lookup uses a short linear scan for up to 16 mappings and binary search for
larger tables; access validation and ownership rules are unchanged.

## Executable command channels

Initialize `gpu_channel_t` with a memory mapper and separate caller-owned command
and transfer buffers. These buffers must not overlap each other, mapped storage
or channel/mapper state. They must remain alive for the channel's lifetime.
The command capacity bounds a GPFIFO segment; transfer capacity bounds a copy's
total row payload. Oversized work fails with `POUND_ERROR_GPU_RESOURCE_LIMIT`.

`gpu_channel_submit` accepts one eight-byte GPFIFO entry and snapshots its mapped
command bytes before execution. Packet state persists between entries. The
synchronous executor supports `SYNC_WAIT`; conditional fetching and subroutine
entries are rejected. Snapshotting is a deterministic initial model, not an
emulation of hardware prefetch timing or producer/consumer races.

Supported host methods include SetObject, NOP, SetReference, WFI and 32/64-bit
semaphore operations. Classes `0xC6B5` (DMA copy) and `0xA140` (inline-to-memory)
can be bound independently on the eight subchannels.
Every engine operation executes synchronously, so WFI and supported release
fences have no asynchronous work to drain after all inline data has arrived.
SetReference, WFI and WFI-enabled releases return `POUND_ERROR_GPU_BUSY` while
an inline upload is incomplete; they must follow its payload. Thread-group
switching remains unsupported.

### Memory barriers and completion clocks

Host `MEM_OP_A/B/C/D` methods accept GPU and system MEMBAR operations. All three
configuration methods must precede each execution, with A/B zero, C zero or one,
and D equal to `5 << 27`. Configuration persists across segments; a successful
barrier consumes its readiness state. Other memory operations and nonzero
reserved fields fail explicitly. The serialized mapper already makes completed
PBDMA accesses visible. These barriers do not finish a partially supplied inline
upload, flush a real host GPU cache, or model a hardware MMU/TLB. Use WFI before
an ordered completion when engine-idle semantics are required.

Host `SEM_EXECUTE` can publish 16-byte timestamped release/reduction records.
Register a nanosecond clock using `gpu_channel_set_clock(channel, callback, context)`.
No clock is assumed by default: timestamp requests without one return unsupported.
The address must be 16-byte aligned and the whole record writable. A 32-bit
payload has zero padding in bytes 4–7; bytes 8–15 contain the 64-bit timestamp.
The timestamp is written before the payload. Validation and any reduction read
precede the clock callback, so a rejected or faulting operation neither samples
the clock nor publishes a partial record. Acquires still read only their payload
width. Timestamp flags on acquires are rejected.

The callback/context remain caller-owned and must not mutate mappings, channel
state or backing memory, reenter GPU execution, or fail. Execution and mapping
changes must remain serialized. This is not an atomic 16-byte host write and
does not support concurrent CPU polling. Clock changes are rejected with a
pending segment, unfinished packet or active upload; channel reset preserves
the registered clock while clearing statistics. DMA and inline-engine four-word
completion formats remain unsupported.

Semaphore acquires support equality, unsigned greater-or-equal, circular
greater-or-equal, nonzero bitwise AND and nonzero bitwise NOR. Circular comparisons
use the half-range rule at the selected 32/64-bit width. Reductions support
signed/unsigned min/max, XOR, AND, OR, addition and bounded increment/decrement.
Unsupported signedness/width combinations are rejected, including signed 64-bit
addition and all 64-bit increment/decrement operations. Reduction updates use
the serialized software mapper; they are not concurrent host atomic operations,
and hardware PTE atomic-disable flags are not modeled.

An unsatisfied acquire returns `POUND_ERROR_GPU_WAIT` without blocking a host
thread. The pending snapshot and failing payload are retained. Update the
semaphore through its memory mapping and call `gpu_channel_resume`. Completed
methods are not executed twice. New submissions return `POUND_ERROR_GPU_BUSY`
while a segment is pending. Other execution errors also retain the snapshot;
fix a recoverable memory fault and resume, or reset the channel to discard it.
`gpu_channel_finish` detects an unfinished packet or inline upload. Reset clears bindings,
registers, statistics and pending work, while preserving the supplied buffers
and mapper.

### Bounded GPFIFO queues

`gpu_queue_init` attaches a caller-owned, power-of-two ring of at least two
eight-byte entries to one channel. The aligned ring address must fit in the
40-bit GPFIFO address space. The caller owns producer PUT updates and must leave
one slot empty, avoid overrunning GET, and serialize ring changes with execution.
This is a software consumer, not a hardware USERD/runlist implementation.

`gpu_queue_drain(queue, put, budget, &completed)` retires at most `budget` entries,
wrapping GET at the ring boundary. Budget zero does no work. A successful return
does not necessarily mean the ring is empty; inspect GET against the supplied PUT.
On a wait or fault, GET stays on the failing entry. If its segment was already
snapshotted, draining again resumes that snapshot rather than rereading the slot
or repeating earlier methods. A command-fetch fault can be repaired before retry.
Packets and inline payloads may span entries. Completed counts refer to whole
retired entries, not methods or completed uploads.

While attached, the queue must exclusively own channel submission/resume;
do not independently mutate its state. To discard an unrecoverable segment,
reset the channel and reinitialize the queue at a caller-chosen GET. Earlier
writes are not rolled back. Other channels may update shared
semaphores between drain calls. No polling thread, automatic timeout, TSG switch
or hardware scheduler is implied.

### Cooperative multi-channel scheduler

`gpu_scheduler_init` attaches 1–32 distinct queues, each on a distinct channel,
without allocating storage or creating a thread. Command buffers must be private
and must not overlap any other channel's command or transfer buffer; attachment
rejects these overlaps to protect pending snapshots. Transfer buffers may be
shared between channels because transfer operations execute synchronously.
Attach before any failed
pending submission; initial producer PUT values equal each queue's GET.
Publish new producer positions with `gpu_scheduler_publish`. The same ring
ownership, one-empty-slot and serialized execution rules apply. A scheduler
must exclusively own its attached queues/channels; do not drain or reset them
independently or change its internal slots. Queues, channel buffers and state
must not alias scheduler/result storage. To discard work, reset the affected
channel/queue and reinitialize the scheduler with the intended producer positions.

`gpu_scheduler_run(scheduler, attempt_budget, &result)` visits queues round-robin,
trying at most one entry per visit. Failed acquire polls and fault attempts count
against the budget, not just retired entries. The next queue cursor persists
between calls; budget zero does no execution. This bounds attempts, not wall
time or methods inside one already-bounded command segment. A queue that waits
does not prevent other channels from releasing its dependency. A full traversal
with no retired entries or successfully executed methods stops promptly instead
of busy-spinning. This reports a stall, not proof of deadlock: a caller may later
update external semaphores and retry.

Faulting queues are isolated and attempted at most once per run, while unrelated
work continues. Fix the mapping and call run again to retry the retained snapshot.
The result reports attempts, retired entries, pending queues, last-observed waits,
busy queues, faults, the lowest faulted slot, and budget exhaustion. Each slot
retains its last attempted status, which may be stale until polled again.
Faults take return-status precedence over BUSY and WAIT. Without a fault, budget
exhaustion returns success with `budget_exhausted` set, even if some queues last
waited; a fully stalled traversal returns WAIT or BUSY. Inspect `pending` rather
than treating success as completion. An empty queue can still have an unfinished
packet/upload: call `gpu_channel_finish` separately at the end of the stream.
This is an offline cooperative executor, not NVIDIA TSG/runlist scheduling,
USERD emulation, interrupt delivery, timeslicing or guest-service integration.

## Inline-to-memory uploads

The `0xA140` engine accepts byte line lengths/counts, a destination address/pitch,
block-linear geometry, origins, launch flags, inline data and completion semaphore
registers. Pitched uploads and selected 2D/3D block-linear rectangles use the same
explicit Tegra/Turing layout choice as DMA copies. Inline block-size methods have
no GOB-height selector; their checked layout uses eight-row GOBs.

This initial subset supports line lengths divisible by four and little-endian
four-byte payload words. Other lengths fail explicitly rather than guessing row
padding behavior. Payload words can cross packet/GPFIFO boundaries. Upload size
is bounded by the channel transfer capacity, although no full-image staging
buffer is needed. Flush-only launches and one-word completion releases are
supported; interrupts, notifications, reductions in inline launch flags,
conditional rendering and four-word completion records are not.

Launch prevalidates the entire destination rectangle and completion address.
Payload words become visible as they arrive; an upload is not an all-or-nothing
transaction. Each word's mapped pieces are validated before any piece is written,
including at a tiled sector boundary. A late mapping fault leaves that word
unconsumed and retryable, while earlier words remain visible. The final word
revalidates the completion address before writing, then publishes the semaphore.
Zero-sized uploads can release completion immediately. Extra payload words fail.

Configuration methods, rebinding and layout changes are rejected during an active
upload. Direct `gpu_inline_method` callers must also keep its state unchanged
between calls. Reset discards an incomplete upload without undoing its writes.

## DMA copy support

The `0xC6B5` engine implements source/destination addresses, pitches, line length,
line count, semaphore registers and LaunchDMA. Both copy transfer modes execute
synchronously. Pitch-to-pitch supports single-line and multiline copies with an
optional four-byte completion semaphore release.

All destination rows and the completion address are checked before writing.
Source rows are staged before any destination update, making overlapping and
aliased copies deterministic. A failed copy does not publish its semaphore.
Physical addresses, conditional rendering, interrupts and four-word semaphore
releases return unsupported-command errors.
These restrictions prevent unimplemented operations from appearing to succeed.

### Block-linear texture transfers

Multiline copies can convert pitch-to-block-linear, block-linear-to-pitch and
block-linear-to-block-linear surfaces, including different block heights/depths
and nonzero source/destination origins. The supported subset is uncompressed
2D and selected XY slices of 3D textures, one GOB wide, with Fermi-height GOBs.
Block heights and depths can each be 1, 2, 4, 8, 16 or 32 GOBs.
The layer register selects a Z slice below the surface depth; each launch copies
one XY rectangle, so a complete volume needs multiple launches. These launches
are individually validated, not one rollback-capable transaction. Widths,
line lengths and horizontal origins are
in bytes when remapping is disabled. Pitches apply only to pitched sides.
Both packed origin registers and separate full-width X/Y registers are accepted.
Rectangle bounds and padded surface size are checked before memory access.
Only the copied rectangle must be mapped; padding is not read or written.

Sector ordering is **not inferred from the engine class**. Call
`gpu_channel_set_sector_layout` with `GPU_SECTOR_LAYOUT_16BX2` or
`GPU_SECTOR_LAYOUT_TURING2D` to opt into a known sector layout for the supplied
memory image. The default
is unspecified and rejects nonempty block-linear transfers. Layout changes
are rejected while a segment or packet is pending; channel reset preserves
this host setting. Direct copy-engine callers set `copy.sector_layout` instead.

This is the historical Tegra K1/Parker layout described by Linux's
`DRM_FORMAT_MOD_NVIDIA_16BX2_BLOCK`, not a claim that raw Switch 2 or desktop
Ampere physical memory uses that sector ordering. The separate Turing 2D GOB
ordering follows Mesa's NIL layout description. It also requires explicit host
selection: neither option detects the memory format used by a game.
Page kinds, compression and independently strided 2D array layers are not
implemented. The 3D layer register is not an arbitrary array-layer stride.
Guest service integration will need explicit mapping metadata before choosing
the appropriate layout. Unsupported geometry fails rather than using a
speculative address formula.

`gpu_surface_prepare` creates a checked layout for a rectangle.
It defaults the prepared layout to 16Bx2; direct surface callers may select
Turing ordering through the prepared layout's `sector_layout` field.
`gpu_surface_offset` requires that successfully prepared layout and coordinates
inside the same rectangle; it is not an unchecked guest-input entry point.
`gpu_surface_prepare_elements` checks and converts element-count widths and
horizontal origins into bytes for remapped surfaces, using a 1–16 byte element
size. Byte widths and origins must fit in 32 bits; overflow is rejected.

### Component remapping and fills

Pitched and supported block-linear transfers, including 3D slices, support one to four
source/destination components, one- to four-byte component sizes, component
reordering, constant A/B insertion
and per-component `NO_WRITE`. With remapping enabled, line length counts
elements rather than bytes; tiled surface widths and horizontal origins also
count elements. Each side converts to bytes using its own component count and
size, so RGB-to-RGBA transfers use distinct source and destination element sizes.
Pitches remain byte strides. Constants are emitted
little-endian, using their lowest component-size bytes. Skipped destination
components remain unchanged and do not require read or write access.

Constant-only remaps implement buffer fills without reading the source address.
Remaps using source components stage whole source rows before any destination
write. Active destination components and the completion semaphore are validated
before writing, including for fills. Remap row widths are limited to 32-bit byte
counts, and source/output payload sizes are bounded by transfer capacity even
when the source is not used. Tiled remapping requires the same explicit
sector-layout selection and multiline mode as ordinary tiled transfers.
Components crossing a 16-byte sector boundary are split into separate mapped
accesses; skipped components remain untouched even when they occupy separate
sectors. Constant-only tiled clears ignore the unused source address and geometry.

## Inspect a command stream

Configure with `-DPOUND_BUILD_GPU_TOOLS=ON`, then build `PoundGpuTrace`.
The executable is placed alongside Pound in the configured platform directory.
Run `PoundGpuTrace pushbuffer.bin`, or pass multiple files in segment order.
Files contain little-endian words, not GPFIFO entries.

For example, words `0x20020040`, `0x12345678`, `0x9abcdef0` dispatch methods
`0x0100` and `0x0104` on subchannel zero. The trace tool fails on a partial word,
unsupported packet, or unfinished stream.

## Replay mapped memory

Build `PoundGpuReplay` with the same `POUND_BUILD_GPU_TOOLS` option, then run:

```text
PoundGpuReplay memory.bin gpfifo.bin output-memory.bin [--sector-layout=16bx2|turing2d] [--clock-step-ns=1..1000000000]
```

The input memory image is mapped read/write at GPU address zero. The GPFIFO file
contains little-endian eight-byte entries pointing into that image. The final
memory image is saved only when the entire stream completes successfully.
Images are limited to 64 MiB, command snapshots to 8 MiB and staged copy payloads
to 16 MiB. A wait or unsupported command fails promptly instead of hanging.
This tool operates on offline memory images; it is not a game loader.
The optional clock starts at zero and advances by the supplied step for each
successful timestamped record. It is an explicit synthetic clock, not measured
GPU time or a performance metric; arithmetic wraps at 64 bits. Without this
option timestamped releases remain unsupported. Both options may be supplied
in either order; duplicate/unknown options and invalid steps fail.

A portable synthetic copy demo generates commands, replays them, checks every
modified byte and saves a PNG of the copied texture:

```text
python tools/gpu/make_copy_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
```

The default output directory is `build/gpu-demo`. On other platforms, pass the
appropriate built executable path. The preview is a verified memory copy of an
input texture, not a GPU-rendered scene. The demo checks the destination pitch,
completion semaphore and all bytes outside the intended writes.

Additional demos exercise texture layouts, remapping and volumes:

```text
python tools/gpu/make_tiled_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
python tools/gpu/make_remap_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
python tools/gpu/make_remap_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe --tiled --output-dir build/gpu-tiled-remap-demo
python tools/gpu/make_volume_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
```

The tiled demo uploads to a block-linear surface, copies into a different block
height and origin, then reads back to pitched memory. It checks the intermediate
tiled bytes against a separate scalar address calculation, all padding and the
entire output memory image, not just a round trip. Pass `--block-height-log2`
with values 0 through 5 to exercise every supported height. Its output directory
defaults to `build/gpu-tiled-demo`. The replay option
`--sector-layout=16bx2` or `--sector-layout=turing2d` explicitly selects the
synthetic image's sector format. The tiled and remap demos accept the matching
`--sector-layout` argument and default to `16bx2`.

The remap demo clears a pitched image with a constant, reorders RGB channels
while preserving its alpha component, then expands RGB to RGBA with constant
alpha. With `--tiled`, it clears and updates a tiled image while preserving alpha,
expands into a second tiled surface, then reads back through an identity remap.
Intermediate surfaces are checked against the independent address calculation,
including both element-scaled origins. Pass `--block-height-log2` (0 through 5)
to select the first tiled block height. It also checks every output byte and padding.
Its output directory defaults to `build/gpu-remap-demo`. These are software DMA
previews, not rendered frames or demonstrations of game compatibility.

The volume demo uploads seven RGB slices into a tiled BGRA volume, clears a
second volume, reverses slice order while reordering channels and preserving
alpha with `NO_WRITE`, then reads the slices back to pitched memory. Its 28 DMA
launches use different block heights/depths and cropped origins. An independent
address calculation checks every intermediate byte, padding, untouched memory
and the completion semaphore. The PNG shows the seven copied slices side by
side, not a rendered scene. This demo defaults to `turing2d`; choose either
layout with `--sector-layout` and heights/depths 0 through 5 with
`--block-height-log2` and `--block-depth-log2`. Outputs default to
`build/gpu-volume-demo`.

Inline and queue demos exercise the new execution paths:

```text
python tools/gpu/make_inline_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
build/gcc/bin/windows/PoundGpuQueueDemo.exe
build/gcc/bin/windows/PoundGpuSchedulerDemo.exe
python tools/gpu/make_completion_demo.py --replay build/gcc/bin/windows/PoundGpuReplay.exe
```

The inline demo streams four slices directly from command payloads, using pitched
and tiled destinations, five dependent DMA copies, acquire comparisons and an
addition reduction. The default stream crosses 122 GPFIFO segments; pass
`--segment-words 1` to split at every word. The complete memory image and tiled
intermediates are checked independently. Heights/depths and layouts accept the
same options as the volume demo. Outputs default to `build/gpu-inline-demo`.
The default increment-once packet combines launch and inline data; pass
`--packet-style non-incrementing` to exercise separate launch/data packets.

`PoundGpuQueueDemo` uses a wrapped ring and a one-entry budget, completes a split
inline upload, waits for another software channel to release a semaphore, then
resumes a dependent DMA copy. It also verifies that repeated waits do not repeat
methods and that editing the original commands does not change a pending snapshot.
Its checks remain enabled in release builds and fail with a nonzero exit code.

`PoundGpuSchedulerDemo` runs four channels, resolves a two-stage DMA dependency
chain from an inline upload, and checks wrapped rings, retained snapshots,
budgeted execution, isolated command-fetch faults, repair/resume, GPU/system
barriers and timestamp ordering. Its clock advances by 100 synthetic nanoseconds
per successful completion. All checks remain active in release builds.

The completion demo independently checks four release/reduction records with
32/64-bit payloads, zero padding, four barriers, segmented commands and every
untouched byte. It saves `completion-records.json` under `build/gpu-completion-demo`.
Pass `--segment-words 1` to split each command word and `--clock-step-ns` to
choose a synthetic step; these timestamps are not benchmark results.

## Experimental SM86 shader frontend

`gpu_shader_decode` reads one little-endian 16-byte code instruction into a
typed representation: operation, register/immediate operands, execution
predicate, select/branch condition, predicate inputs/outputs, comparison and
boolean operations, truth tables, branch displacement and scheduling fields.
Unaligned input is supported. This is code-only input, not an ELF/CUBIN reader,
Switch shader-header parser or automatic architecture detector. The decoder is
explicitly a checked SM86 subset; it must not be used to guess another ISA.

| Family | Checked subset |
| --- | --- |
| MOV | GPR or 32-bit immediate, full quad-lane mask |
| IADD3 | GPR/immediate middle operand, negation, dual carry outputs; .X complements and two predicate carry inputs |
| LEA | Low/high shifted address arithmetic, register/immediate addend or immediate high source, negation, carry output and .X carry input |
| IMAD | Low 32-bit result with register/immediate multiplicand/addend; signed/unsigned .WIDE and .HI with register-pair accumulator and register/immediate multiplicand |
| IDP.4A | Four packed 8-bit products accumulated modulo 32 bits, independently signed/unsigned multiplicands, register inputs |
| IDP.2A | Two packed 16-bit-by-8-bit products accumulated modulo 32 bits, independent signedness, low/high byte-pair selection, register inputs |
| LOP3.LUT | GPR/immediate middle operand, all 256 truth tables, nonzero-result predicate combined with an input using OR or PAND |
| POPC | GPR/immediate, optional register complement |
| SEL | GPR/immediate alternatives and predicate condition |
| ISETP | Signed/unsigned comparisons, .EX low-comparison chaining for multiword integers, AND/OR/XOR predicate combination, two outputs |
| PLOP3.LUT | Three optionally inverted predicates, two independent 8-bit truth tables and predicate outputs |
| IMNMX | Signed/unsigned 32-bit min/max selected by a predicate, GPR/immediate middle operand |
| FSETP | FP32 ordered/unordered comparisons, NUM/NAN and F/T, AND/OR/XOR, dual predicates, register/immediate sources, FTZ and abs/neg |
| FMNMX | FP32 numeric min/max selected by a predicate, register/immediate sources, FTZ and abs/neg |
| FADD | FP32 addition with all four rounding modes, register/immediate sources, FTZ, saturation and abs/neg |
| FMUL | FP32 multiplication with all four rounding modes, register/immediate sources, FTZ, saturation and abs/neg; default scale, no DNZ |
| FFMA | Single-rounding FP32 fused multiply-add, three register inputs or one immediate multiplicand/addend, all rounding modes, FTZ, saturation and abs/neg; no DNZ |
| F2I | FP32 to signed/unsigned 32-bit integer, four rounding modes, register/immediate sources, FTZ, abs/neg and NTZ |
| I2F | Signed/unsigned 32-bit integer to FP32, four rounding modes and register/immediate sources |
| FRND | FP32 integral rounding (nearest-even, floor, ceil, truncation), register/immediate sources, FTZ and abs/neg |
| I2FP | Signed/unsigned 32-bit integer to FP32 using the separate RN/RZ conversion encoding; register/immediate sources |
| F2F | Scalar FP32-to-FP16 conversion with four rounding modes and register/immediate sources; FP16-to-FP32 widening from low-half register sources; FTZ and abs/neg supported |
| BREV | Full 32-bit reversal, GPR/immediate |
| IABS | Modular signed absolute value, GPR/immediate |
| SHF | Signed/unsigned 32/64-bit funnel-shift width, wrap/clamp, left/right, low/high result, register or immediate shift/high operand |
| FLO | Signed/unsigned leading non-sign bit or shift amount, GPR/immediate, optional register complement, discarded predicate output |
| PRMT | Generic byte permutation/sign replication plus F4E/B4E/RC8/ECL/ECR/RC16, register/immediate selector or immediate high source |
| SGXT | Extract low bits with signed/unsigned extension, wrap/clamp count, register/immediate count |
| BMSK | Generate a positioned bit mask, wrap/clamp parameters, register/immediate width |
| NOP | Plain no-operation |
| BRA | Relative aligned branch with predicate condition |
| EXIT | Plain exit with instruction predication |

Constant/uniform registers, constant buffers, alternate quad masks,
remaining FP arithmetic/conversion widths, FP16 arithmetic, FP64, memory/texture operations and unchecked arithmetic variants
fail explicitly. Unknown semantic fields are not silently discarded. The
third-immediate IADD3 form is rejected: NVIDIA's SM86 disassembler does not
recognize it, even though generic operand helpers can represent similar forms
for other operations. Plain IMAD supports an immediate multiplicand or addend.
Scheduling metadata is decoded but not used to simulate scoreboard timing.
Decoder failures leave the output structure unchanged.

ISETP supports F, LT, EQ, LE, GT, NE, GE and T. Its first predicate result is
the comparison combined with the input predicate; its second combines the
**complement of the comparison** with that same input, not the complement of
the already-combined first output. `.EX` selects an additional low-comparison
predicate when the two 32-bit inputs are equal; F and T remain constant even
on equality. A low-word unsigned comparison followed by the matching high-word
`.EX` comparison implements a 64-bit comparison, with signedness determined by
the high word. The low-comparison and accumulator predicates can alias either
output and may be independently inverted. Plain ISETP still requires canonical
PT/!PT low-comparison fields. Uniform/constant-buffer inputs remain unsupported.
PLOP3 supports
ordinary P0–P6/PT sources, not uniform predicates. Both families capture all
inputs, including execution predication, before updating either output. PT
discards predicate writes and remains true. Two equal writable predicate
destinations are deliberately unsupported rather than assigning an unverified
write order. ISETP and PLOP3 have no GPR destination; LEA and IADD3 can update
a GPR and predicates together, including a discarded RZ destination.

LOP3 computes its 32-bit truth-table result before writing the GPR or predicate.
The predicate output combines `result != 0` with its optionally inverted input:
plain LOP3 uses OR, while `.PAND` uses AND. Discarding the GPR through RZ does
not discard the predicate result; PT discards only the predicate write.
Predicate input/output aliases use the original input value. Quad/uniform
variants and unknown semantic fields remain explicitly unsupported.

LEA shifts its low source left by an encoded distance of 0–31 before adding
the addend. LEA.HI instead extracts the high 32 bits of the shifted low/high
source pair. Low forms require the canonical RZ high-source field; .HI also
supports an immediate high source. Negation applies to the extracted shifted
value, not independently to the two source registers. LEA.X complements a
modified input instead of negating it, then adds the predicate carry input.
All predicate inputs, including execution predication, are captured before
outputs change, so a carry input may alias its output.

IADD3 has two carry outputs indicating sums of at least 2^32 and 2*2^32.
Plain negation contributes the full unsigned complement plus one, including
the carry from negating zero; merely adding already-truncated 32-bit operands
would lose that carry. IADD3.X uses input complements and adds two independently
inverted predicate inputs. LEA has one carry output. Both families reject
simultaneous modification of the first and middle operands; two identical
writable IADD3 predicate destinations are also unsupported. Immediate high
bits remain payload, not register modifiers.

IMNMX uses the minimum when its condition is true and the maximum otherwise.
IABS of `0x80000000` remains `0x80000000`; it is not saturating. BREV reverses
all 32 bits. These operations do not implement extended/64-bit variants.

IDP.4A multiplies corresponding bytes of its first two registers and adds all
four products to the accumulator. IDP.2A uses the first register's two 16-bit
elements and either the second register's low two or high two bytes. Each
multiplicand has its own signedness; signed bytes/halfwords are extended before
multiplication. Accumulation wraps modulo 2^32, with no saturation. These are
single-lane packed operations, not cooperative tensor/warp instructions.
Immediate IDP forms are rejected: SM86 disassembly does not recognize them.

IMAD.WIDE multiplies two signed or unsigned 32-bit operands, adds a **64-bit
register-pair accumulator**, and writes both halves of the 64-bit result.
IMAD.HI computes the same sum but writes only its upper 32 bits. It is not
high-product-plus-32-bit-addend: carries from the accumulator's low half matter.
RZ means an entirely zero accumulator or discarded destination. All operands,
including an accumulator or multiplicand aliased with the destination pair,
are captured before either result register changes. Signed multiplication
and 64-bit wrapping avoid signed-overflow and signed-shift assumptions in C.

This checked .WIDE/.HI subset requires accumulator pairs to start at an even
R0–R252, or RZ; .WIDE destinations have the same restriction. .HI destinations
are ordinary single registers. Odd/last partial pairs, source modifiers,
immediate accumulators, carry predicates and .X variants are not implemented
and fail explicitly. A register or full 32-bit immediate multiplicand is
supported. Plain low-result IMAD retains its existing immediate-addend form.

SHF concatenates two 32-bit inputs, shifts that pair and extracts a 32-bit
low or high result. The type controls whether distances wrap modulo 32/64 or
clamp to 32/64, and whether right shifts sign-fill from the high input's sign
bit. A 64-bit shift width does not introduce 64-bit GPR destinations. Zero,
exact-width and larger distances are handled without undefined C shifts.

FLO searches for the most significant one bit, or the most significant zero
bit in negative signed inputs. `.SH` returns its distance from the most
significant bit instead of its index. Both modes return `0xffffffff` when
there is no non-sign bit; FLO.SH is **not** an ordinary count-leading-zeros
operation for zero input. Register complement is applied before the signed
interpretation. Immediate high bits are payload, not complement modifiers.

Generic PRMT selects each output byte with one selector nibble: its low three
bits choose among eight source bytes and its high bit optionally replicates
that byte's sign bit. Only the low 16 selector bits matter in generic mode.
Specialized modes instead use only the low two selector bits: F4E extracts
four forward bytes, B4E extracts backward across the source pair, RC8/RC16
replicate a byte/halfword, and ECL/ECR clamp byte selection to an edge of the
low input. Specialized modes copy bytes literally, without sign replication.
Reserved mode 7, uniform operands and unknown semantic fields are rejected.

SGXT retains the requested number of low bits and sign-extends or zero-extends
them to 32 bits. Zero width produces zero; clamp counts of 32 or greater retain
the entire input. Wrapped counts use only five bits, so wrapped width 32 is
zero, not a full-width extraction. The sign comes from the retained field's
top bit, not necessarily the original input's top bit.

BMSK creates a run of ones starting at its position, truncated at the end of
the 32-bit destination. Clamp positions of 32 or greater produce zero and
widths of 32 or greater use a full-width base mask. Wrapped positions and
widths are reduced modulo 32 independently; wrap does not rotate overflowing
ones back to bit zero. Both operations avoid shifts by 32 or larger in C.

`gpu_shader_lane_init` creates an explicitly zero-initialized reference lane.
Direct callers can supply GPR and P0–P6 inputs before running. RZ reads zero and
discards writes; PT is always true. `gpu_shader_run` implements modular 32-bit
integer arithmetic, bitwise truth tables, population count, predicate production,
min/max, absolute value, bit reversal, funnel shifts, leading-bit search, byte
permutation, signed/unsigned extraction, mask generation, shifted address
arithmetic, carry chains, selection and aligned
in-program branches. Comparisons can drive
data-dependent counted loops. Source values are captured before destination
writes, including register aliases. Untaken branches do not access their target.
An executed out-of-range branch fails without advancing that instruction.

Execution is instruction-budgeted, including predicated-off instructions, so
loops do not hang the host. Budget exhaustion returns resource-limit and retains
registers, byte PC and step count for another run call. Earlier successful
instructions remain executed on a later fault. EXIT is required for successful
termination; falling off the supplied code is a truncated-stream error. Program
storage must remain stable during execution and between resumptions, and must
not overlap lane state. Lane state and its reserved predicate bits must remain
valid; no concurrent execution or mutation is supported.

Build `PoundGpuShader` with `POUND_BUILD_GPU_TOOLS` and run:

```text
PoundGpuShader trace raw-sm86.bin
PoundGpuShader run raw-sm86.bin 100000
PoundGpuShader analyze raw-sm86.bin
PoundGpuShader run-prepared raw-sm86.bin 100000
python tools/gpu/make_shader_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_control_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_bits_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_extract_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_address_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_dot_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_compare_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
python tools/gpu/make_shader_program_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe
```

Trace emits one JSON object per instruction, retaining the raw words. The run
tool emits a reference-lane result; absent registers in its output map are zero.
The `predicates` integer is a bit mask of P0–P6 (bit 7/PT is never stored).
Trace comparison values are 0=F, 1=LT, 2=EQ, 3=LE, 4=GT, 5=NE, 6=GE, 7=T;
boolean-operation values are 0=AND, 1=OR, 2=XOR. Predicate input/output arrays
describe predicate-writing instructions separately from their GPR operands.
`compare_extended` exposes ISETP.EX; its predicate inputs are the accumulator
followed by the low comparison. LOP3's `boolean_operation` describes its
nonzero-result/input combination, independently of the GPR truth-table lookup.
Trace shift types are 0=S64, 1=U64, 2=S32, 3=U32. The `shift_right`,
`shift_wrap`, `shift_high` and `return_shift` fields expose the checked modes.
`shift_wrap` also describes SGXT/BMSK wrapping; `compare_signed` selects signed
SGXT interpretation. PRMT `permute_mode` values are 0=index, 1=F4E, 2=B4E,
3=RC8, 4=ECL, 5=ECR, 6=RC16.
`address_shift`, `intermediate_negate` and `carry_in` describe LEA's immediate
distance, shifted-value modifier and extended mode; `shift_high` selects .HI.
`carry_in` also selects IADD3.X. Source `negate` fields mean complement in .X
mode rather than integer negation, and predicate arrays expose carry inputs
and outputs independently of GPR operands.
`dot_signed` contains the first/second multiplicand signedness flags; `dot_high`
selects IDP.2A's high byte pair. `multiply_mode` is 0=low, 1=wide, 2=high.
In wide/high multiply mode the third register operand names the low half of a
64-bit pair (RZ names zero), and wide destinations name two consecutive GPRs.
Files are bounded to 16 MiB and CLI budgets to one million instructions. The
default run budget is 100,000. The synthetic demo independently checks all output
registers, predicates and a skipped instruction, and saves `shader-report.json`
under `build/gpu-shader-demo`. Pass `--nvdisasm <path-to-installed-nvdisasm>` to
also check the fixture using NVIDIA's `--binary SM86` disassembly and save its
listing. The tool is optional, not a build/runtime dependency.

The control demo checks five counted loops (including zero and one limits),
both comparison outputs, dual predicate truth tables, signed min/max, bit
reversal, INT_MIN absolute value and a budget-exhaustion boundary. It writes
`control-report.json` under `build/gpu-shader-control-demo`; the optional NVIDIA
check also verifies its backward branch target. Neither demo runs on the GPU.

The bits demo checks nine boundary-distance programs over all 32 SHF modes,
FLO's no-sign-bit sentinel, complemented input, byte sign replication,
immediate operand forms and a predicated-off shift. It writes `bits-report.json`
under `build/gpu-shader-bits-demo`. Like the other demos, it runs only the
single-lane reference executor; optional nvdisasm checks are disassembly only.

The extraction demo checks 142 instructions covering every specialized PRMT
mode/selector, signed/unsigned and wrap/clamp SGXT boundaries, BMSK position
and width boundaries, an immediate high source, a destination/source alias
and a predicated-off instruction. It writes `extraction-report.json` under
`build/gpu-shader-extract-demo` and supports the same optional nvdisasm check.

The address demo checks four programs with zero, overflow and signed-bit
inputs, LEA low/high shifts and modifiers, immediate forms, LEA.X/IADD3.X carry
chains, carry input/output aliasing, negated zero, RZ and predication. It writes
`address-report.json` under `build/gpu-shader-address-demo` and supports the
same optional nvdisasm check. It does not submit a Vulkan shader.

The dot demo checks eight boundary programs, every packed-dot sign and byte
selection mode, signed/unsigned wide/high multiplication, immediate
multiplicands, 64-bit accumulator carries, operand/result-pair aliases, RZ and
predication. It writes `dot-report.json` under `build/gpu-shader-dot-demo` and
supports the same optional nvdisasm check. Like the other durable demos, it
runs the single-lane executor, not a graphics dispatch.

The comparison demo checks 12 signed/unsigned multiword boundary programs and
16 logic-predicate programs. It covers every comparison and boolean combination,
both outputs, low/accumulator/output predicate aliases, all 256 LOP3 tables,
OR/PAND, input inversion and discarded GPR writes. It writes
`compare-report.json` under `build/gpu-shader-compare-demo` and supports the
same optional nvdisasm check.

This frontend has been checked against 70,000 synthetic encodings using NVIDIA's
SM86 disassembler, including operand order, immediates, negation, execution
predicates, comparison modes, predicate outputs, both PLOP3 tables and branch
targets. Additional scalar checks cover 100,000 new-family programs, all 65,536
pairs of predicate truth tables over all eight inputs (524,288 cases), and 1,001
counted loops resumed with varying instruction budgets. Bit operations add
150,000 randomized programs and exhaustive checking of all 65,536 generic
byte selectors. These supplement the original arithmetic and malformed-input
checks, using independent scalar calculations and ASan/UBSan. Extraction and
specialized permutation checks add 100,000 randomized programs, 131,072 mask
boundaries, 8,192 signed/unsigned extraction boundaries and 1,536 specialized
permutation cases, including ignored selector high bits and resumable budgets.
Address arithmetic adds 100,000 randomized programs, including carry and
execution predicate aliases, both GPR/predicate outputs, instruction-budget
resumption and failure-state preservation, also checked with ASan/UBSan.
Packed/wide arithmetic adds 100,000 randomized programs and all 262,144 byte
pair/signedness combinations, using independent scalar calculations and
bit-by-bit multiplication, checked natively and with ASan/UBSan. These also
check 64-bit register aliases, budget resumption and malformed-state preservation.
Multiword comparisons and logic predicates add another 100,000 randomized
programs with independent calculations, including equality ties, predicate/GPR
aliases, predicated-off instructions, resumable budgets and rejection-state
preservation, checked natively and with ASan/UBSan.

Separate development probes also compared Pound's SHF/FLO/PRMT/SGXT/BMSK results
directly against 413,696 cases on an NVIDIA GeForce RTX 3080. Those probes replace only
the arithmetic instruction of a small CUDA kernel, retain its private input
and output addressing, and cover all 32 SHF modes, FLO modes/complement,
all seven PRMT modes, signed/unsigned/wrapped SGXT, BMSK clamp/wrap boundaries
and selected immediate forms. Actual hardware confirmed the
`0xffffffff` FLO.SH no-sign-bit result, which differs from treating it as CLZ.
This validates the probed arithmetic, not shader scheduling, other instruction
families or a complete graphics pipeline. CUDA is not a runtime dependency.

Separate LEA/IADD3 probes check another 1,146,880 result/carry pairs across
280 variants on that RTX 3080, including .HI, .X, boundary distances, immediate
forms, complements, negation and both IADD3 carry thresholds. Only arithmetic
and predicate-comparison instructions are replaced in a bounded private CUDA
kernel; its input/output addressing and stores remain unchanged. Results are
compared with both independent scalar calculations and Pound's executor.
These development probes are not a production GPU backend or a rendering test.

Packed-dot and multiply probes add 679,936 hardware comparisons: 65,536 packed
dot/high-multiply results plus 614,400 wide/high register-pair results across
150 multiply variants. They cover all packed signedness modes and byte-pair
selections, signed/unsigned wide/high multiplication, immediate multiplicands,
overflow boundaries, source/destination aliases and discarded RZ writes.
Only arithmetic instructions are replaced in private, bounded CUDA kernels;
input/output addressing and stores remain unchanged. Candidate encodings are
checked by nvdisasm before GPU submission. CUDA remains development-only.

ISETP.EX and logic-predicate probes add 7,340,032 hardware comparisons across
7,168 variants on the RTX 3080. They cover all comparison/boolean modes,
signedness, equality ties, both outputs, constant and inverted predicate inputs,
input/output aliases, all 256 LOP3 truth tables, OR/PAND and register/immediate
operands. Results agree with independent calculations and Pound's executor.
Candidate encodings are disassembled before submission; only arithmetic and
predicate-comparison slots are replaced, leaving private addressing and stores
unchanged. These results do not validate warp control flow or rendering.

The reference executor is not a warp interpreter, SPIR-V translator, Vulkan
shader dispatch or proof of game compatibility. The separate emitter below
does not change that scope. No
shader-translation percentage is assigned from these prototype results.

## Prepared shader programs and control flow

`gpu_shader_program_prepare` snapshots supported instructions into caller-owned
`gpu_shader_node_t` storage and builds `gpu_shader_block_t` control-flow records.
After whole-program validation, it stores decoded instructions for subsequent
execution rather than decoding every loop iteration.
This is a reusable frontend representation for the experimental emitter below,
**not** x86 machine-code emission. Preparation itself does not generate SPIR-V
or execute on the host GPU.

The caller supplies node and block arrays, each with capacity for the raw
instruction count, plus a program descriptor. This conservative capacity contract
bounds every write without allocating inside the core. Successful preparation
copies all instruction fields, including raw words, so the original input bytes
can be released or changed afterwards. Nodes, blocks and the descriptor must
remain unchanged and alive while executing; they must not overlap lane state.
Preparation input, storage arrays and the output descriptor must not overlap,
and input bytes must remain stable throughout preparation.

Every encoding and branch target is validated before any output storage is
written. Errors preserve the descriptor and both storage arrays. Unlike raw
step-by-step execution, preparation rejects unsupported unreachable instructions
and out-of-range targets even when their branch is statically skipped. It accepts
programs without EXIT, but reaching their end still faults at execution time.
Preparation requires nonempty, instruction-aligned input; raw decoding still
supports unaligned byte pointers.

Blocks begin at entry, branch targets and instructions after BRA/EXIT. Nodes
map back to their containing block; branches retain their target instruction.
Each block records up to two deduplicated successor block IDs, possible exit
and possible falloff. PT/!PT constants and contradictory execution/condition
guards prune impossible branch edges. Predicate values produced by earlier
instructions are not propagated, and unreachable blocks are retained: this is
structural control-flow analysis, not global constant folding or SIMT reconvergence.

`gpu_shader_run_prepared` uses the same semantic executor as `gpu_shader_run`,
with immutable prepared instructions instead of runtime decoding. Byte PC,
predication, register/predicate aliases, fault states and resumable instruction
budgets retain their raw-execution meaning. A single prepared program can be
reused across separately owned lanes and multiple budgeted calls. There is no
automatic cache lookup or invalidation; the caller chooses when to prepare
a fresh snapshot after guest code changes.

The CLI's `analyze` mode emits JSON block IDs, byte offsets, instruction counts,
successors and exit/falloff flags. `run-prepared` emits the same lane-state JSON
as `run`. Both prepared modes limit input to 65,536 instructions to bound host
allocation; raw trace/run retain their existing 16 MiB input limit.

The durable program demo checks 14 straight-line, diamond, guarded-exit,
self-loop and counted-loop programs, deduplicated/contradictory edges, falloff,
raw/prepared state agreement at different budgets and strict whole-program
rejection. It writes `program-report.json` under `build/gpu-shader-program-demo`.
Additional native and ASan/UBSan checks cover 10,000 randomized programs,
all 128 predicate masks at every block terminator, decoded-node equality,
resumption from interior PCs, snapshot independence, caller capacities and
failure-state preservation. All existing shader instruction checks are retained.

A development-only MinGW GCC `-O2` benchmark of seven alternating-order samples
measured the prepared CPU reference path at approximately 4.1x the raw path for
an integer/branch loop and 2.2x for a mixed bit-operation loop. Every sample
compares final raw/prepared states and executes five million guest instructions;
preparation is outside the timed interval. These are synthetic CPU interpreter
measurements, **not** GPU shader throughput, game frame rates or a JIT benchmark.

## Experimental integer SPIR-V emission

`gpu_shader_spirv_emit` lowers an immutable, successfully prepared program into
a SPIR-V 1.3 compute module for Vulkan 1.1. This is the first host-GPU shader
translation step, separate from the command engines and production rendering.
The core emits words without depending on a Vulkan loader or allocating memory.

The accepted straight-line subset is MOV, IADD3, IMAD, LOP3, POPC, SEL, ISETP,
PLOP3, IMNMX, BREV, IABS, SHF, FLO, PRMT, SGXT, BMSK, LEA, IDP4, IDP2, FSETP,
FMNMX, FADD, FMUL, FFMA, F2I, I2F, FRND, I2FP, F2F, NOP and EXIT: 31 instruction families. This covers all variants currently accepted
by the experimental decoder for those families, not every hardware encoding.
Register/immediate operands, RZ/PT, execution predication,
source/destination aliasing and dual predicate outputs retain the reference
semantics. Integer overflow wraps at 32 bits, including IABS of INT_MIN;
wide multiply-accumulate wraps at 64 bits.

IADD3 supports negation, .X complements, both predicate carry inputs and both
carry thresholds, including the carry contributed by negated zero. LEA
supports .HI/.X, intermediate negation, middle-source negation/complement and
its carry output. IMAD.WIDE/HI use paired 32-bit extended multiplies and
carry-aware accumulation, so they do not require Vulkan's shaderInt64 feature.
Both accumulator words are captured before writing a destination pair.

SHF covers all 32 signedness/width/direction/wrap/output-half modes. Every
dynamic SPIR-V shift operand is bounded below 32, with explicit selection for
guest distances 0, 32 and 64. FLO covers signed/unsigned, complement and shift
results, retaining the 0xffffffff sentinel when no qualifying bit exists.
PRMT implements all seven accepted modes, including byte sign replication.
SGXT/BMSK retain signed extension and clamp/wrap boundaries. IDP4/IDP2 unpack
and multiply 8/16-bit fields with independent input signedness and IDP2 byte
selection; no integer-dot-product extension is needed.

Programs must end with a single unpredicated EXIT and contain at most 1,024
instructions including that EXIT. Branches, early/predicated exits and missing
exits are rejected even if predication would skip them. Unsupported encodings
are still rejected by the frontend. There is no branch lowering,
warp synchronization, guest memory access, remaining FP arithmetic/conversion widths, texture access,
shader-header parsing, graphics-stage input/output or rendering integration.

Pass `words = NULL, capacity = 0` to query the required word count, then pass
caller-owned writable word storage with that capacity. Invalid arguments,
unsupported programs and insufficient capacity preserve both the output
buffer and `word_count`. The prepared descriptor and its storage must remain
valid and unchanged during both calls; output storage and `word_count` must
not overlap each other or the prepared program/storage. Successful emission
returns host-endian uint32 words; the CLI serializes them little-endian.

### Private compute buffer ABI

The module exposes compute entry point `main` with local size `(1, 1, 1)` and
one writable storage buffer at descriptor set 0, binding 0. Each lane occupies
256 uint32 words (1,024 bytes): words 0–254 are R0–R254; word 255 is the packed
P0–P6 predicate mask, whose input must be in 0–127. RZ and PT have no storage.
The module loads that state, executes the whole program and writes back the
registers and predicate mask. PC, budget, step count and halted state are not
part of this private ABI; the final EXIT is validated statically.

`GlobalInvocationId.x` selects a lane record. A host harness must dispatch
exactly `(lane_count, 1, 1)` workgroups, bind at least `lane_count * 1,024`
bytes and satisfy Vulkan's device dispatch/storage-buffer limits and host/GPU
memory visibility rules. There is no lane-count bounds check in the module.
Extra Y/Z workgroups would race on the same records. This ABI is for bounded,
independent-lane development checks, not a guest compute launch or SIMT model.

### Emission tools and checks

```sh
PoundGpuShader spirv raw-sm86.bin output.spv
spirv-val --target-env vulkan1.1 output.spv
python tools/gpu/make_shader_spirv_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe --spirv-val /path/to/spirv-val
```

The integer demo emits its original 21 families, checks an independently specified
reference result and produces `integer-compute.bin`, `integer-compute.spv`,
`lane-input.bin`, `lane-expected.bin` and `spirv-report.json` under
`build/gpu-shader-spirv-demo`. Without `--spirv-val` it explicitly reports
that Khronos validation was not checked. The emission CLI does not create a
Vulkan device or dispatch work. The demo can optionally dispatch and compare
its fixtures using `--compute-tool` and the headless runner described below.
Without that option, it explicitly reports that no Vulkan dispatch was
performed. Unsupported emission does not open/truncate the output file.

The earlier integer development checks validate generated modules with Khronos `spirv-val` before
dispatching them in a private Vulkan harness on an NVIDIA GeForce RTX 3080.
Across 147 programs, all 301,056 lane states (77,070,336 uint32 words) agree;
the same number also passes on Mesa's software Vulkan implementation using
Clang-emitted modules, separately from the hardware check.
They compare all 255 registers and the predicate mask against Pound's CPU
reference, with boundary/random inputs, all 128 starting predicate masks,
comparison modes, all 256 LOP3 truth tables, all 65,536 indexed PRMT selectors,
all SHF modes, carry inputs/outputs, paired multiply aliases and packed signedness.
Capacity/limit/error preservation
and deterministic emission are also checked natively and with ASan/UBSan.
These are synthetic integer compute checks, not measurements of game FPS,
rendering correctness or a percentage of full shader compatibility.

### Bounded branch and loop mode

`gpu_shader_spirv_emit_bounded(program, budget, words, capacity, word_count)`
adds BRA, guarded/early EXIT, loops and resumable instruction budgets to the
same scalar subset: all 32 currently decoded instruction families.
It keeps the original straight-line API and its 256-word ABI unchanged.
Use the original path for straight-line emission rather than paying for the
experimental dispatcher when it is unnecessary.

The bounded mode generates a structured SPIR-V loop and a guest-PC switch,
with per-invocation function-local registers and predicates. Each switch case
loads its operands, executes the already supported arithmetic, commits outputs
and advances or redirects the byte PC. Inputs and execution predicates are
captured before aliasing outputs. This is a GPU-resident control-flow baseline,
not optimized direct CFG lowering or a model of warp reconvergence, shared
memory or inter-lane synchronization.

Budget is a compile-time argument in 1–4,096 and counts guest instructions,
including skipped instructions and EXIT. Each dispatch starts a fresh budget;
the cumulative 64-bit step counter remains in lane state. Self-loops stop at
the budget instead of running indefinitely. This instruction bound does not
guarantee a wall-clock runtime or bounded driver compilation time: the host
must still choose sensible dispatch sizes and budgets. The 1,024-instruction
preparation/emission limit and whole-program encoding/target validation apply.
Missing EXIT and falloff are accepted for bounded execution and reported in
the lane result; the strict straight-line path still rejects them.

#### Bounded buffer ABI

The compute entry point, descriptor set/binding and local size are unchanged,
but each bounded lane occupies **261 uint32 words (1,044 bytes)**:

| Words | State |
| --- | --- |
| 0–254 | R0–R254 |
| 255 | Packed P0–P6 |
| 256 | Byte PC |
| 257, 258 | Cumulative step count, low/high 32-bit halves |
| 259 | Halted, zero or one |
| 260 | Output `error_t` status; input is ignored |

Dispatch exactly `(lane_count, 1, 1)` and bind at least `lane_count * 1,044`
bytes, observing the same device limits and memory-visibility rules as the
straight-line ABI. Do not mix the two lane strides. Byte PC must be aligned
to 16 and no greater than program size; predicates must be in 0–127 and halted
in 0–1. Invalid initial state returns `POUND_ERROR_INVALID_ARGUMENT` and
preserves every input state word except output status, including invalid
predicate bits. Already-halted valid state succeeds without executing.

Successful EXIT sets halted and leaves PC after EXIT. Unfinished budgeted
execution returns `POUND_ERROR_GPU_RESOURCE_LIMIT`; reuse its output as input
to continue with the same compiled program. Falloff returns
`POUND_ERROR_GPU_TRUNCATED_COMMAND` without advancing PC or steps further.
Step-count overflow also returns resource-limit without executing that step.
The fault/budget ordering matches the CPU reference, including falloff before
overflow checks and budget exhaustion immediately after the last instruction.
There is no automatic code-cache invalidation: the host must retain the same
prepared/compiled snapshot or deliberately replace it when guest code changes.

```sh
PoundGpuShader spirv-bounded raw-sm86.bin output.spv 256
spirv-val --target-env vulkan1.1 output.spv
python tools/gpu/make_shader_spirv_control_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe --spirv-val /path/to/spirv-val
```

The CLI default budget is 256. The control demo emits 13 independent fixtures
under `build/gpu-shader-spirv-control-demo`: diamonds, predicated exits, counted
loop resumption, self-loops (including budget 4,096), falloff, step overflow,
low-step carry, already-halted and invalid-state preservation. It writes raw
code, modules, input/expected lane buffers and `control-report.json`; it does
not dispatch unless `--compute-tool` is provided. With that option, it checks
every full lane result and feeds the actual counted-loop partial output back
into the runner for the resumption fixture.
Invalid CLI budgets and failed preparation/emission preserve existing output
files. A compute harness must validate each module before submission.

Separate development checks compare full 261-word states with the CPU reference
across varying budgets, interior PCs, all 128 predicate masks, randomized
cyclic/irreducible graphs, mixed arithmetic, 64-bit step boundaries and invalid
inputs. Native and ASan/UBSan checks also cover output preservation, deterministic
modules, bad targets and 1,024/1,025-instruction limits. These remain independent
lane probes, not Switch game execution or production graphics integration.
Across 55 programs and four budgets, all 112,640 complete lane states
(29,399,040 uint32 words) agree on each of RTX 3080 Vulkan and software Vulkan.
All 13 independent fixtures also agree on both backends, including chained
resumption and budget 4,096. The 301,056-state straight-line hardware regression
still passes separately.

## Optional headless Vulkan compute runner

`POUND_BUILD_GPU_VULKAN=ON` adds `gpu_shader_vulkan_run` to PoundCore and,
when GPU tools are enabled, builds **PoundGpuCompute**. The option defaults to
OFF: ordinary core/shader-tool builds do not gain a Vulkan SDK dependency.
Enabling it requires Vulkan 1.1 development headers and a loader, discoverable
by CMake's `find_package(Vulkan)`, plus a compatible runtime driver.

```sh
cmake -S . -B build/gpu -DPOUND_BUILD_GPU_TOOLS=ON -DPOUND_BUILD_GPU_VULKAN=ON
cmake --build build/gpu --target PoundGpuShader PoundGpuCompute
PoundGpuCompute straight raw-sm86.bin lane-input.bin lane-output.bin
PoundGpuCompute bounded raw-sm86.bin lane-input.bin lane-output.bin 256 auto
```

The runner accepts **raw SM86 from the supported subset**, not arbitrary SPIR-V.
It prepares/validates the entire program and emits its own module before
creating an instance, selecting a compute device and allocating a storage
buffer. It then creates descriptors, a compute pipeline and command buffer,
dispatches exactly `(lane_count, 1, 1)`, waits on a fence, reads the resulting
state and destroys its resources. Host/shader barriers and explicit
flush/invalidate handling support both coherent and noncoherent host-visible
memory. `spirv-val` is not invoked inside the runner; the public fixture demos
can run it separately with `--spirv-val` before checking GPU execution.

The little-endian file ABI is unchanged: **256 words per straight-line lane**
or **261 words per bounded lane**. Do not mix the strides. Input must contain
1–4,096 complete records. Straight-line predicate masks must be 0–127;
invalid masks reject the entire call. Bounded mode retains the per-lane
invalid-state/fault behavior documented above. Budget defaults to 256 in the
bounded CLI and is in 1–4,096; it counts guest instructions, not wall time.

An optional final device argument accepts `auto` or a physical-device index.
Auto selection prefers a compatible discrete GPU, otherwise the first
compatible compute device. Indices follow Vulkan enumeration and are not
stable device identities across runs. The runner prints the selected device.
It checks Vulkan version, compute-queue availability, dispatch-count and
storage-buffer limits instead of assuming the first adapter is usable.

```sh
PoundGpuCompute straight raw-sm86.bin lane-input.bin lane-output.bin 0
PoundGpuCompute bounded loop.bin partial-input.bin partial-output.bin 3 0
PoundGpuCompute bounded loop.bin partial-output.bin resumed-output.bin 109 0
```

`gpu_shader_vulkan_run(program, budget, state, state_word_count, device, report)`
is a synchronous, one-shot API. Budget **0** selects the strict straight-line
emitter; 1–4,096 selects bounded mode. State is a caller-owned, writable uint32
array in host byte order; word count determines the lane count. The prepared
program/storage must remain valid and immutable throughout the call. State,
the optional report and program storage must not overlap. A host/API/driver
failure preserves the state and report. Success updates the state and optional
device report, even when individual bounded lanes report faults or unfinished
execution. Read word 260 of each bounded lane to distinguish those results.

The CLI similarly leaves an existing output file unopened on argument,
preparation or execution failure. A write failure after successful execution
can still leave a partial output file. An exit code of zero means dispatch
and output writing succeeded, **not that every bounded lane exited normally**.

The public demos now accept an optional runner from the same build/checkout:

```sh
python tools/gpu/make_shader_spirv_demo.py --shader-tool /path/to/PoundGpuShader --compute-tool /path/to/PoundGpuCompute --spirv-val /path/to/spirv-val
python tools/gpu/make_shader_spirv_control_demo.py --shader-tool /path/to/PoundGpuShader --compute-tool /path/to/PoundGpuCompute --spirv-val /path/to/spirv-val
```

Their reports distinguish module validation from actual Vulkan dispatch;
without `--compute-tool` no dispatch is claimed. Fresh runner checks compare
**37,888 full lane states across 58 dispatches on each of RTX 3080 and software
Vulkan**, including the 4,096-lane boundary, all predicate masks, interior PCs,
overflow, fault preservation and randomized branch/loop programs. All 13
independent control fixtures also pass on both backends. Additional developer
checks cover 26 CLI rejections with output preservation and 50 simulated Vulkan
device-lost/out-of-memory failures with complete cleanup. Native and ASan/UBSan
checks exercise coherent/noncoherent cache paths and API rejection separately
from real device execution.

### Reusable sessions and compiled kernels

The one-shot `gpu_shader_vulkan_run` remains available, but now wraps the same
reusable implementation as the persistent APIs:

| API | Purpose |
| --- | --- |
| `gpu_shader_vulkan_session_create(device, lane_capacity, &session)` | Create a Vulkan instance/device, mapped state buffer, descriptors, command buffer and fence once |
| `gpu_shader_vulkan_kernel_create(session, program, budget, &kernel)` | Compile an immutable shader snapshot into a reusable pipeline |
| `gpu_shader_vulkan_dispatch(session, kernel, state, word_count, report)` | Upload caller state, dispatch the compiled kernel and read complete lane results |
| `gpu_shader_vulkan_kernel_destroy(kernel)` | Release a compiled pipeline |
| `gpu_shader_vulkan_session_destroy(session)` | Release the session after all its kernels are destroyed |

Session capacity is fixed at creation, in 1–4,096 lanes, with enough storage
for the bounded 261-word stride. The same session can alternate straight-line
and bounded kernels and dispatch smaller batches. Capacity is checked before
reading lane contents. Buffers, descriptors and fences are reused; command
buffers/fences are reset only after the preceding synchronous submission has
completed. This does **not** add asynchronous queues or automatic GPU-resident
state chaining: each dispatch still uploads caller state and reads results.
`report.dispatch_index` counts successful dispatches within that session.

The prepared source and its storage may be discarded after kernel creation
succeeds. Changing guest shader code requires compiling a new kernel; there
is no automatic code invalidation or shared/global shader-cache lookup. A
kernel belongs to exactly one session. Destroying a session with live kernels
returns `POUND_ERROR_GPU_BUSY` without changing it; destroy its kernels first.
The caller must externally serialize all use/destruction of a session and its
kernels. Handles must remain valid for each call, and state/report storage
must not overlap handles or each other.

Creation failures preserve output handle slots. Invalid arguments preserve
the existing session. Vulkan compilation/dispatch failures conservatively
mark it unusable: later compilation/dispatch returns a Vulkan failure without
retrying the driver. Destroy kernels and recreate the session to recover.
Dispatch failures preserve caller state and report; safe teardown drains any
possibly pending submission before releasing referenced pipelines/resources.
Teardown requires a completion or device-lost result; it does not free
in-flight resources merely because an idle wait failed. If a broken driver
cannot acknowledge either state, teardown can remain blocked.

**`bounded-chain`** keeps one session and compiled kernel alive while feeding
each completed dispatch's output into the next input:

```sh
PoundGpuCompute bounded-chain loop.bin lane-input.bin lane-output.bin 3 40 auto
```

The last two numeric arguments are the instruction budget **per dispatch**
and a dispatch-count cap, both in 1–4,096; defaults are 256 and 256. It stops
when no lane has resumable budget exhaustion, or the dispatch cap is reached.
Step-counter overflow is terminal and is not retried indefinitely. Mixed
halted, faulting and unfinished lanes retain their per-lane semantics.
Exhausting the cap still writes partial state with resource-limit statuses,
which can be resumed later; it is not an execution error or normal guest exit.

The public control demo checks three persistent chains in addition to its
13 individual fixtures: a counted loop finishes after 38 budget-3 dispatches,
a self-loop stops after four, and step overflow stops after one. Separate
session checks pass **63,168 full lane states across 384 dispatches per backend**
on RTX 3080 and software Vulkan, alternating three compiled kernels and batch
sizes 1/17/128/512 in one session. They discard/mutate prepared source storage,
check guard words, kernel ownership, capacity and busy-session destruction.

A local, single-lane, 32-iteration setup-overhead microbenchmark measures about
**128.46 ms per one-shot call versus 0.321 ms per reused dispatch** on RTX 3080,
and 19.57 ms versus 0.381 ms on software Vulkan. These numbers primarily measure
avoided instance/device/pipeline creation, not faster shader arithmetic,
rendering frame rates or expected game speedups.

This is a **development compute runner**, not a production renderer or guest
submission service. Explicit compiled-kernel reuse is available, but there is
no persistent disk pipeline cache, asynchronous queue or guest resource model.
Fence waits and safe teardown can block on a driver; the instruction budget
does not provide a wall-clock timeout. Keep experiments small. Textures,
graphics draws, presentation, guest OS integration and warp synchronization
remain unimplemented.

### FP32 comparison and min/max

`FSETP` and `FMNMX` now execute in the CPU reference, prepared programs and
both SPIR-V paths. Register operands support absolute value followed by
negation; immediate operands carry their complete 32-bit pattern. FTZ flushes
subnormal inputs to zero while preserving their sign. Instruction predication,
predicate inversion, RZ/PT and source/destination aliases follow the existing
lane model. Unsupported constant/uniform sources and additional min/max
modifiers fail explicitly.

For these two instructions, JSON traces include `float_comparison` (the
0–15 FSETP relation code), `float_ftz` and each source's `absolute` flag.
The source `negate` flag flips its FP sign after applying absolute value.

FSETP supports F, LT, EQ, LE, GT, NE, GE, NUM, NAN, LTU, EQU, LEU, GTU, NEU,
GEU and T. Ordered comparisons are false for NaNs; unordered comparisons are
true for NaNs. Positive and negative zero compare equal. The second predicate
combines the **complement of the comparison** with the condition using the
selected AND/OR/XOR operation. Condition and execution predicates are captured
before either result is written; overlapping non-PT output predicates are
rejected.

FMNMX selects numeric minimum for a true condition and maximum for a false
condition. A single NaN selects the other operand; two NaNs produce SM86's
canonical `0x7fffffff`. Minimum of opposite signed zeros is negative zero;
maximum is positive zero. Both implementations operate on integer bit patterns,
so these results do not depend on host floating-point rounding or denormal
modes, and SPIR-V does not require float-controls extensions.

Native SM86 checks compare **4,718,592 results across 1,152 encodings** against
the CPU reference on RTX 3080, with zero mismatches. The native test kernels
include explicit scheduling delays to respect hardware predicate dependencies.
Separate Vulkan checks pass **62,464 complete lane states across 122 dispatches
on each of RTX 3080 and software Vulkan**. These span 26 programs, all 128
starting predicate masks, modifiers, aliases, branch graphs, interior PCs,
fault preservation and budgets 1/3/16/97. Both GCC- and Clang-produced modules
pass Khronos validation; their instruction-ID ordering can differ.

The public demo supplies **648 independently specified edge states per path**,
including signed zeros, subnormals, infinities and quiet/signaling NaNs. It
also checks both CPU execution paths for zero-initialized input and 12
unsupported-emission cases with output preservation:

```sh
python tools/gpu/make_shader_spirv_float_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

Omit `--compute-tool` to generate inputs, expected states and modules without
dispatching. The report records whether validation and dispatch were actually
performed. Maximum-size emission, constant-pool capacity, deterministic output
within one compiler, guard words and failure preservation also pass ASan/UBSan
checks. FADD/FMUL/FFMA and scalar conversions are covered below; FP16 arithmetic and FP64
and graphics-stage integration are still unimplemented.

### FP32 addition and multiplication

`FADD` and `FMUL` execute in the CPU reference, prepared programs and both
SPIR-V paths. They support nearest-even, toward-negative-infinity,
toward-positive-infinity and toward-zero rounding, plus FTZ, saturation,
register abs/neg modifiers, execution predicates, RZ/PT and register aliases.
FADD uses SM86's second immediate form; FMUL uses the fourth. Constant/uniform
sources, nondefault FMUL scaling and DNZ are rejected explicitly.

The reference uses integer significands with guard/round/sticky bits.
SPIR-V uses 32-bit integer operations and paired multiplication, retaining
subnormals without a shaderInt64 or float-controls requirement. Signed zeros,
exact cancellation, rounding ties, overflow, gradual underflow, infinities,
invalid operations and canonical NaNs follow the measured SM86 behavior.
Saturation clamps results to [0, 1] and maps NaNs to positive zero. Tests
also preserve the measured, rounding-dependent FTZ underflow cutoff. Some
tiny results flush even when FP32 rounding alone would produce the smallest
normal value; nearby results can recover a normal value. The fused tests
below exposed and corrected this boundary in both implementations.

Native hardware checks pass **12,582,912 results across 1,536 encodings** on
RTX 3080, including all rounding modes, FTZ/SAT, register modifiers,
immediates and execution guards. Maximum-size emission, constant-pool
capacity, guard words, deterministic output and failure preservation pass
ASan/UBSan and Khronos validation checks.

Separately, **77,824 complete lane states across 152 dispatches on each of
RTX 3080 and software Vulkan** match the reference. The 32 programs cover
all 128 predicate masks, source/destination aliases, FSETP-driven branches,
interior PCs, fault preservation and budgets 1/3/16/97. Every tested module
passes Khronos validation before dispatch.

The new public demo computes expected values with Python's exact rational
arithmetic, independently of the C significand implementation. It checks
**1,800 edge states for each SPIR-V path**, including the smallest-normal
rounding boundary, plus 12 unsupported-emission cases:

```sh
python tools/gpu/make_shader_spirv_arithmetic_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

With a compute runner, it also checks a 128-lane arithmetic/compare loop
resumed through six budget-3 dispatches while retaining one compiled kernel.
Reports distinguish emitted fixtures, Khronos validation and actual dispatch.
Other conversion widths, FP16 arithmetic, FP64 and graphics stages remain unimplemented.

### Fused FP32 multiply-add

`FFMA` now executes in the CPU reference, prepared programs and both SPIR-V
paths. It keeps the product through the addition and rounds once to FP32.
The first source is a GPR; the other two are GPRs or one 32-bit immediate
multiplicand/addend. All four rounding modes, FTZ, saturation, abs/neg source
modifiers, execution predicates, RZ/PT and aliases are supported. DNZ,
constant/uniform sources and unchecked semantic fields fail explicitly.

The reference uses a wide integer significand and sticky alignment; SPIR-V
uses pairs of 32-bit words, extended multiplication and carry/borrow-aware
addition/subtraction. Neither requires host FP controls or shaderInt64.
Large intermediate products can be cancelled by the addend before overflow
is decided, and cancellation retains bits lost by separate multiply/add.

Native SM86 checks pass **29,360,128 results across 3,584 encodings** on
RTX 3080. Separately, **45,056 complete lane states across 88 dispatches on
each of RTX 3080 and software Vulkan** match the reference across 32
stratified programs. These include all predicate masks, aliases,
FSETP-driven branches, interior PCs, fault preservation and budgets 1/37.
Each module passes Khronos validation before dispatch. Maximum-size
emission, constant-pool capacity and output preservation pass ASan/UBSan.

The public fused demo checks **538 exact rational states per path per
backend** and eight unsupported-emission rejections. Its fixtures include
the fused result for `(1 + 2^-23) * (1 - 2^-23) - 1`, which is `-2^-46`
instead of the zero produced by separately rounded multiplication and
addition, plus overflow cancellation, signed zeros, NaNs and underflow sweeps.

```sh
python tools/gpu/make_shader_spirv_fused_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

The underflow sweeps also fix an overly aggressive FTZ cutoff in the earlier
FMUL implementation. For example, nearest-even `FMUL.FTZ` returns zero for
`0x00800000 * 0x3f7fffff`, but returns `0x00800000` for
`0x00800001 * 0x3f7ffffe`. Directed rounding has a different cutoff.
The shared CPU/SPIR-V rounding helper now reproduces these measured cases,
and addition/multiplication hardware and Vulkan regressions are rerun.

With a compute runner, the fused demo also checks a 128-lane loop with an
aliased accumulator through seven budget-2 dispatches while reusing one
compiled kernel. Reports record whether modules were only emitted,
validated or actually dispatched.

This remains an independent-lane compute subset. Other conversion widths, FP16 arithmetic, FP64,
textures, graphics stages and warp synchronization remain unimplemented.

### FP32 / 32-bit integer conversions

F2I converts FP32 to signed/unsigned 32-bit integers; I2F converts signed/unsigned
32-bit integers to FP32. Both paths support nearest-even, down, up and toward-zero
rounding, register/immediate inputs, execution guards, RZ/PT and aliases.
F2I also supports FTZ and register absolute/negate modifiers.

F2I clamps finite and infinite results to the destination integer range.
NTZ converts NaNs to zero, matching PTX `cvt` behavior; without NTZ, raw SM86
returns `0x80000000` for both signed and unsigned destinations. NaN sign/payload
do not change that result. Subnormal inputs are retained unless FTZ is enabled,
so directed rounding can produce a nonzero integer from a tiny nonzero input.

Both the CPU reference and SPIR-V use integer significands and bounded shifts.
I2F rounds the integer magnitude once to FP32. F2I avoids host C float-to-integer
casts and Vulkan conversion instructions whose out-of-range results are unsuitable
for reproducing this instruction. No host FP environment, shaderInt64 or
floating-point-control features are required. Other widths, constant/uniform
sources, and I2F source modifiers or unchecked semantic fields are rejected.
The separate I2FP encoding is covered below for its verified FP32/32-bit RN/RZ subset.

Native RTX 3080 checks pass **10,289,152 results across 1,256 encodings**.
Separate differential checks pass **45,056 complete lane states / 88 dispatches
per backend** on RTX 3080 and software Vulkan across 32 programs. These cover
aliases, all 128 predicate masks, branches, invalid/interior PCs, overflow and
budgets 1/37. Modules pass Khronos validation before dispatch. Maximum-size
emission, constant-pool capacity and output preservation pass ASan/UBSan.

The public conversion demo independently specifies **512 exact rational states
per path per backend** and checks 24 output-preserving emission rejections.
It includes signed/unsigned boundaries, half-integer ties, integer-to-float ties,
NaNs, infinities, subnormals, guards and aliases. With a compute runner it also
checks a 128-lane conversion/arithmetic loop through seven budget-3 dispatches,
reusing one session and compiled kernel.

```sh
python tools/gpu/make_shader_spirv_conversion_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

Without `--compute-tool`, modules and expected state buffers are generated, but
no Vulkan dispatch is claimed. This remains a synthetic independent-lane
compute subset, not game rendering or complete shader conversion coverage.

### FP32 integral rounding

FRND rounds an FP32 input to an integral value, keeping the result in FP32.
Unlike F2I, it does not clamp to the signed or unsigned 32-bit integer range.
Inputs already integral at large magnitudes remain unchanged. Nearest-even,
floor, ceil and truncation support register/immediate inputs, FTZ, register
abs/neg, execution predicates, RZ/PT and aliases in both SPIR-V paths.

Signed zero is preserved, including negative results rounded to zero.
Non-FTZ subnormals can round to positive or negative one in an outward
direction; FTZ flushes the input to signed zero first. Infinities retain
their sign, and quiet/signaling NaNs produce the measured SM86 canonical
`0x7fffffff`. Other widths, constant/uniform sources and unknown fields
are rejected rather than approximated.

The CPU reference clears and rounds significand bits. SPIR-V reproduces
the same semantics using 32-bit integer operations and masked shifts, not
host FP rounding instructions or implementation-dependent float controls.
No shaderInt64 or additional device capabilities are required.

Native RTX 3080 checks pass **2,752,512 results across 336 encodings**.
Separate differential checks pass **45,056 complete lane states / 88 dispatches
per backend** on RTX 3080 and software Vulkan across 32 programs, including
aliases, all predicate masks, branches, interior/invalid PCs, step overflow
and budgets 1/37. Modules pass Khronos validation before dispatch.
Maximum-size emission, constant-pool capacity and output preservation pass
ASan/UBSan; sanitized raw/prepared execution compares 86,016 full-state pairs.

The public rounding demo checks **512 exact rational states per path per
backend**, 24 output-preserving emission rejections, and a 128-lane aliased
rounding/arithmetic loop through seven budget-3 dispatches using one
compiled kernel. Its edge cases include half-integer ties, signed zeros,
subnormals, infinities, NaNs and integral values beyond the integer range.

```sh
python tools/gpu/make_shader_spirv_round_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

Without a compute runner, modules and expected state buffers are emitted,
with optional Khronos validation, but no Vulkan dispatch is claimed.

### I2FP FP32/32-bit conversion encoding

I2FP's separate encoding now accepts signed/unsigned 32-bit integer inputs
and FP32 results with nearest-even (RN) or toward-zero (RZ) rounding.
The native PTX fixture emits this encoding for `cvt.rn.f32.s32` and
`cvt.rz.f32.u32`, which previously rejected even though I2F was supported.
Register/immediate inputs, execution guards, RZ/PT and source/destination
aliases use the existing integer-only conversion implementation in the
CPU reference and both SPIR-V paths.

Other widths, packing/suboperations, constant/uniform inputs and source
modifiers are still rejected. RM/RP are not accepted or coerced into a
different rounding mode; the separately supported I2F encoding implements
all four modes. This extends encoding coverage and does not claim a measured
emulator speedup or complete I2FP packing support.

These tests also uncovered a software Vulkan bit-preservation issue affecting
ordinary MOV and I2F, not just I2FP: a predicated zero write over `0xbf800000`
returned `0x80000000` on llvmpipe/LLVM 15 with a plain word OpSelect.
Register commits now use a full-word mask/XOR merge for both normal and wide
destinations. The public fixture checks MOV, I2F and I2FP zero overwrites
across all predicate masks, including preserving the previous bits when inactive.

Native RTX 3080 checks pass **1,474,560 results across 180 encodings**,
including an aliased source/destination, RZ, signed boundaries and rounding ties.
Separate differential checks pass **45,056 complete lane states / 88 dispatches
per backend** on RTX 3080 and software Vulkan across 32 programs with
aliases, all predicate masks, branches, interior/invalid PCs, step overflow
and budgets 1/37. Modules pass Khronos validation before dispatch.
Maximum-size emission, constant capacity and output preservation pass
ASan/UBSan; sanitized raw/prepared execution compares 46,080 full-state pairs.

The public I2FP demo checks **512 exact rational states per path per backend**,
paired I2F/I2FP cases and 28 output-preserving emission rejections. With a
compute runner it also checks a 128-lane conversion/arithmetic loop through
seven budget-3 dispatches while reusing one compiled kernel.

```sh
python tools/gpu/make_shader_spirv_i2fp_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

Without `--compute-tool`, modules and expected states are generated, with
optional Khronos validation, but no Vulkan dispatch is claimed.

### Scalar FP32 to FP16 conversion

F2F's F16.F32 direction converts an FP32 input
into an FP16 bit pattern in the low 16 bits of a zero-extended 32-bit register.
All four rounding modes, FTZ, register abs/neg, register/immediate inputs,
execution guards, RZ/PT and source/destination aliases are supported.

Integer significands and exact rounding reproduce subnormals, nearest-even
ties, signed zero, directional overflow and infinities. NaNs produce the
measured native half canonical `0x7fff`, regardless of sign or payload.
The raw SM86 FTZ flag flushes FP32 subnormal inputs and half-subnormal outputs;
rounding into a half-normal result is retained. For example, converting
`0x33800000` normally yields `0x0001`, while FTZ yields zero. Nearest-even
conversion of `0x387fe000` yields `0x0400`, including with FTZ enabled.

CPU and SPIR-V use only 32-bit integer arithmetic and bounded shifts.
No host FP mode, shaderInt64 or Vulkan shaderFloat16 support is required.
Same-width F2F, packed F2FP conversions, FP64, saturation,
constant/uniform sources and unchecked fields are rejected explicitly.
This is not FP16 arithmetic or complete F2F-family support.

The F32.F16 direction now widens the low 16 bits of a register exactly to FP32.
Upper source bits are ignored, abs/neg modify the half sign, and FTZ flushes
half subnormals to signed zero. NaNs become canonical FP32 `0x7fffffff`.
All four rounding fields are accepted; finite widening is exact and does not
round. Immediate widening and high-half selectors remain unsupported.
CPU and SPIR-V normalize half subnormals with integer arithmetic; no Vulkan
shaderFloat16 support is required.

Native comparisons pass 12,582,912 cases across 192 encodings supplied every
binary16 bit pattern, random upper halves and execution guards. Multi-instruction
checks pass 45,056 complete states per backend on RTX 3080 and software Vulkan.
The public widening fixture independently compares Python binary16 expansion
against 4,096 states per fast/bounded path on each backend. Maximum-size emission,
output preservation and 49,152 raw/prepared state pairs pass ASan/UBSan.

```powershell
python tools/gpu/make_shader_spirv_widen_demo.py --shader-tool build/gcc/bin/windows/PoundGpuShader.exe --compute-tool build/gcc/bin/windows/PoundGpuCompute.exe
```

Native RTX 3080 checks pass **2,752,512 results across 336 encodings**.
Separate differential checks pass **45,056 complete lane states / 88 dispatches
per backend** on RTX 3080 and software Vulkan across 32 programs with
aliases, all predicate masks, branches, interior/invalid PCs, step overflow
and budgets 1/37. Modules pass Khronos validation before dispatch.
Maximum-sized emission, constant-pool capacity and output preservation pass
ASan/UBSan; sanitized raw/prepared execution compares 86,016 full-state pairs.

The public half-conversion demo checks **512 exact rational states per path
per backend**, 24 output-preserving emission rejections and a 128-lane
conversion/arithmetic loop through six budget-3 dispatches using one kernel.

```sh
python tools/gpu/make_shader_spirv_half_demo.py --shader-tool /path/to/PoundGpuShader --spirv-val /path/to/spirv-val --compute-tool /path/to/PoundGpuCompute
```

Without `--compute-tool`, modules and expected states are generated, with
optional Khronos validation, but no Vulkan dispatch is claimed.

## Next layers

1. Connect GPU mappings and channel submission to guest OS services.
2. Extend copy layouts, synchronization and engine method coverage.
3. Extend SM86/SPIR-V subsets, optimize direct CFG lowering and add graphics-stage interfaces.
4. Vulkan resource management, draws, compute and presentation.

Command parsing alone does not establish Switch 2 compatibility or a GPU
completion percentage. Validation currently uses synthetic streams rather than
captured game submissions.

## Format references

- [NVIDIA FIFO_DMA reference](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_ram.ref.txt)
- [NVIDIA Ampere GPFIFO class header](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/host/clc56f.h)
- [NVIDIA host semaphore and GPFIFO semantics](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_pbdma.ref.txt)
- [NVIDIA inline-to-memory class header](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/inline-to-memory/cla140.h)
- [NVIDIA Ampere DMA copy class header](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/dma-copy/clc6b5.h)
- [Linux NVIDIA block-linear format modifiers](https://github.com/torvalds/linux/blob/master/include/uapi/drm/drm_fourcc.h)
- [Mesa NVK copy and fill command generation](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/vulkan/nvk_cmd_copy.c)
- [Mesa NIL GOB ordering and 3D block addressing](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/nil/copy.rs)
- [Mesa NIL tiling geometry](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/nil/tiling.rs)
- [Mesa NAK SM70+ encoder used for SM86 field references](https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/compiler/nak/sm70_encode.rs)
- [NVIDIA CUDA binary utilities and raw SM86 disassembly](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
- [NVIDIA PTX byte permutation semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt)
- [NVIDIA PTX packed byte dot product](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a)
- [NVIDIA PTX packed halfword/byte dot product](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a)
- [NVIDIA PTX floating-point comparisons and predicate combination](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)
- [NVIDIA PTX floating-point min/max semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min)
- [NVIDIA PTX floating-point addition](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add)
- [NVIDIA PTX floating-point multiplication](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul)
- [NVIDIA PTX fused multiply-add](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma)
- [NVIDIA PTX conversion, rounding and saturation](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
- [Khronos SPIR-V specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [Khronos SPIR-V core instruction grammar](https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json)
- [Khronos SPIRV-Tools validator](https://github.com/KhronosGroup/SPIRV-Tools)
- [Khronos Vulkan host memory and noncoherent cache rules](https://docs.vulkan.org/spec/latest/chapters/memory.html)
- [Khronos Vulkan synchronization and cache control](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)
- [Khronos Vulkan command-buffer reset requirements](https://docs.vulkan.org/refpages/latest/refpages/source/vkResetCommandBuffer.html)
- [Khronos Vulkan fence reset requirements](https://docs.vulkan.org/refpages/latest/refpages/source/vkResetFences.html)
- [Khronos Vulkan device-loss and resource-lifetime rules](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-lost-device)
