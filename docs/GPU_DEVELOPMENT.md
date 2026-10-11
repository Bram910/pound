# GPU development

`PoundCore` now includes GPU address mappings, command channels, host semaphore
operations, bounded GPFIFO queues, cooperative multi-channel scheduling,
inline uploads and a software implementation
of the Ampere DMA copy engine. Commands
can read and write mapped GPU memory. Shader translation and rendering are still
unimplemented; no guest OS GPU submission service is connected yet. A separate
experimental SM86 integer shader decoder and single-lane reference executor are
available for developing the shader frontend.

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
floating point, memory/texture operations and unchecked arithmetic variants
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

This is not a warp interpreter, SPIR-V translator, Vulkan shader dispatch or
proof of game compatibility. No
shader-translation percentage is assigned from these prototype results.

## Prepared shader programs and control flow

`gpu_shader_program_prepare` snapshots supported instructions into caller-owned
`gpu_shader_node_t` storage and builds `gpu_shader_block_t` control-flow records.
After whole-program validation, it stores decoded instructions for subsequent
execution rather than decoding every loop iteration.
This is a reusable frontend representation for a future compiler, **not** x86
machine-code emission, SPIR-V generation or host-GPU execution.

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

## Next layers

1. Connect GPU mappings and channel submission to guest OS services.
2. Extend copy layouts, synchronization and engine method coverage.
3. Extend the SM86 shader subset and add SPIR-V generation.
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
