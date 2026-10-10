# GPU development

The first GPU component is a command frontend, compiled into `PoundCore`.
It decodes NVIDIA pushbuffer methods into a callback with a subchannel,
byte-addressed method and 32-bit payload. It does not execute methods or render.

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

## Inspect a command stream

Configure with `-DPOUND_BUILD_GPU_TOOLS=ON`, then build `PoundGpuTrace`.
The executable is placed alongside Pound in the configured platform directory.
Run `PoundGpuTrace pushbuffer.bin`, or pass multiple files in segment order.
Files contain little-endian words, not GPFIFO entries.

For example, words `0x20020040`, `0x12345678`, `0x9abcdef0` dispatch methods
`0x0100` and `0x0104` on subchannel zero. The trace tool fails on a partial word,
unsupported packet, or unfinished stream.

## Next layers

1. GPU virtual-memory mappings and guest command submission services.
2. Engine method dispatch, synchronization and copy operations.
3. SM86 shader decoding and SPIR-V generation.
4. Vulkan resource management, draws, compute and presentation.

Command parsing alone does not establish Switch 2 compatibility or a GPU
completion percentage. Validation currently uses synthetic streams rather than
captured game submissions.

## Format references

- [NVIDIA FIFO_DMA reference](https://github.com/NVIDIA/open-gpu-doc/blob/master/manuals/turing/tu104/dev_ram.ref.txt)
- [NVIDIA Ampere GPFIFO class header](https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/host/clc56f.h)
