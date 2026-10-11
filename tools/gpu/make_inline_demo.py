#!/usr/bin/env python3
"""Verify segmented inline uploads, volume copies and semaphore reductions."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess

from make_copy_demo import write_png
from make_volume_demo import volume_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-inline-demo"), type=Path)
    parser.add_argument("--block-height-log2", type=int, choices=range(6), default=2)
    parser.add_argument("--block-depth-log2", type=int, choices=range(6), default=2)
    parser.add_argument("--sector-layout", choices=("16bx2", "turing2d"), default="turing2d")
    parser.add_argument("--segment-words", type=int, default=97)
    parser.add_argument("--packet-style", choices=("increment-once", "non-incrementing"),
                        default="increment-once")
    args = parser.parse_args()
    if not 1 <= args.segment_words <= 0x1FFFFF:
        parser.error("--segment-words must be between 1 and 2097151")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height, depth = 80, 36, 4
    row_bytes, pitch = width * 4, width * 4 + 16
    surface_width, surface_height = row_bytes + 35, height + 7
    origin = (13, 3)
    linear, volume, readback = 0x80000, 0x100000, 0x700000
    semaphore, counter, command_address = 0x7FF000, 0x7FF010, 0x100
    pixels = bytes(component for layer in range(depth) for row in range(height)
                   for column in range(width) for component in
                   (column * 3, row * 7, layer * 70, 255))
    image = bytearray(b"\xA5") * (8 * 1024 * 1024)
    struct.pack_into("<I", image, counter, 0)
    words = []

    def methods(subchannel, method, values):
        words.append((1 << 29) | (len(values) << 16) | (subchannel << 13) | (method // 4))
        words.extend(values)

    def surface(method, layer):
        methods(4, method, [0x1000 | (args.block_height_log2 << 4)
                           | (args.block_depth_log2 << 8), surface_width,
                           surface_height, depth, layer, origin[0] | (origin[1] << 16)])

    methods(0, 0, [0xA140])
    methods(4, 0, [0xC6B5])
    for layer in range(depth):
        destination = linear if layer == 0 else volume
        methods(0, 0x180, [row_bytes, height, 0, destination, pitch,
                           (args.block_height_log2 << 4) | (args.block_depth_log2 << 8),
                           surface_width, surface_height, depth, layer, *origin])
        methods(0, 0x1DC, [0, semaphore, 100 + layer])
        launch_flags = 0x1021 if layer == 0 else 0x1020
        payload = pixels[layer * row_bytes * height:(layer + 1) * row_bytes * height]
        payload_words = struct.unpack("<" + "I" * (len(payload) // 4), payload)
        if args.packet_style == "increment-once":
            words.append((5 << 29) | ((len(payload_words) + 1) << 16) | (0x1B0 // 4))
            words.append(launch_flags)
        else:
            methods(0, 0x1B0, [launch_flags])
            words.append((3 << 29) | (len(payload_words) << 16) | (0x1B4 // 4))
        words.extend(payload_words)
        methods(0, 0x5C, [semaphore, 0, 100 + layer, 0, 2])
        if layer == 0:
            methods(4, 0x400, [0, linear, 0, volume, pitch, 0, row_bytes, height])
            surface(0x70C, layer)
            methods(4, 0x300, [0x281])
        methods(4, 0x400, [0, volume, 0, readback + layer * pitch * height,
                           0, pitch, row_bytes, height])
        surface(0x728, layer)
        methods(4, 0x300, [0x301])
        methods(0, 0x5C, [counter, 0, 1, 0, 0xA8000006])
    methods(0, 0x64, [depth, 0, 3])
    methods(0, 0x64, [depth, 0, 4])
    methods(0, 0x64, [0, 0, 5])
    methods(0, 0x78, [1])
    commands = struct.pack("<" + "I" * len(words), *words)
    if command_address + len(commands) > linear:
        raise RuntimeError("Command buffer overlaps upload destination")
    image[command_address:command_address + len(commands)] = commands
    entries = bytearray()
    for first_word in range(0, len(words), args.segment_words):
        count = min(args.segment_words, len(words) - first_word)
        entries.extend(struct.pack("<II", command_address + first_word * 4,
                                   (count << 10) | 0x80000000))
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(entries)
    subprocess.run([str(args.replay.resolve()), str(memory_path), str(entries_path),
                    str(output_path), f"--sector-layout={args.sector_layout}"], check=True)

    expected = bytearray(image)
    for layer in range(depth):
        for row in range(height):
            for column in range(row_bytes):
                value = pixels[(layer * height + row) * row_bytes + column]
                offset = volume_offset(surface_width, surface_height, args.block_height_log2,
                                       args.block_depth_log2, column + origin[0], row + origin[1],
                                       layer, args.sector_layout)
                expected[volume + offset] = value
                expected[readback + (layer * height + row) * pitch + column] = value
                if layer == 0:
                    expected[linear + row * pitch + column] = value
    struct.pack_into("<I", expected, semaphore, 100 + depth - 1)
    struct.pack_into("<I", expected, counter, depth)
    result = output_path.read_bytes()
    if result != expected:
        raise RuntimeError("Inline upload, volume copy, semaphore, padding or untouched bytes differ")
    contact_sheet = b"".join(result[readback + (layer * height + row) * pitch:
                                    readback + (layer * height + row) * pitch + row_bytes]
                             for row in range(height) for layer in range(depth))
    png_path = args.output_dir / "uploaded-slices.png"
    write_png(png_path, width * depth, height, contact_sheet)
    print(f"Verified {depth} inline uploads and {depth + 1} DMA copies across {len(entries) // 8} segments")
    print(f"SHA-256 {hashlib.sha256(contact_sheet).hexdigest()}")
    print(f"Saved {png_path}; this is a verified upload/readback preview, not a rendered scene.")


if __name__ == "__main__":
    main()
