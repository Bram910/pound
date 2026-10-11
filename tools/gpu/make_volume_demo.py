#!/usr/bin/env python3
"""Verify tiled volume uploads, clears, reversed-slice remaps and pitched readback."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess

from make_copy_demo import write_png
from make_tiled_demo import tiled_offset


def volume_offset(width, height, height_log2, depth_log2, column, row, layer, sector_layout):
    block_height, block_depth = 8 << height_log2, 1 << depth_log2
    width_blocks = (width + 63) // 64
    height_blocks = (height + block_height - 1) // block_height
    block_bytes = 64 * block_height * block_depth
    block_index = ((layer // block_depth) * height_blocks + row // block_height) * width_blocks
    block_index += column // 64
    gob_index = (layer % block_depth) * (block_height // 8) + row % block_height // 8
    in_gob = tiled_offset(64, 0, column % 64, row % 8, sector_layout)
    return block_index * block_bytes + gob_index * 512 + in_gob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-volume-demo"), type=Path)
    parser.add_argument("--block-height-log2", type=int, choices=range(6), default=1)
    parser.add_argument("--block-depth-log2", type=int, choices=range(6), default=1)
    parser.add_argument("--sector-layout", choices=("16bx2", "turing2d"), default="turing2d")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height, depth = 48, 20, 7
    source, first, second, readback = 0x10000, 0x100000, 0x400000, 0x700000
    semaphore, command_address = 0x7FF000, 0x100
    surface_width, surface_height = width + 5, height + 4
    first_origin, second_origin = (3, 2), (2, 1)
    first_height, first_depth = args.block_height_log2, args.block_depth_log2
    second_height, second_depth = (first_height + 1) % 6, (first_depth + 1) % 6
    input_pitch, output_pitch = width * 3, width * 4 + 16
    payload = 0xC0DEC0DE
    rgb = bytes(component for layer in range(depth) for row in range(height)
                for column in range(width) for component in
                (column * 5, row * 12, layer * 36))
    image = bytearray(b"\xA5") * (8 * 1024 * 1024)
    image[source:source + len(rgb)] = rgb
    words = []

    def methods(method, values):
        words.append((1 << 29) | (len(values) << 16) | (4 << 13) | (method // 4))
        words.extend(values)

    def launch(flags):
        words.append((4 << 29) | (flags << 16) | (4 << 13) | (0x300 // 4))

    def surface(method, height_log2, depth_log2, layer, origin):
        methods(method, [0x1000 | (height_log2 << 4) | (depth_log2 << 8),
                         surface_width, surface_height, depth, layer,
                         origin[0] | (origin[1] << 16)])

    methods(0, [0xC6B5])
    methods(0x240, [0, semaphore, payload])
    for layer in range(depth):
        methods(0x400, [0, source + layer * input_pitch * height,
                         0, first, input_pitch, 0, width, height])
        surface(0x70C, first_height, first_depth, layer, first_origin)
        methods(0x700, [255, 0, 0x03204012])
        launch(0x681)

    for layer in range(depth):
        destination_layer = depth - layer - 1
        methods(0x400, [0x1FFFF, 0xFFFFFFFF, 0, second, 0, 0, width, height])
        surface(0x70C, second_height, second_depth, destination_layer, second_origin)
        methods(0x700, [0x7F171513, 0, 0x00034444])
        launch(0x601)
        methods(0x400, [0, first, 0, second, 0, 0, width, height])
        surface(0x728, first_height, first_depth, layer, first_origin)
        methods(0x708, [0x03306012])
        launch(0x602)
        methods(0x400, [0, second, 0, readback + destination_layer * output_pitch * height,
                         0, output_pitch, width, height])
        surface(0x728, second_height, second_depth, destination_layer, second_origin)
        methods(0x708, [0x03303210])
        launch(0x709 if layer == depth - 1 else 0x701)

    commands = struct.pack("<" + "I" * len(words), *words)
    if command_address + len(commands) > source:
        raise RuntimeError("Command buffer overlaps texture input")
    image[command_address:command_address + len(commands)] = commands
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(struct.pack("<II", command_address, len(words) << 10))
    subprocess.run([str(args.replay.resolve()), str(memory_path), str(entries_path),
                    str(output_path), f"--sector-layout={args.sector_layout}"], check=True)

    expected = bytearray(image)
    for layer in range(depth):
        destination_layer = depth - layer - 1
        for row in range(height):
            for column in range(width):
                source_offset = ((layer * height + row) * width + column) * 3
                red, green, blue = rgb[source_offset:source_offset + 3]
                for address, height_log2, depth_log2, target_layer, origin, rgba in (
                        (first, first_height, first_depth, layer, first_origin, (blue, green, red, 255)),
                        (second, second_height, second_depth, destination_layer,
                         second_origin, (red, green, blue, 127))):
                    for component, value in enumerate(rgba):
                        offset = volume_offset(surface_width * 4, surface_height, height_log2,
                                               depth_log2, (column + origin[0]) * 4 + component,
                                               row + origin[1], target_layer, args.sector_layout)
                        expected[address + offset] = value
                start = readback + (destination_layer * height + row) * output_pitch + column * 4
                expected[start:start + 4] = bytes((red, green, blue, 127))
    struct.pack_into("<I", expected, semaphore, payload)
    result = output_path.read_bytes()
    if result != expected:
        raise RuntimeError("Volume upload, clear, slice remap, readback, padding or semaphore is incorrect")
    contact_sheet = b"".join(result[readback + (layer * height + row) * output_pitch:
                                    readback + (layer * height + row) * output_pitch + width * 4]
                             for row in range(height) for layer in range(depth))
    png_path = args.output_dir / "volume-slices.png"
    write_png(png_path, width * depth, height, contact_sheet)
    print(f"Verified {depth} volume slices through {4 * depth} DMA launches ({args.sector_layout})")
    print(f"SHA-256 {hashlib.sha256(contact_sheet).hexdigest()}")
    print(f"Saved {png_path}; this is a DMA volume readback preview, not a rendered scene.")


if __name__ == "__main__":
    main()
