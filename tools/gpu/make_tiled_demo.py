#!/usr/bin/env python3
"""Verify pitched-to-tiled, tiled-to-tiled and tiled-to-pitched DMA transfers."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess

from make_copy_demo import write_png


def tiled_offset(width, height_log2, column, row, sector_layout="16bx2"):
    block_gobs = 1 << height_log2
    width_gobs = (width + 63) // 64
    gob = ((row // (8 * block_gobs)) * width_gobs * block_gobs
           + (column // 64) * block_gobs + (row % (8 * block_gobs)) // 8) * 512
    if sector_layout == "turing2d":
        return (gob + (column % 64 // 32) * 256 + (row % 8 // 4) * 128
                + (column % 32 // 16) * 64 + (row % 4) * 16 + column % 16)
    if sector_layout != "16bx2":
        raise ValueError("Unsupported sector layout")
    return (gob + (column % 64 // 32) * 256 + (row % 8 // 2) * 64
            + (column % 32 // 16) * 32 + (row % 2) * 16 + column % 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-tiled-demo"), type=Path)
    parser.add_argument("--block-height-log2", type=int, choices=range(6), default=3)
    parser.add_argument("--sector-layout", choices=("16bx2", "turing2d"), default="16bx2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 256, 128
    row_bytes = width * 4
    source, first, second, destination = 0x1000, 0x40000, 0x100000, 0x1C0000
    semaphore, command_address = 0x1FF000, 0x100
    destination_pitch = row_bytes + 16
    surface_width, surface_height = row_bytes + 37, height + 11
    first_origin, second_origin = (13, 5), (21, 3)
    first_height = args.block_height_log2
    second_height = (first_height + 1) % 6
    payload = 0xC0DEC0DE
    pixels = bytes(component for row in range(height) for column in range(width)
                   for component in (column, row * 2,
                                     255 * ((column // 16 ^ row // 16) & 1), 255))
    image = bytearray(b"\xA5") * (2 * 1024 * 1024)
    image[source:source + len(pixels)] = pixels

    words = []

    def methods(method, values):
        words.append((1 << 29) | (len(values) << 16) | (4 << 13) | (method // 4))
        words.extend(values)

    def surface(method, height_log2, origin):
        methods(method, [0x1000 | (height_log2 << 4), surface_width, surface_height,
                         1, 0, origin[0] | (origin[1] << 16)])

    def launch(flags):
        words.append((4 << 29) | (flags << 16) | (4 << 13) | (0x300 // 4))

    methods(0, [0xC6B5])
    methods(0x400, [0, source, 0, first, row_bytes, 0, row_bytes, height])
    surface(0x70C, first_height, first_origin)
    launch(0x281)
    methods(0x400, [0, first, 0, second, 0, 0, row_bytes, height])
    surface(0x728, first_height, first_origin)
    surface(0x70C, second_height, second_origin)
    launch(0x202)
    methods(0x400, [0, second, 0, destination, 0, destination_pitch, row_bytes, height])
    surface(0x728, second_height, second_origin)
    methods(0x240, [0, semaphore, payload])
    launch(0x309)

    commands = struct.pack("<" + "I" * len(words), *words)
    image[command_address:command_address + len(commands)] = commands
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(struct.pack("<II", command_address, len(words) << 10))
    subprocess.run([str(args.replay.resolve()), str(memory_path), str(entries_path),
                    str(output_path), f"--sector-layout={args.sector_layout}"], check=True)

    expected = bytearray(image)
    for row in range(height):
        for column in range(row_bytes):
            value = pixels[row * row_bytes + column]
            for address, height_log2, origin in (
                    (first, first_height, first_origin), (second, second_height, second_origin)):
                offset = tiled_offset(surface_width, height_log2,
                                      column + origin[0], row + origin[1], args.sector_layout)
                expected[address + offset] = value
        start = destination + row * destination_pitch
        expected[start:start + row_bytes] = pixels[row * row_bytes:(row + 1) * row_bytes]
    struct.pack_into("<I", expected, semaphore, payload)
    result = output_path.read_bytes()
    if result != expected:
        raise RuntimeError("Tiled upload, tiled copy, readback, padding or semaphore is incorrect")
    copied = b"".join(result[destination + row * destination_pitch:
                             destination + row * destination_pitch + row_bytes]
                      for row in range(height))
    png_path = args.output_dir / "copied-texture.png"
    write_png(png_path, width, height, copied)
    print(f"Verified three {len(copied)}-byte transfers, block heights "
          f"{1 << first_height}/{1 << second_height} GOBs and nonzero origins")
    print(f"SHA-256 {hashlib.sha256(copied).hexdigest()}")
    print(f"Saved {png_path}; this is a DMA layout preview, not a rendered scene.")


if __name__ == "__main__":
    main()
