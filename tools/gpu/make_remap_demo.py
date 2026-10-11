#!/usr/bin/env python3
"""Verify a DMA clear, RGB channel remapping and RGB-to-RGBA expansion."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess

from make_copy_demo import write_png
from make_tiled_demo import tiled_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-remap-demo"), type=Path)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--block-height-log2", type=int, choices=range(6), default=3)
    parser.add_argument("--sector-layout", choices=("16bx2", "turing2d"), default="16bx2")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 256, 128
    source, preserved, expanded, semaphore, command_address = 0x1000, 0x40000, 0x70000, 0xF0000, 0x100
    readback = expanded
    if args.tiled:
        expanded, readback, semaphore = 0x100000, 0x1C0000, 0x1FF000
    surface_width, surface_height = width + 17, height + 11
    preserved_origin, expanded_origin = (3, 5), (7, 3)
    preserved_height = args.block_height_log2
    expanded_height = (preserved_height + 1) % 6
    destination_pitch = width * 4 + 16
    payload = 0xC0DEC0DE
    rgb = bytes(component for row in range(height) for column in range(width)
                for component in (column, row * 2, 255 * ((column // 16 ^ row // 16) & 1)))
    image = bytearray(b"\xA5") * ((2 if args.tiled else 1) * 1024 * 1024)
    image[source:source + len(rgb)] = rgb
    words = []

    def methods(method, values):
        words.append((1 << 29) | (len(values) << 16) | (4 << 13) | (method // 4))
        words.extend(values)

    def launch(flags):
        words.append((4 << 29) | (flags << 16) | (4 << 13) | (0x300 // 4))

    def surface(method, height_log2, origin):
        methods(method, [0x1000 | (height_log2 << 4), surface_width, surface_height,
                         1, 0, origin[0] | (origin[1] << 16)])

    methods(0, [0xC6B5])
    methods(0x400, [0x1FFFF, 0xFFFFFFFF, 0, preserved, 0, destination_pitch, width, height])
    methods(0x700, [0x7F181008, 0, 0x00034444])
    if args.tiled:
        surface(0x70C, preserved_height, preserved_origin)
    launch(0x601 if args.tiled else 0x781)
    methods(0x400, [0, source, 0, preserved, width * 3, destination_pitch, width, height])
    methods(0x708, [0x03206012])
    launch(0x682 if args.tiled else 0x782)
    methods(0x408, [0, expanded])
    methods(0x700, [255, 0, 0x03204012])
    if args.tiled:
        surface(0x70C, expanded_height, expanded_origin)
        launch(0x681)
        methods(0x400, [0, expanded, 0, readback, 0, destination_pitch, width, height])
        surface(0x728, expanded_height, expanded_origin)
        methods(0x708, [0x03303210])
    methods(0x240, [0, semaphore, payload])
    launch(0x709 if args.tiled else 0x789)

    commands = struct.pack("<" + "I" * len(words), *words)
    image[command_address:command_address + len(commands)] = commands
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(struct.pack("<II", command_address, len(words) << 10))
    replay_command = [str(args.replay.resolve()), str(memory_path), str(entries_path), str(output_path)]
    if args.tiled:
        replay_command.append(f"--sector-layout={args.sector_layout}")
    subprocess.run(replay_command, check=True)

    expected = bytearray(image)
    for row in range(height):
        for column in range(width):
            offset = (row * width + column) * 3
            red, green, blue = rgb[offset:offset + 3]
            for address, alpha, height_log2, origin in (
                    (preserved, 127, preserved_height, preserved_origin),
                    (expanded, 255, expanded_height, expanded_origin)):
                rgba = bytes((blue, green, red, alpha))
                if args.tiled:
                    for component, value in enumerate(rgba):
                        offset = tiled_offset(surface_width * 4, height_log2,
                                              (column + origin[0]) * 4 + component, row + origin[1],
                                              args.sector_layout)
                        expected[address + offset] = value
                else:
                    start = address + row * destination_pitch + column * 4
                    expected[start:start + 4] = rgba
            if args.tiled:
                start = readback + row * destination_pitch + column * 4
                expected[start:start + 4] = bytes((blue, green, red, 255))
    struct.pack_into("<I", expected, semaphore, payload)
    result = output_path.read_bytes()
    if result != expected:
        raise RuntimeError("Fill, remap, preserved alpha, expanded alpha, padding or semaphore is incorrect")
    copied = b"".join(result[readback + row * destination_pitch:
                             readback + row * destination_pitch + width * 4]
                      for row in range(height))
    png_path = args.output_dir / "remapped-texture.png"
    write_png(png_path, width, height, copied)
    print(f"Verified clear, remap and expansion for {width * height} pixels each")
    if args.tiled:
        print(f"Verified tiled intermediate bytes and readback, block heights "
              f"{1 << preserved_height}/{1 << expanded_height} GOBs")
    print(f"SHA-256 {hashlib.sha256(copied).hexdigest()}")
    print(f"Saved {png_path}; this is a DMA remap preview, not a rendered scene.")


if __name__ == "__main__":
    main()
