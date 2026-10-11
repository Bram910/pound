#!/usr/bin/env python3
"""Replay a synthetic GPU texture copy and save the copied pixels as a PNG."""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import zlib


def png_chunk(kind, data):
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_png(path, width, height, rgba):
    rows = b"".join(b"\x00" + rgba[row * width * 4:(row + 1) * width * 4]
                    for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header)
                     + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 256, 128
    row_bytes = width * 4
    source, destination, semaphore, command_address = 0x1000, 0x40000, 0xF0000, 0x100
    destination_pitch = row_bytes + 16
    payload = 0xC0DEC0DE
    pixels = bytes(component for row in range(height) for column in range(width)
                   for component in (column, row * 2,
                                     255 * ((column // 16 ^ row // 16) & 1), 255))
    image = bytearray(1024 * 1024)
    image[source:source + len(pixels)] = pixels

    def header(method, count):
        return (1 << 29) | (count << 16) | (4 << 13) | (method // 4)

    words = [header(0, 1), 0xC6B5,
             header(0x400, 8), 0, source, 0, destination,
             row_bytes, destination_pitch, row_bytes, height,
             header(0x240, 3), 0, semaphore, payload,
             (4 << 29) | (0x389 << 16) | (4 << 13) | (0x300 // 4)]
    commands = struct.pack("<" + "I" * len(words), *words)
    image[command_address:command_address + len(commands)] = commands
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(struct.pack("<II", command_address, len(words) << 10))
    subprocess.run([str(args.replay.resolve()), str(memory_path), str(entries_path),
                    str(output_path)], check=True)

    result = output_path.read_bytes()
    if len(result) != len(image):
        raise RuntimeError("Replay changed the memory image size")
    copied = b"".join(result[destination + row * destination_pitch:
                             destination + row * destination_pitch + row_bytes]
                      for row in range(height))
    if copied != pixels or struct.unpack_from("<I", result, semaphore)[0] != payload:
        raise RuntimeError("Texture copy or completion semaphore is incorrect")
    expected = bytearray(image)
    for row in range(height):
        start = destination + row * destination_pitch
        expected[start:start + row_bytes] = pixels[row * row_bytes:(row + 1) * row_bytes]
    struct.pack_into("<I", expected, semaphore, payload)
    if result != expected:
        raise RuntimeError("Replay wrote outside the destination rows and semaphore")

    png_path = args.output_dir / "copied-texture.png"
    write_png(png_path, width, height, copied)
    print(f"Verified {len(copied)} copied bytes; SHA-256 {hashlib.sha256(copied).hexdigest()}")
    print(f"Saved {png_path}; this is a DMA copy preview, not a rendered scene.")


if __name__ == "__main__":
    main()
