#!/usr/bin/env python3
"""Verify SM86 specialized byte permutations, bit extraction and mask generation."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-extract-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instructions, operations, expected = [], [], {}
    low, high = 0x80FF7F80, 0x81FE0100
    source_bytes = low.to_bytes(4, "little") + high.to_bytes(4, "little")
    selections = (
        ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)),
        ((0, 7, 6, 5), (1, 0, 7, 6), (2, 1, 0, 7), (3, 2, 1, 0)),
        ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)),
        ((0, 1, 2, 3), (1, 1, 2, 3), (2, 2, 2, 3), (3, 3, 3, 3)),
        ((0, 0, 0, 0), (0, 1, 1, 1), (0, 1, 2, 2), (0, 1, 2, 3)),
        ((0, 1, 0, 1), (2, 3, 2, 3), (0, 1, 0, 1), (2, 3, 2, 3)),
    )

    def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7):
        instructions.append((opcode | predicate << 12 | destination << 16 | first << 24,
                             second, third, 0xFC000))
        operations.append(operation)

    def output(destination, value):
        if value:
            expected[f"R{destination}"] = value
        else:
            expected.pop(f"R{destination}", None)

    for destination, value in ((1, low), (2, high)):
        emit("MOV", 0x802, destination, second=value, third=0xF00)
        output(destination, value)
    for mode, selectors in enumerate(selections, 1):
        for selector, indices in enumerate(selectors):
            destination = 10 + (mode - 1) * 4 + selector
            emit("PRMT", 0x816, destination, first=1, second=0xFEDCBA00 | selector,
                 third=2 | mode << 8)
            output(destination, int.from_bytes(bytes(source_bytes[index] for index in indices), "little"))
    widths = (0, 1, 8, 16, 31, 32, 0xFFFFFFFF)
    for mode in range(4):
        signed, wrap = mode & 1, mode >> 1
        for index, width in enumerate(widths):
            destination = 40 + mode * len(widths) + index
            emit("SGXT", 0x81A, destination, first=1, second=width, third=signed << 9 | wrap << 11)
            count = width % 32 if wrap else min(width, 32)
            value = low % (1 << count)
            if signed and count and value & (1 << (count - 1)):
                value = (value - (1 << count)) & 0xFFFFFFFF
            output(destination, value)
    boundaries = (0, 1, 31, 32, 33, 0xFFFFFFFF)
    for wrap in range(2):
        for position_index, position in enumerate(boundaries):
            emit("MOV", 0x802, 5, second=position, third=0xF00)
            output(5, position)
            for width_index, width in enumerate(boundaries):
                destination = 100 + wrap * 36 + position_index * 6 + width_index
                emit("BMSK", 0x81B, destination, first=5, second=width, third=wrap << 11)
                start, length = (position % 32, width % 32) if wrap else (position, width)
                value = sum(1 << bit for bit in range(32) if start <= bit < start + length)
                output(destination, value)
    emit("PRMT", 0x416, 180, first=1, second=0xFF7F8000, third=5 | 2 << 8)
    output(180, int.from_bytes(low.to_bytes(4, "little")[::-1], "little"))
    emit("SGXT", 0x81A, 1, first=1, second=8, third=0x200)
    output(1, 0xFFFFFF80)
    emit("BMSK", 0x81B, 190, first=1, second=32, predicate=0)
    emit("EXIT", 0x94D, third=0x03800000)
    path = args.output_dir / "extraction-program.bin"
    path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in instructions))
    tool = str(args.shader_tool.resolve())
    trace = subprocess.run([tool, "trace", str(path)], capture_output=True, text=True, check=True, timeout=30)
    decoded = [json.loads(line) for line in trace.stdout.splitlines()]
    if [instruction["operation"] for instruction in decoded] != operations:
        raise RuntimeError("Extraction instruction sequence differs")
    run = subprocess.run([tool, "run", str(path), "1000"], capture_output=True, text=True, check=True, timeout=30)
    lane = json.loads(run.stdout)
    if lane != {"pc": len(instructions) * 16, "steps": len(instructions),
                "halted": 1, "predicates": 0, "registers": expected}:
        raise RuntimeError(f"Extraction result differs: {lane}, expected {expected}")
    nvidia_checked = False
    if args.nvdisasm:
        official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86", "--no-dataflow",
                                   "--print-raw", str(path)], capture_output=True, text=True, check=True, timeout=30)
        listing = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
        if listing != operations:
            raise RuntimeError(f"NVIDIA SM86 listing differs: {official.stdout}")
        (args.output_dir / "nvidia-sm86.txt").write_text(official.stdout, encoding="utf-8")
        nvidia_checked = True
    report = {"kind": "single-lane extraction reference, not Vulkan execution",
              "nvidia_sm86_disassembly_checked": nvidia_checked, "instructions": decoded, "result": lane}
    (args.output_dir / "extraction-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(instructions)} instructions: all six specialized permutations, extraction and mask")
    print("boundary cases, wrap/clamp, signed/unsigned, immediate high operand, source alias and predication.")


if __name__ == "__main__":
    main()
