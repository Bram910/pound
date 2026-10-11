#!/usr/bin/env python3
"""Check packed dot products and wide/high multiply-accumulate in a reference lane."""

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
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-dot-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    inputs = ((0, 0, 0, 0), (1, 1, 0xFFFFFFFF, 0),
              (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
              (0x80000000, 0x80000000, 0, 0x80000000),
              (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
              (0xFF01807F, 0x7F80FF01, 0x7FFFFFFF, 0xFFFFFFFF),
              (0x80007FFF, 0x0102FEFF, 0x80000000, 0),
              (0x13579BDF, 0xFEDCBA98, 0xFFFFFFFF, 0x01234567))
    reports = []
    for case, values in enumerate(inputs):
        instructions, operations, fields, registers = [], [], [], {}

        def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7, **checked):
            instructions.append((opcode | predicate << 12 | destination << 16 | first << 24,
                                 second, third, 0xFC000))
            operations.append(operation)
            fields.append(checked)

        def output(destination, value):
            if destination == 255:
                return
            registers.pop(f"R{destination}", None)
            if value & 0xFFFFFFFF:
                registers[f"R{destination}"] = value & 0xFFFFFFFF

        def read(source):
            return registers.get(f"R{source}", 0)

        def dot(destination, two, high, signed0, signed1):
            first, second, accumulator = read(1), read(2), read(4)
            emit("IDP2" if two else "IDP4", 0x226, destination, first=1, second=2,
                 third=4 | signed0 << 9 | signed1 << 10 | two << 12 | high << 13,
                 dot_signed=[signed0, signed1], dot_high=high)
            total = accumulator
            width = 16 if two else 8
            for element in range(2 if two else 4):
                left = first >> (width * element) & ((1 << width) - 1)
                right = second >> (8 * (element + 2 * high)) & 255
                if signed0 and left >= 1 << (width - 1):
                    left -= 1 << width
                if signed1 and right >= 128:
                    right -= 256
                total += left * right
            output(destination, total)

        def multiply(destination, mode, signed, immediate=None, accumulator=4):
            first = read(1)
            second = read(2) if immediate is None else immediate
            accumulated = read(accumulator) | read(accumulator + 1) << 32 if accumulator != 255 else 0
            left = first - (1 << 32) if signed and first & 0x80000000 else first
            right = second - (1 << 32) if signed and second & 0x80000000 else second
            total = (left * right + accumulated) & 0xFFFFFFFFFFFFFFFF
            emit("IMAD", (0x25 if mode == 1 else 0x27) | (1 if immediate is None else 4) << 9,
                 destination, first=1, second=2 if immediate is None else immediate,
                 third=accumulator | signed << 9 | 0x078E0000,
                 multiply_mode=mode, compare_signed=signed)
            output(destination, total if mode == 1 else total >> 32)
            if mode == 1 and destination != 255:
                output(destination + 1, total >> 32)

        for destination, value in zip((1, 2, 4, 5), values):
            emit("MOV", 0x802, destination, second=value, third=0xF00)
            output(destination, value)
        destination = 10
        for two in range(2):
            for high in range(2) if two else (0,):
                for signed0 in range(2):
                    for signed1 in range(2):
                        dot(destination, two, high, signed0, signed1)
                        destination += 1
        for signed in range(2):
            for index, immediate in enumerate((None, 0, 1, 0x80000000, 0xFFFFFFFF)):
                multiply(40 + signed * 10 + index * 2, 1, signed, immediate)
                multiply(80 + signed * 5 + index, 2, signed, immediate)
        multiply(90, 1, 0, accumulator=255)
        multiply(255, 1, 1)
        multiply(4, 1, 1)
        multiply(1, 2, 1)
        dot(2, 1, 1, 1, 1)
        emit("IMAD", 0x225, 100, first=1, second=2, third=0x078E0004, predicate=0,
             multiply_mode=1)
        emit("EXIT", 0x94D, third=0x03800000)
        path = args.output_dir / f"dot-{case}.bin"
        path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in instructions))
        trace = subprocess.run([tool, "trace", str(path)], capture_output=True, text=True, check=True, timeout=30)
        decoded = [json.loads(line) for line in trace.stdout.splitlines()]
        if [instruction["operation"] for instruction in decoded] != operations:
            raise RuntimeError("Dot/multiply instruction sequence differs")
        for instruction, wanted in zip(decoded, fields):
            if any(instruction[key] != value for key, value in wanted.items()):
                raise RuntimeError(f"Instruction fields differ: {instruction}, expected {wanted}")
        run = subprocess.run([tool, "run", str(path), "1000"], capture_output=True, text=True, check=True, timeout=30)
        lane = json.loads(run.stdout)
        wanted = {"pc": len(instructions) * 16, "steps": len(instructions),
                  "halted": 1, "predicates": 0, "registers": registers}
        if lane != wanted:
            raise RuntimeError(f"Dot/multiply result differs: {lane}, expected {wanted}")
        nvidia_checked = False
        if args.nvdisasm:
            official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86", "--no-dataflow",
                                       "--print-raw", str(path)], capture_output=True, text=True, check=True, timeout=30)
            listing = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
            expected_listing = ["IDP" if operation in ("IDP2", "IDP4") else operation for operation in operations]
            if listing != expected_listing:
                raise RuntimeError(f"NVIDIA SM86 listing differs: {official.stdout}")
            (args.output_dir / f"nvidia-dot-{case}.txt").write_text(official.stdout, encoding="utf-8")
            nvidia_checked = True
        reports.append({"case": case, "nvidia_sm86_disassembly_checked": nvidia_checked,
                        "instructions": decoded, "result": lane})
    report = {"kind": "single-lane packed/wide arithmetic reference, not Vulkan execution", "programs": reports}
    (args.output_dir / "dot-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified eight programs: all packed dot sign modes and byte selections, signed/unsigned")
    print("wide/high multiply, 64-bit accumulator carries, immediate multiplicands, aliases, RZ and predication.")


if __name__ == "__main__":
    main()
