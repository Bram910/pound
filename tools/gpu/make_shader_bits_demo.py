#!/usr/bin/env python3
"""Verify SM86 shift modes, leading-bit edge cases and signed byte permutation."""

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
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-bits-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    cases = []
    low, high = 0x807F0180, 0xFEDCBA98
    for amount in (0, 1, 31, 32, 33, 63, 64, 65, 0xFFFFFFFF):
        instructions, operations = [], []
        expected = {"R1": low, "R2": high}
        if amount:
            expected["R3"] = amount

        def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7):
            instructions.append((opcode | predicate << 12 | destination << 16 | first << 24,
                                 second, third, 0xFC000))
            operations.append(operation)

        emit("MOV", 0x802, 1, second=low, third=0xF00)
        emit("MOV", 0x802, 2, second=high, third=0xF00)
        emit("MOV", 0x802, 3, second=amount, third=0xF00)
        for mode in range(32):
            kind, wrap, right, output_high = mode & 3, mode >> 2 & 1, mode >> 3 & 1, mode >> 4 & 1
            emit("SHF", 0x219, 10 + mode, first=1, second=3,
                 third=2 | kind << 9 | wrap << 11 | right << 12 | output_high << 16)
            width = 64 if kind < 2 else 32
            distance = amount & (width - 1) if wrap else min(amount, width)
            value = low | high << 32
            if right:
                if kind in (0, 2):
                    value -= 1 << 64
                value >>= distance
            else:
                value <<= distance
            result = (value >> 32 if output_high else value) & 0xFFFFFFFF
            if result:
                expected[f"R{10 + mode}"] = result
        emit("FLO", 0x900, 50, second=0, third=0xE0400)
        expected["R50"] = 0xFFFFFFFF
        emit("FLO", 0x900, 51, second=0xFFFFFFFF, third=0xE0600)
        expected["R51"] = 0xFFFFFFFF
        emit("FLO", 0x300, 52, second=1, third=0xE0000)
        expected["R52"] = low.bit_length() - 1
        emit("FLO", 0x300, 53, second=2, third=0xE0600)
        expected["R53"] = 32 - (high ^ 0xFFFFFFFF).bit_length()
        emit("FLO", 0x300, 54, second=2 | 1 << 31, third=0xE0000)
        expected["R54"] = (high ^ 0xFFFFFFFF).bit_length() - 1
        for destination, selector in ((60, 0x0123), (61, 0x5410), (62, 0xBA98)):
            emit("PRMT", 0x816, destination, first=1, second=selector, third=2)
            source = low.to_bytes(4, "little") + high.to_bytes(4, "little")
            result = bytearray()
            for octet in range(4):
                choice = selector >> (4 * octet) & 15
                byte = source[choice & 7]
                result.append((255 if byte & 128 else 0) if choice & 8 else byte)
            value = int.from_bytes(result, "little")
            if value:
                expected[f"R{destination}"] = value
        emit("MOV", 0x802, 4, second=0x7654, third=0xF00)
        expected["R4"] = 0x7654
        emit("PRMT", 0x416, 63, first=1, second=0x80808080, third=4)
        expected["R63"] = 0x80808080
        emit("SHF", 0x419, 64, first=1, second=0xFFFFFFFF, third=3 | 0x1E00)
        expected["R64"] = ((low | 0xFFFFFFFF << 32) >> (amount & 31)) & 0xFFFFFFFF
        emit("SHF", 0x819, 90, first=1, second=1, third=2 | 0x1E00, predicate=0)
        emit("EXIT", 0x94D, third=0x03800000)
        path = args.output_dir / f"bit-program-{amount}.bin"
        path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in instructions))
        trace = subprocess.run([tool, "trace", str(path)], capture_output=True,
                               text=True, check=True, timeout=30)
        decoded = [json.loads(line) for line in trace.stdout.splitlines()]
        if [instruction["operation"] for instruction in decoded] != operations:
            raise RuntimeError("Bit-operation sequence differs")
        run = subprocess.run([tool, "run", str(path), "100"], capture_output=True,
                             text=True, check=True, timeout=30)
        lane = json.loads(run.stdout)
        if lane != {"pc": len(instructions) * 16, "steps": len(instructions),
                    "halted": 1, "predicates": 0, "registers": expected}:
            raise RuntimeError(f"Bit-operation results differ: {lane}, expected {expected}")
        nvidia_checked = False
        if args.nvdisasm:
            official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86",
                                       "--no-dataflow", "--print-raw", str(path)],
                                      capture_output=True, text=True, check=True, timeout=30)
            listing = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
            if listing != operations:
                raise RuntimeError(f"NVIDIA SM86 listing differs: {official.stdout}")
            (args.output_dir / f"nvidia-sm86-{amount}.txt").write_text(official.stdout, encoding="utf-8")
            nvidia_checked = True
        cases.append({"amount": amount, "instructions": decoded, "result": lane,
                      "nvidia_sm86_disassembly_checked": nvidia_checked})
    report = {"kind": "single-lane bit-operation reference, not Vulkan execution", "cases": cases}
    (args.output_dir / "bits-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified nine shift-boundary programs, all 32 signed/unsigned funnel-shift modes,")
    print("FLO zero/no-sign-bit cases, complement, signed byte replication, immediate source forms and predication.")


if __name__ == "__main__":
    main()
