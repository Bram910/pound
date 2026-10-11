#!/usr/bin/env python3
"""Verify SM86 LEA and integer-add carry chains in the single-lane executor."""

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
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-address-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    reports = []
    for case, (low, addend, high) in enumerate(((0, 0, 0), (0xFFFFFFFF, 1, 0x80000000),
                                               (0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
                                               (0x13579BDF, 0xFEDCBA98, 0x2468ACE0))):
        instructions, operations, registers = [], [], {}
        predicates = 0

        def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7):
            instructions.append((opcode | predicate << 12 | destination << 16 | first << 24,
                                 second, third, 0xFC000))
            operations.append(operation)

        def output(destination, value):
            if destination == 255:
                return
            registers.pop(f"R{destination}", None)
            if value & 0xFFFFFFFF:
                registers[f"R{destination}"] = value & 0xFFFFFFFF

        def predicate_output(destination, value):
            nonlocal predicates
            if destination != 7:
                predicates = (predicates & ~(1 << destination)) | int(value) << destination

        for destination, value in ((1, low), (2, addend), (3, high)):
            emit("MOV", 0x802, destination, second=value, third=0xF00)
            output(destination, value)
        destination = 10
        for extended in range(2):
            for upper in range(2):
                for shift in (0, 1, 16, 31):
                    for modifier in range(3):
                        third = (3 if upper else 255) | shift << 11 | upper << 16
                        third |= (modifier == 1) << 8 | extended << 10
                        if extended:
                            third |= 7 << 23
                        emit("LEA", 0x211, destination, first=1,
                             second=2 | (modifier == 2) << 31, third=third)
                        shifted = (((low | high << 32) >> (32 - shift)) & 0xFFFFFFFF
                                   if upper else (low << shift) & 0xFFFFFFFF)
                        if extended:
                            total = ((shifted ^ (0xFFFFFFFF if modifier == 1 else 0))
                                     + (addend ^ (0xFFFFFFFF if modifier == 2 else 0)) + 1)
                        else:
                            total = ((0x100000000 - shifted if modifier == 1 else shifted)
                                     + (0x100000000 - addend if modifier == 2 else addend))
                        output(destination, total)
                        predicate_output(0, total > 0xFFFFFFFF)
                        destination += 1
        emit("LEA", 0x811, 80, first=1, second=addend, third=255 | 1 << 17)
        output(80, low + addend)
        predicate_output(1, low + addend > 0xFFFFFFFF)
        carry = (predicates >> 1) & 1
        emit("LEA", 0x811, 81, first=1, second=0, third=3 | 1 << 16 | 0x400 | 1 << 17 | 1 << 23)
        output(81, high + carry)
        predicate_output(1, high + carry > 0xFFFFFFFF)
        emit("LEA", 0x411, 82, first=1, second=high, third=2 | 1 << 16 | 31 << 11 | 7 << 17)
        output(82, ((low | high << 32) >> 1) + addend)
        emit("IADD3", 0x210, 90, first=1, second=2,
             third=3 | 0x0781E000 | 2 << 17 | 3 << 20)
        total = low + addend + high
        output(90, total)
        predicate_output(2, total >= 1 << 32)
        predicate_output(3, total >= 2 << 32)
        carry = ((predicates >> 2) & 1) + ((predicates >> 3) & 1)
        emit("IADD3", 0x810, 91, first=255, second=0xFFFFFFFF,
             third=255 | 0x400 | 2 << 13 | 2 << 17 | 3 << 20 | 3 << 23)
        output(91, 0xFFFFFFFF + carry)
        predicate_output(2, 0xFFFFFFFF + carry >= 1 << 32)
        predicate_output(3, 0xFFFFFFFF + carry >= 2 << 32)
        emit("SEL", 0x807, 92, first=255, second=1, third=0x04000000 | 2 << 23)
        output(92, (predicates >> 2) & 1)
        emit("LEA", 0x211, 1, first=1, second=2, third=255 | 7 << 17)
        output(1, low + addend)
        emit("LEA", 0x811, 100, first=1, second=1, third=255 | 7 << 17, predicate=6)
        emit("IADD3", 0x210, 255, first=255, second=255,
             third=255 | 0x0781E000 | 1 << 8 | 4 << 17 | 5 << 20)
        predicate_output(4, True)
        predicate_output(5, False)
        emit("EXIT", 0x94D, third=0x03800000)
        path = args.output_dir / f"address-{case}.bin"
        path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in instructions))
        trace = subprocess.run([tool, "trace", str(path)], capture_output=True, text=True, check=True, timeout=30)
        decoded = [json.loads(line) for line in trace.stdout.splitlines()]
        if [instruction["operation"] for instruction in decoded] != operations:
            raise RuntimeError("Address instruction sequence differs")
        run = subprocess.run([tool, "run", str(path), "1000"], capture_output=True, text=True, check=True, timeout=30)
        lane = json.loads(run.stdout)
        wanted = {"pc": len(instructions) * 16, "steps": len(instructions),
                  "halted": 1, "predicates": predicates, "registers": registers}
        if lane != wanted:
            raise RuntimeError(f"Address result differs: {lane}, expected {wanted}")
        nvidia_checked = False
        if args.nvdisasm:
            official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86", "--no-dataflow",
                                       "--print-raw", str(path)], capture_output=True, text=True, check=True, timeout=30)
            listing = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
            if listing != operations:
                raise RuntimeError(f"NVIDIA SM86 listing differs: {official.stdout}")
            (args.output_dir / f"nvidia-address-{case}.txt").write_text(official.stdout, encoding="utf-8")
            nvidia_checked = True
        reports.append({"case": case, "nvidia_sm86_disassembly_checked": nvidia_checked,
                        "instructions": decoded, "result": lane})
    report = {"kind": "single-lane address arithmetic reference, not Vulkan execution", "programs": reports}
    (args.output_dir / "address-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified four address programs: LEA low/high, boundary shifts/modifiers, carry chaining,")
    print("IADD3 dual carries, predicate input/output aliasing, negated zero, RZ and predication.")


if __name__ == "__main__":
    main()
