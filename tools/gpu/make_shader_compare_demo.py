#!/usr/bin/env python3
"""Verify SM86 multiword comparisons and logic-generated predicates."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess


def compare(first, second, operation):
    return (False, first < second, first == second, first <= second,
            first > second, first != second, first >= second, True)[operation]


def combine(first, second, operation):
    return (first and second, first or second, first != second)[operation]


class Program:
    def __init__(self):
        self.instructions = []
        self.operations = []
        self.registers = {}
        self.predicates = 0

    def emit(self, operation, opcode, destination=0, first=0, second=0, third=0):
        self.instructions.append((opcode | 7 << 12 | destination << 16 | first << 24,
                                  second, third, 0xFC000))
        self.operations.append(operation)

    def output(self, destination, value):
        if destination != 255:
            self.registers.pop(f"R{destination}", None)
            if value & 0xFFFFFFFF:
                self.registers[f"R{destination}"] = value & 0xFFFFFFFF

    def predicate_output(self, destination, value):
        if destination != 7:
            self.predicates = (self.predicates & ~(1 << destination)) | int(value) << destination

    def move(self, destination, value):
        self.emit("MOV", 0x802, destination, second=value, third=0xF00)
        self.output(destination, value)

    def capture(self, destination, source, value):
        self.emit("SEL", 0x807, destination, first=255, second=1,
                  third=source << 23 | 1 << 26)
        self.output(destination, int(value))

    def finish(self, label, args):
        self.emit("EXIT", 0x94D, third=0x03800000)
        path = args.output_dir / f"{label}.bin"
        path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in self.instructions))
        tool = str(args.shader_tool.resolve())
        trace = subprocess.run([tool, "trace", str(path)], capture_output=True,
                               text=True, check=True, timeout=30)
        decoded = [json.loads(line) for line in trace.stdout.splitlines()]
        if [instruction["operation"] for instruction in decoded] != self.operations:
            raise RuntimeError(f"{label}: decoded operations differ")
        run = subprocess.run([tool, "run", str(path), "10000"], capture_output=True,
                             text=True, check=True, timeout=30)
        lane = json.loads(run.stdout)
        wanted = {"pc": len(self.instructions) * 16, "steps": len(self.instructions),
                  "halted": 1, "predicates": self.predicates, "registers": self.registers}
        if lane != wanted:
            raise RuntimeError(f"{label}: result {lane}, expected {wanted}")
        if args.nvdisasm:
            official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86",
                                       "--no-dataflow", "--print-raw", str(path)],
                                      capture_output=True, text=True, check=True, timeout=30)
            listing = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;",
                                 official.stdout)
            if listing != self.operations:
                raise RuntimeError(f"{label}: NVIDIA SM86 listing differs")
            (args.output_dir / f"nvidia-{label}.txt").write_text(official.stdout, encoding="utf-8")
        return {"case": label, "nvidia_sm86_disassembly_checked": bool(args.nvdisasm),
                "instructions": decoded, "result": lane}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--nvdisasm", type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-compare-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    pairs = ((0, 0), (0, 1), (1, 0), (0xFFFFFFFF, 0x100000000),
             (0x100000000, 0xFFFFFFFF), (0x1234567800000001, 0x1234567800000002),
             (0x12345678FFFFFFFF, 0x1234567800000000),
             (0x7FFFFFFFFFFFFFFF, 0x8000000000000000),
             (0x8000000000000000, 0x7FFFFFFFFFFFFFFF),
             (0xFFFFFFFFFFFFFFFF, 0), (0, 0xFFFFFFFFFFFFFFFF),
             (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF))
    for case, (first, second) in enumerate(pairs):
        program = Program()
        for destination, value in ((1, first & 0xFFFFFFFF), (2, first >> 32),
                                   (3, second & 0xFFFFFFFF), (4, second >> 32)):
            program.move(destination, value)
        destination = 10
        for signed in range(2):
            left = first - (1 << 64) if signed and first >> 63 else first
            right = second - (1 << 64) if signed and second >> 63 else second
            for comparison in range(8):
                for boolean in range(3):
                    low = compare(first & 0xFFFFFFFF, second & 0xFFFFFFFF, comparison)
                    program.emit("ISETP", 0x20C, first=1, second=3,
                                 third=0x70 | comparison << 12 | 1 << 20 | 7 << 23)
                    program.predicate_output(0, low)
                    program.predicate_output(1, not low)
                    condition = not low
                    result = compare(left, right, comparison)
                    program.emit("ISETP", 0x20C, first=2, second=4,
                                 third=1 << 8 | signed << 9 | boolean << 10 | comparison << 12
                                 | 1 << 20 | 1 << 23)
                    primary = combine(result, condition, boolean)
                    secondary = combine(not result, condition, boolean)
                    program.predicate_output(0, primary)
                    program.predicate_output(1, secondary)
                    program.capture(destination, 0, primary)
                    program.capture(destination + 1, 1, secondary)
                    destination += 2
        reports.append(program.finish(f"compare-{case}", args))
    for group in range(4):
        for pand in range(2):
            for invert in range(2):
                program = Program()
                first, second, third = 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0
                for destination, value in ((1, first), (2, second), (3, third)):
                    program.move(destination, value)
                for index in range(64):
                    lookup = group * 64 + index
                    incoming = bool(index & 1)
                    program.emit("ISETP", 0x20C, first=255, second=255,
                                 third=0x70 | (7 if incoming else 0) << 12 | 7 << 20 | 7 << 23)
                    program.predicate_output(0, incoming)
                    value = sum(((lookup >> (((first >> bit & 1) << 2)
                                             | ((second >> bit & 1) << 1)
                                             | (third >> bit & 1))) & 1) << bit for bit in range(32))
                    condition = incoming != bool(invert)
                    result = bool(value) and condition if pand else bool(value) or condition
                    program.emit("LOP3", 0x212, 10 + index, first=1, second=2,
                                 third=3 | lookup << 8 | pand << 16 | invert << 26)
                    program.output(10 + index, value)
                    program.predicate_output(0, result)
                    program.capture(100 + index, 0, result)
                program.emit("LOP3", 0x212, 255, first=255, second=255,
                             third=255 | 0xFF << 8)
                program.predicate_output(0, True)
                reports.append(program.finish(f"logic-{group}-{pand}-{invert}", args))
    report = {"kind": "single-lane comparison/logic reference, not Vulkan execution", "programs": reports}
    (args.output_dir / "compare-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified 12 multiword comparison programs and 16 logic-predicate programs:")
    print("signed/unsigned 64-bit boundaries, dual outputs, input/output aliasing, all 256 LUTs, PAND/POR and RZ.")


if __name__ == "__main__":
    main()
