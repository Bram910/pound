#!/usr/bin/env python3
"""Verify SM86 predicate writes, bounded counted loops and integer edge cases."""

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
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-control-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    reports = []
    for limit in (0, 1, 2, 37, 1000):
        instructions = []
        operations = []

        def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7):
            instructions.append((opcode | predicate << 12 | destination << 16 | first << 24,
                                 second, third, 0xFC000))
            operations.append(operation)

        emit("MOV", 0x802, 1, second=0, third=0xF00)
        emit("MOV", 0x802, 2, second=0, third=0xF00)
        emit("IADD3", 0x810, 1, first=1, second=1, third=0x07FFE0FF)
        emit("IADD3", 0x210, 2, first=2, second=1, third=0x07FFE0FF)
        emit("ISETP", 0x80C, first=1, second=limit, third=0x039010F0)
        relative = ((32 - (len(instructions) + 1) * 16) // 4) & ((1 << 48) - 1)
        emit("BRA", 0x947, second=(relative << 2) & 0xFFFFFFFF,
             third=(relative >> 30) | 0x03800000, predicate=0)
        emit("PLOP3", 0x81C, destination=0x3C,
             third=(0xFC & 7) | ((0xFC >> 3) << 8) | (2 << 17) | (3 << 20)
             | (7 << 4) | (1 << 13))
        emit("IMNMX", 0x817, 3, first=2, second=0x80000000, third=0x03800200)
        emit("IMNMX", 0x817, 4, first=2, second=0x80000000, third=0x07800200)
        emit("BREV", 0x301, 5, second=2)
        emit("IABS", 0x813, 6, second=0x80000000)
        emit("IABS", 0x813, 7, second=0xFFFFFFEF)
        emit("EXIT", 0x94D, third=0x03800000)
        path = args.output_dir / f"counted-loop-{limit}.bin"
        path.write_bytes(b"".join(struct.pack("<IIII", *words) for words in instructions))
        trace = subprocess.run([tool, "trace", str(path)], capture_output=True,
                               text=True, check=True, timeout=30)
        decoded = [json.loads(line) for line in trace.stdout.splitlines()]
        if [instruction["operation"] for instruction in decoded] != operations:
            raise RuntimeError("Instruction sequence differs")
        result = subprocess.run([tool, "run", str(path), "10000"], capture_output=True,
                                text=True, check=True, timeout=30)
        iterations = max(1, limit)
        total = iterations * (iterations + 1) // 2
        reversed_bits = int(f"{total:032b}"[::-1], 2)
        expected = {"pc": len(instructions) * 16, "steps": iterations * 4 + 9,
                    "halted": 1, "predicates": 14,
                    "registers": {"R1": iterations, "R2": total, "R3": 0x80000000,
                                  "R4": total, "R5": reversed_bits, "R6": 0x80000000, "R7": 17}}
        lane = json.loads(result.stdout)
        if lane != expected:
            raise RuntimeError(f"Counted-loop result differs: {lane}, expected {expected}")
        bounded = None
        if limit == 37:
            partial = subprocess.run([tool, "run", str(path), "14"], capture_output=True,
                                     text=True, timeout=30)
            bounded = json.loads(partial.stdout)
            if partial.returncode == 0 or bounded != {
                "pc": 32, "steps": 14, "halted": 0, "predicates": 1,
                "registers": {"R1": 3, "R2": 6}
            }:
                raise RuntimeError(f"Budget boundary differs: {partial}")
        official_text = None
        if args.nvdisasm:
            official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86",
                                       "--no-dataflow", "--print-raw", str(path)],
                                      capture_output=True, text=True, check=True, timeout=30)
            listings = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
            if listings != operations or "@P0 BRA 0x20;" not in official.stdout:
                raise RuntimeError(f"NVIDIA SM86 disassembly differs:\n{official.stdout}")
            official_text = official.stdout
            (args.output_dir / f"nvidia-sm86-{limit}.txt").write_text(official_text, encoding="utf-8")
        reports.append({"limit": limit, "instructions": decoded, "result": lane,
                        "budget_exhaustion_result": bounded,
                        "nvidia_sm86_disassembly_checked": official_text is not None})
    report = {"kind": "single-lane integer reference, not SIMT or Vulkan execution", "cases": reports}
    (args.output_dir / "control-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Verified five counted loops, predicate/complement writes, dual truth tables,")
    print("signed min/max, bit reversal, INT_MIN absolute value and a bounded partial execution.")


if __name__ == "__main__":
    main()
