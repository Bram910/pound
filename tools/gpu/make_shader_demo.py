#!/usr/bin/env python3
"""Check a synthetic SM86 integer program with the shader reference executor."""

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
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instructions = []
    operations = []

    def emit(operation, opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
        instructions.append((opcode | (predicate << 12) | (invert << 15)
                             | (destination << 16) | (first << 24), second, third, 0xFC000))
        operations.append(operation)

    emit("MOV", 0x802, 1, second=0xFFFFFFF0, third=0xF00)
    emit("MOV", 0x802, 2, second=37, third=0xF00)
    emit("IADD3", 0x210, 3, first=1, second=2, third=0x07FFE0FF)
    emit("IMAD", 0x824, 4, first=3, second=17, third=0xE0202)
    emit("LOP3", 0x212, 5, first=1, second=3, third=0x078E9604)
    emit("POPC", 0x309, 6, second=5)
    emit("SEL", 0x807, 7, first=1, second=0xC001CAFE)
    emit("MOV", 0x802, 8, second=0xAAAAAAAA, third=0xF00, predicate=0, invert=1)
    emit("MOV", 0x802, 9, second=0xDEADBEEF, third=0xF00, predicate=0)
    emit("NOP", 0x918)
    emit("BRA", 0x947, second=0, third=0)
    emit("BRA", 0x947, second=4 << 2, third=0x03800000)
    emit("MOV", 0x802, 10, second=0xDEADBEEF, third=0xF00)
    emit("EXIT", 0x94D, third=0x03800000)
    path = args.output_dir / "integer-program.bin"
    path.write_bytes(b"".join(struct.pack("<IIII", *instruction) for instruction in instructions))
    tool = str(args.shader_tool.resolve())
    trace = subprocess.run([tool, "trace", str(path)], capture_output=True, text=True, check=True, timeout=30)
    decoded = [json.loads(line) for line in trace.stdout.splitlines()]
    if [instruction["operation"] for instruction in decoded] != operations:
        raise RuntimeError("Decoded instruction sequence differs")
    result = subprocess.run([tool, "run", str(path), "100"], capture_output=True, text=True, check=True, timeout=30)
    lane = json.loads(result.stdout)
    third = (0xFFFFFFF0 + 37) & 0xFFFFFFFF
    fourth = (third * 17 + 37) & 0xFFFFFFFF
    fifth = 0xFFFFFFF0 ^ third ^ fourth
    expected = {"R1": 0xFFFFFFF0, "R2": 37, "R3": third, "R4": fourth,
                "R5": fifth, "R6": fifth.bit_count(), "R7": 0xC001CAFE, "R8": 0xAAAAAAAA}
    if lane != {"pc": len(instructions) * 16, "steps": 13, "halted": 1, "predicates": 0, "registers": expected}:
        raise RuntimeError(f"Reference result differs: {lane}")
    nvidia_checked = False
    if args.nvdisasm:
        official = subprocess.run([str(args.nvdisasm.resolve()), "--binary", "SM86", "--no-dataflow",
                                   "--print-raw", str(path)], capture_output=True, text=True, check=True, timeout=30)
        listings = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P[0-7T]+\s+)?([A-Z0-9]+)[^;]*;", official.stdout)
        if listings != operations or "BRA 0xd0;" not in official.stdout:
            raise RuntimeError(f"NVIDIA SM86 disassembly differs:\n{official.stdout}")
        (args.output_dir / "nvidia-sm86.txt").write_text(official.stdout, encoding="utf-8")
        nvidia_checked = True
    report = {"kind": "single-lane integer reference, not SPIR-V or GPU execution",
              "nvidia_sm86_disassembly_checked": nvidia_checked, "instructions": decoded, "result": lane}
    (args.output_dir / "shader-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(decoded)} instructions across nine families, 13 execution steps and all output registers.")
    print(f"NVIDIA SM86 disassembly checked: {nvidia_checked}. No Vulkan dispatch or rendered frame is implied.")


if __name__ == "__main__":
    main()
