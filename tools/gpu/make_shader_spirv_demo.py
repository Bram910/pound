#!/usr/bin/env python3
"""Emit a straight-line SM86 integer compute module with a reference lane result."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


def encode(opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
    return struct.pack("<IIII", opcode | predicate << 12 | invert << 15
                       | destination << 16 | first << 24, second, third, 0xFC000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-spirv-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions = [
        encode(0x802, 1, second=0xFFFFFFFF, third=0xF00),
        encode(0x802, 2, second=0x80000000, third=0xF00),
        encode(0x210, 3, first=1, second=1, third=1 | 0x0781E000 | 1 << 20),
        encode(0x824, 4, first=1, second=3, third=2 | 0xE0000),
        encode(0x812, 5, first=1, second=0xAAAAAAAA, third=2 | 0x96 << 8 | 7 << 23),
        encode(0x309, 6, second=5 | 1 << 31),
        encode(0x807, 7, first=1, second=7),
        encode(0x20C, first=1, second=2, third=0x70 | 0x200 | 1 << 12 | 2 << 17 | 3 << 20 | 7 << 23),
        encode(0x20C, first=2, second=2, third=2 << 4 | 0x100 | 5 << 12 | 2 << 17 | 3 << 20 | 7 << 23),
        encode(0x81C, 0x96, third=2 << 23 | 3 << 13 | 7 << 4 | 4 << 17 | 5 << 20 | (0xF0 & 7) | (0xF0 >> 3) << 8),
        encode(0x817, 8, first=2, second=1, third=0x200 | 4 << 23),
        encode(0x301, 9, second=5),
        encode(0x213, 10, second=2),
        encode(0x802, 11, second=0xCAFEBABE, third=0xF00, predicate=7, invert=1),
        encode(0x918),
        encode(0x94D, third=7 << 23),
    ]
    raw = args.output_dir / "integer-compute.bin"
    module = args.output_dir / "integer-compute.spv"
    raw.write_bytes(b"".join(instructions))
    subprocess.run([tool, "spirv", str(raw), str(module)], capture_output=True, text=True, check=True, timeout=30)
    data = module.read_bytes()
    header = struct.unpack_from("<5I", data)
    if header[0] != 0x07230203 or header[1] != 0x00010300 or len(data) % 4:
        raise RuntimeError("Invalid SPIR-V header")
    validated = False
    if args.spirv_val:
        subprocess.run([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)],
                       capture_output=True, text=True, check=True, timeout=30)
        validated = True
    run = subprocess.run([tool, "run-prepared", str(raw)], capture_output=True, text=True, check=True, timeout=30)
    result = json.loads(run.stdout)
    expected = {"R1": 0xFFFFFFFF, "R2": 0x80000000, "R3": 0xFFFFFFFD,
                "R4": 0x7FFFFFFD, "R5": 0xD5555555, "R6": 15, "R7": 0xFFFFFFFF,
                "R8": 1, "R9": 0xAAAAAAAB, "R10": 0x80000000}
    if (result["registers"] != expected or result["predicates"] != 0x0B
            or result["pc"] != len(instructions) * 16
            or result["steps"] != len(instructions) or result["halted"] != 1):
        raise RuntimeError(f"Reference result differs: {result}")
    rejected = (
        ("missing-exit", instructions[:-1]),
        ("predicated-exit", instructions[:-1] + [encode(0x94D, third=7 << 23, predicate=0)]),
        ("early-exit", [instructions[-1], instructions[-1]]),
        ("branch", [encode(0x947, third=7 << 23), instructions[-1]]),
        ("instruction-limit", [encode(0x918)] * 1024 + [instructions[-1]]),
    )
    rejected_output = args.output_dir / "rejected-output.spv"
    sentinel = b"preserve existing output on unsupported emission"
    for name, code in rejected:
        rejected_input = args.output_dir / f"rejected-{name}.bin"
        rejected_input.write_bytes(b"".join(code))
        rejected_output.write_bytes(sentinel)
        failure = subprocess.run([tool, "spirv", str(rejected_input), str(rejected_output)],
                                 capture_output=True, text=True, timeout=30)
        if failure.returncode == 0 or rejected_output.read_bytes() != sentinel:
            raise RuntimeError(f"Rejected program changed output: {name}")
    records = [0] * 256
    (args.output_dir / "lane-input.bin").write_bytes(struct.pack("<256I", *records))
    for name, value in expected.items():
        records[int(name[1:])] = value
    records[255] = result["predicates"]
    (args.output_dir / "lane-expected.bin").write_bytes(struct.pack("<256I", *records))
    report = {"kind": "experimental straight-line integer compute SPIR-V, not game rendering",
              "spirv_version": "1.3", "word_count": len(data) // 4,
              "khronos_validation_checked": validated, "descriptor_set": 0, "binding": 0,
              "lane_words": 256, "workgroup_size": [1, 1, 1], "reference_result": result,
              "rejection_checks": [name for name, _ in rejected]}
    (args.output_dir / "spirv-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Emitted a 13-family integer compute SPIR-V module and independent lane input/expected fixtures.")
    print(f"Khronos validation checked: {validated}. Actual dispatch needs a Vulkan compute harness.")


if __name__ == "__main__":
    main()
