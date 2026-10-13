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
    parser.add_argument("--compute-tool", type=Path, help="optionally dispatch with PoundGpuCompute")
    parser.add_argument("--compute-device", default="auto", help="auto or Vulkan physical-device index")
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
    instructions[-2:-2] = [
        encode(0x802, 20, second=0x807F0180, third=0xF00),
        encode(0x802, 21, second=0xFEDCBA98, third=0xF00),
        encode(0x802, 22, second=0xFFFFFFFF, third=0xF00),
        encode(0x802, 23, second=32, third=0xF00),
        encode(0x819, 30, first=20, second=32, third=21 | 1 << 9 | 1 << 12),
        encode(0x819, 31, first=20, second=64, third=21 | 1 << 12 | 1 << 16),
        encode(0x816, 32, first=20, second=0xBA98, third=21),
        encode(0x81A, 33, first=20, second=8, third=1 << 9),
        encode(0x81B, 34, first=23, second=0xFFFFFFFF),
        encode(0x900, 35, second=0x807F0180, third=0xE0400),
        encode(0x802, 40, second=0xFFFFFFFF, third=0xF00),
        encode(0x802, 41, second=1, third=0xF00),
        encode(0x225, 40, first=1, second=1, third=40 | 0x078E0000),
        encode(0x227, 42, first=2, second=2, third=40 | 1 << 9 | 0x078E0000),
        encode(0x226, 43, first=20, second=21, third=255 | 1 << 9 | 1 << 10),
        encode(0x226, 44, first=20, second=21, third=255 | 1 << 10 | 1 << 12 | 1 << 13),
        encode(0x211, 45, first=1, second=255, third=255 | 7 << 17),
        encode(0x210, 46, first=255, second=255, third=255 | 0x0781E000 | 1 << 8 | 6 << 17 | 5 << 20),
        encode(0x210, 47, first=1, second=1, third=255 | 1 << 8 | 1 << 10 | 7 << 13 | 6 << 17 | 5 << 20 | 6 << 23),
        encode(0x211, 48, first=20, second=1, third=21 | 1 << 8 | 1 << 10 | 31 << 11 | 1 << 16 | 6 << 17 | 6 << 23),
        encode(0x300, 49, second=22, third=0xE0600),
    ]
    raw = args.output_dir / "integer-compute.bin"
    module = args.output_dir / "integer-compute.spv"
    raw.write_bytes(b"".join(instructions))
    trace = subprocess.run([tool, "trace", str(raw)], capture_output=True, text=True, check=True, timeout=30)
    families = sorted({json.loads(line)["operation"] for line in trace.stdout.splitlines()})
    wanted_families = sorted(("MOV", "IADD3", "IMAD", "LOP3", "POPC", "SEL", "ISETP", "PLOP3",
                             "IMNMX", "BREV", "IABS", "SHF", "FLO", "PRMT", "SGXT", "BMSK",
                             "LEA", "IDP4", "IDP2", "NOP", "EXIT"))
    if families != wanted_families:
        raise RuntimeError(f"Instruction families differ: {families}")
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
    expected.update({"R20": 0x807F0180, "R21": 0xFEDCBA98, "R22": 0xFFFFFFFF, "R23": 32,
                     "R30": 0xFEDCBA98, "R31": 0xFFFFFFFF, "R32": 0xFF0000FF,
                     "R33": 0xFFFFFF80, "R42": 0x40000000,
                     "R43": -128 * -104 + 1 * -70 + 127 * -36 + -128 * -2,
                     "R44": (384 * -36 + 32895 * -2) & 0xFFFFFFFF,
                     "R45": 0xFFFFFFFF, "R47": 1, "R48": 0xBFC07F3F, "R49": 0xFFFFFFFF})
    if (result["registers"] != expected or result["predicates"] != 0x4B
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
    dispatch_report = None
    if args.compute_tool:
        actual = args.output_dir / "lane-actual.bin"
        dispatch = subprocess.run(
            [str(args.compute_tool.resolve()), "straight", str(raw),
             str(args.output_dir / "lane-input.bin"), str(actual), args.compute_device],
            capture_output=True, text=True, check=True, timeout=60)
        if actual.read_bytes() != (args.output_dir / "lane-expected.bin").read_bytes():
            raise RuntimeError("Vulkan lane result differs from independent expected state")
        dispatch_report = dispatch.stdout.strip()
    report = {"kind": "experimental straight-line integer compute SPIR-V, not game rendering",
              "spirv_version": "1.3", "word_count": len(data) // 4,
              "khronos_validation_checked": validated, "descriptor_set": 0, "binding": 0,
              "lane_words": 256, "workgroup_size": [1, 1, 1], "reference_result": result,
              "instruction_families": families,
              "vulkan_dispatch_checked": dispatch_report is not None,
              "vulkan_dispatch_report": dispatch_report,
              "rejection_checks": [name for name, _ in rejected]}
    (args.output_dir / "spirv-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Emitted a 21-family integer compute SPIR-V module and independent lane input/expected fixtures.")
    print(f"Khronos validation checked: {validated}. Vulkan dispatch checked: {dispatch_report is not None}.")
    if dispatch_report:
        print(dispatch_report)
    else:
        print("No Vulkan dispatch performed. Use --compute-tool to check emitted fixtures on a device.")


if __name__ == "__main__":
    main()
