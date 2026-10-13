#!/usr/bin/env python3
"""Check FP32 compare/min-max fixtures; optionally dispatch both SPIR-V paths."""

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess


def encode(opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
    return struct.pack("<4I", opcode | predicate << 12 | invert << 15
                       | destination << 16 | first << 24, second, third, 0xFC000)


def operand(bits, modifiers=0, ftz=0):
    if modifiers & 2:
        bits &= 0x7FFFFFFF
    if modifiers & 1:
        bits ^= 0x80000000
    if ftz and bits & 0x7F800000 == 0:
        bits &= 0x80000000
    return bits


def number(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def compare(first, second, relation):
    left, right = number(first), number(second)
    unordered = math.isnan(left) or math.isnan(right)
    result = (False, left < right, left == right, left <= right,
              left > right, left != right, left >= right, True)[relation & 7]
    return result or unordered if relation & 8 else result and not unordered


def minmax(first, second, minimum):
    left, right = number(first), number(second)
    if math.isnan(left):
        return 0x7FFFFFFF if math.isnan(right) else second
    if math.isnan(right):
        return first
    if left == right == 0.0:
        return (first | second) if minimum else (first & second)
    return first if (left < right) == bool(minimum) else second


def command(arguments):
    return subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-float-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions, checks = [], []
    for relation in range(16):
        for logic in range(3):
            ftz = (relation + logic) & 1
            instructions.append(encode(0x20B, second=1,
                                       third=logic << 10 | relation << 12 | ftz << 16
                                       | 1 << 20 | 6 << 23))
            destination = 32 + (relation * 3 + logic) * 2
            instructions.append(encode(0x807, destination, first=255, second=1,
                                       third=1 << 26))
            instructions.append(encode(0x807, destination + 1, first=255, second=1,
                                       third=1 << 23 | 1 << 26))
            checks.append((destination, relation, logic, ftz))
    minimum_checks = []
    for modifiers in range(16):
        for minimum in range(2):
            ftz = modifiers & 1
            destination = 128 + modifiers * 2 + minimum
            instructions.append(encode(0x209, destination, second=1 | (modifiers >> 2 & 1) << 31
                                       | (modifiers >> 3 & 1) << 30,
                                       third=(modifiers & 3) << 8 | ftz << 16
                                       | 7 << 23 | (minimum ^ 1) << 26))
            minimum_checks.append((destination, modifiers, minimum, ftz))
    instructions.extend((
        encode(0x809, 160, second=0x7F800001, third=7 << 23),
        encode(0x80B, second=0x7F800000, third=4 << 12 | 2 << 17 | 3 << 20 | 7 << 23),
        encode(0x209, 0, second=1, third=7 << 23),
        encode(0x209, 1, first=1, second=0, third=7 << 23 | 1 << 26),
        encode(0x209, 255, second=255, third=7 << 23),
        encode(0x94D, third=7 << 23),
    ))
    raw = args.output_dir / "float-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if (trace[0]["operation"] != "FSETP" or trace[0]["float_comparison"] != 0
            or trace[0]["sources"][0]["absolute"] != 0
            or trace[144]["operation"] != "FMNMX" or trace[144]["float_ftz"] != 0
            or trace[146]["float_ftz"] != 1 or trace[146]["sources"][0]["negate"] != 1
            or trace[148]["sources"][0]["absolute"] != 1
            or trace[160]["sources"][1]["absolute"] != 1):
        raise RuntimeError("Floating-point trace metadata differs")
    edge = (0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
            0x80800000, 0x3F800000, 0xBF800000, 0x7F7FFFFF, 0xFF7FFFFF,
            0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC00000, 0x7F800001, 0xFF800001)
    inputs, expected = [], []
    for first in edge:
        for second in edge:
            for condition in range(2):
                record = [0] * 256
                record[0:2] = [first, second]
                record[255] = condition << 6
                result = record.copy()
                for destination, relation, logic, ftz in checks:
                    comparison = compare(operand(first, ftz=ftz), operand(second, ftz=ftz), relation)
                    operation = (lambda left, right: left and right,
                                 lambda left, right: left or right,
                                 lambda left, right: bool(left) != bool(right))[logic]
                    left = int(operation(comparison, condition))
                    right = int(operation(not comparison, condition))
                    result[destination:destination + 2] = [left, right]
                    result[255] = condition << 6 | left | right << 1
                for destination, modifiers, minimum, ftz in minimum_checks:
                    result[destination] = minmax(operand(first, modifiers & 3, ftz),
                                                 operand(second, modifiers >> 2, ftz), minimum)
                result[160] = first if not math.isnan(number(first)) else 0x7FFFFFFF
                above_infinity = compare(first, 0x7F800000, 4)
                result[255] |= int(above_infinity) << 2 | int(not above_infinity) << 3
                result[0] = minmax(first, second, 1)
                result[1] = minmax(second, result[0], 0)
                inputs.extend(record)
                expected.extend(result)
    lanes = len(inputs) // 256
    initial_registers = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != initial_registers or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions)
                or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from the independent zero-input result")
    reports = []
    for mode in ("straight", "bounded"):
        stride = 256 if mode == "straight" else 261
        source, wanted = [], []
        for index in range(lanes):
            source.extend(inputs[index * 256:(index + 1) * 256])
            wanted.extend(expected[index * 256:(index + 1) * 256])
            if mode == "bounded":
                source.extend((0, 0, 0, 0, 0xDEADBEEF))
                wanted.extend((len(instructions) * 16, len(instructions), 0, 1, 0))
        module = args.output_dir / f"float-{mode}.spv"
        emission = [tool, "spirv" if mode == "straight" else "spirv-bounded", str(raw), str(module)]
        if mode == "bounded":
            emission.append(str(len(instructions)))
        command(emission)
        if args.spirv_val:
            command([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)])
        input_path = args.output_dir / f"{mode}-input.bin"
        expected_path = args.output_dir / f"{mode}-expected.bin"
        input_path.write_bytes(struct.pack(f"<{len(source)}I", *source))
        expected_path.write_bytes(struct.pack(f"<{len(wanted)}I", *wanted))
        dispatch_report = None
        if args.compute_tool:
            actual_path = args.output_dir / f"{mode}-actual.bin"
            dispatch = [str(args.compute_tool.resolve()), mode, str(raw), str(input_path), str(actual_path)]
            if mode == "bounded":
                dispatch.append(str(len(instructions)))
            dispatch.append(args.compute_device)
            dispatch_report = command(dispatch).stdout.strip()
            actual = struct.unpack(f"<{len(wanted)}I", actual_path.read_bytes())
            if actual != tuple(wanted):
                mismatch = next(index for index in range(len(wanted)) if actual[index] != wanted[index])
                raise RuntimeError(f"{mode} mismatch at lane {mismatch // stride}, word {mismatch % stride}: "
                                   f"{actual[mismatch]:08x} != {wanted[mismatch]:08x}")
        reports.append({"mode": mode, "lanes": lanes, "words_per_lane": stride,
                        "khronos_validation_checked": args.spirv_val is not None,
                        "vulkan_dispatch_checked": dispatch_report is not None,
                        "dispatch_report": dispatch_report})
    sentinel = b"unsupported float emission must preserve this output"
    rejected_output = args.output_dir / "rejected.spv"
    rejected = (
        ("constant-buffer", encode(0x60B, third=1 << 20 | 7 << 23)),
        ("invalid-boolean", encode(0x20B, second=1, third=3 << 10 | 1 << 20 | 7 << 23)),
        ("predicate-alias", encode(0x20B, second=1, third=7 << 23)),
        ("extra-source-bits", encode(0x209, 2, second=0x100, third=7 << 23)),
        ("unsupported-minmax-modifier", encode(0x209, 2, second=1, third=2 | 7 << 23)),
        ("unsupported-FFMA-DNZ", encode(0x223, 2, second=1, third=2 | 1 << 12)),
    )
    for name, instruction in rejected:
        rejected_input = args.output_dir / f"rejected-{name}.bin"
        rejected_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            rejected_output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(rejected_input), str(rejected_output)],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode == 0 or rejected_output.read_bytes() != sentinel:
                raise RuntimeError(f"Unsupported instruction changed output: {name}, {mode}")
    report = {"kind": "FP32 comparison/min-max subset, not general FP arithmetic or rendering",
              "instruction_count": len(instructions), "independent_checks": reports,
              "rejections": [name for name, _ in rejected]}
    (args.output_dir / "float-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} independent FP32 edge states for each SPIR-V path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. "
          f"Vulkan dispatch checked: {args.compute_tool is not None}.")
    if args.compute_tool is None:
        print("No Vulkan dispatch performed. Use --compute-tool to run the emitted fixtures.")


if __name__ == "__main__":
    main()
