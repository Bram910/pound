#!/usr/bin/env python3
"""Check fused FP32 multiply-add against exact rational expected results."""

import argparse
import json
from pathlib import Path
import struct
import subprocess

from make_shader_spirv_arithmetic_demo import command, encode, finite, operand, rounded


def fused(first, second, third, mode, ftz, saturate):
    left, right, last = first & 0x7FFFFFFF, second & 0x7FFFFFFF, third & 0x7FFFFFFF
    sign = (first ^ second) & 0x80000000
    last_sign = third & 0x80000000
    if (max(left, right, last) > 0x7F800000
            or (left == 0x7F800000 and right == 0) or (right == 0x7F800000 and left == 0)):
        result = 0x7FFFFFFF
    elif left == 0x7F800000 or right == 0x7F800000:
        result = 0x7FFFFFFF if last == 0x7F800000 and sign != last_sign else sign | 0x7F800000
    elif last == 0x7F800000:
        result = third
    else:
        value = finite(first) * finite(second) + finite(third)
        result_sign = (0x80000000 if value < 0 else 0) if value else (
            sign if sign == last_sign else 0x80000000 if mode == 1 else 0)
        result = rounded(value, result_sign, mode, ftz)
    return (0 if result >> 31 or result > 0x7F800000 else min(result, 0x3F800000)) if saturate else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-fused-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions, checks = [], []

    def append(destination, mode=0, ftz=0, saturate=0, modifiers=0, form=1, immediate=0,
               first=0, middle=1, last=2, predicate=7, invert=0):
        second = middle | (modifiers >> 2 & 1) << 31 | (modifiers >> 3 & 1) << 30
        third = last | (modifiers & 3) << 8 | (modifiers >> 4 & 1) << 11 | (modifiers >> 5 & 1) << 10
        if form == 4:
            second = immediate
        elif form == 2:
            second = immediate
            third = middle | (modifiers & 3) << 8 | (modifiers >> 2 & 1) << 11 | (modifiers >> 3 & 1) << 10
        third |= mode << 14 | ftz << 16 | saturate << 13
        instructions.append(encode(0x23 | form << 9, destination, first, second, third, predicate, invert))
        checks.append((destination, mode, ftz, saturate, modifiers, form, immediate, first, middle, last, predicate, invert))

    for mode in range(4):
        for ftz in range(2):
            for saturate in range(2):
                append(32 + len(checks), mode, ftz, saturate)
    for modifiers in range(0, 64, 3):
        append(32 + len(checks), modifiers % 4, modifiers >> 2 & 1, modifiers >> 3 & 1, modifiers)
    for form in (2, 4):
        for immediate in (0, 0xBF800000, 0x7F800000, 0x7F800001):
            append(32 + len(checks), len(checks) % 4, len(checks) % 2, form=form, immediate=immediate)
    append(0, first=0, middle=1, last=2, predicate=0)
    append(2, mode=2, first=0, middle=1, last=2, predicate=0, invert=1)
    append(255, first=255, middle=0, last=2)
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "fused-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if trace[0]["operation"] != "FFMA" or len(trace[0]["sources"]) != 3 or trace[12]["float_rounding"] != 3:
        raise RuntimeError("Fused trace metadata differs")
    edge = (0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x00800001,
            0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFE, 0x7F7FFFFF,
            0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00000)
    triples = [(first, second, edge[(left + right * 3) % len(edge)])
               for left, first in enumerate(edge) for right, second in enumerate(edge)]
    triples += [(0x3F800001, 0x3F7FFFFE, 0xBF800000),
                (0x7F7FFFFF, 0x40000000, 0xFF7FFFFF),
                (1, 0x3F800001, 0x80000001),
                (0x00800000, 0x3F7FFFFF, 0),
                (0x00800001, 0x3F7FFFFE, 0),
                (0x00800001, 0x3FFFFFFE, 0x80800000)]
    triples += [((127 + exponent) << 23, 0x00800000, 0x80800000)
                for exponent in (-23, -24, -25, -26, -32, -64, -125)]
    inputs, expected = [], []
    for first, second, third in triples:
        for mask in (0, 127):
            record = [0] * 256
            record[0:3] = [first, second, third]
            record[255] = mask
            result = record.copy()
            for destination, mode, ftz, saturate, modifiers, form, immediate, first_register, middle_register, last_register, predicate, invert in checks:
                active = (True if predicate == 7 else bool(mask & (1 << predicate))) != bool(invert)
                if not active or destination == 255:
                    continue
                first_value = operand(result[first_register] if first_register != 255 else 0, modifiers & 3, ftz)
                second_value = operand(result[middle_register] if middle_register != 255 else 0, modifiers >> 2 & 3, ftz)
                third_value = operand(result[last_register] if last_register != 255 else 0, modifiers >> 4 & 3, ftz)
                if form == 2:
                    third_value = operand(immediate, 0, ftz)
                elif form == 4:
                    second_value = operand(immediate, 0, ftz)
                result[destination] = fused(first_value, second_value, third_value, mode, ftz, saturate)
            inputs.extend(record)
            expected.extend(result)
    lanes = len(inputs) // 256
    zero_result = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != zero_result or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions) or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from exact fused results")
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
        module = args.output_dir / f"fused-{mode}.spv"
        emission = [tool, "spirv" if mode == "straight" else "spirv-bounded", str(raw), str(module)]
        if mode == "bounded":
            emission.append(str(len(instructions)))
        command(emission)
        if args.spirv_val:
            command([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)])
        source_path, expected_path = args.output_dir / f"{mode}-input.bin", args.output_dir / f"{mode}-expected.bin"
        source_path.write_bytes(struct.pack(f"<{len(source)}I", *source))
        expected_path.write_bytes(struct.pack(f"<{len(wanted)}I", *wanted))
        dispatch_report = None
        if args.compute_tool:
            actual_path = args.output_dir / f"{mode}-actual.bin"
            dispatch = [str(args.compute_tool.resolve()), mode, str(raw), str(source_path), str(actual_path)]
            if mode == "bounded":
                dispatch.append(str(len(instructions)))
            dispatch.append(args.compute_device)
            dispatch_report = command(dispatch).stdout.strip()
            actual = struct.unpack(f"<{len(wanted)}I", actual_path.read_bytes())
            if actual != tuple(wanted):
                mismatch = next(index for index in range(len(wanted)) if actual[index] != wanted[index])
                raise RuntimeError(f"{mode} lane {mismatch // stride} word {mismatch % stride}: "
                                   f"{actual[mismatch]:08x} != {wanted[mismatch]:08x}")
        reports.append({"mode": mode, "lanes": lanes, "words_per_lane": stride,
                        "khronos_validation_checked": args.spirv_val is not None,
                        "vulkan_dispatch_checked": dispatch_report is not None,
                        "dispatch_report": dispatch_report})
    chain_report = None
    if args.compute_tool:
        chain = args.output_dir / "fused-loop.bin"
        chain.write_bytes(b"".join((encode(0x223, 0, first=1, second=2, third=1 << 16),
                                    encode(0x20B, first=0, second=3, third=1 << 12 | 1 << 20 | 7 << 23),
                                    encode(0x947, second=0xFFFFFFD0, third=0x0383FFFF, predicate=0),
                                    encode(0x94D, third=7 << 23))))
        source, wanted = [], []
        for mask in range(128):
            record = [0] * 261
            record[0:4] = [0, 0x3F000000, 0x3F000000, 0x3F800000]
            record[255] = mask
            source.extend(record)
            record[0] = 0x3F800000
            record[255:261] = [(mask & ~3) | 2, 64, 13, 0, 1, 0]
            wanted.extend(record)
        chain_input, chain_output = args.output_dir / "chain-input.bin", args.output_dir / "chain-actual.bin"
        chain_input.write_bytes(struct.pack(f"<{len(source)}I", *source))
        chain_report = command([str(args.compute_tool.resolve()), "bounded-chain", str(chain),
                                str(chain_input), str(chain_output), "2", "8", args.compute_device]).stdout.strip()
        if chain_output.read_bytes() != struct.pack(f"<{len(wanted)}I", *wanted) or "7 dispatches" not in chain_report:
            raise RuntimeError(f"Persistent fused loop differs: {chain_report}")
    rejected = (("DNZ", encode(0x223, 3, second=1, third=2 | 1 << 12)),
                ("constant-buffer", encode(0x623, 3, second=1, third=2)),
                ("uniform", encode(0xA23, 3, second=1, third=2)),
                ("extra-bits", encode(0x223, 3, second=1, third=2 | 1 << 17)))
    sentinel = b"fused rejection must preserve this output"
    output = args.output_dir / "rejected.spv"
    for name, instruction in rejected:
        bad_input = args.output_dir / f"rejected-{name}.bin"
        bad_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(bad_input), str(output)], capture_output=True, timeout=30)
            if result.returncode == 0 or output.read_bytes() != sentinel:
                raise RuntimeError(f"Fused rejection changed output: {name}, {mode}")
    report = {"kind": "single-rounding FP32 fused multiply-add with exact rational expected states",
              "instruction_count": len(instructions), "independent_checks": reports,
              "rejections": [name for name, _ in rejected],
              "persistent_chain_lanes": 128 if chain_report is not None else 0,
              "persistent_chain_report": chain_report}
    (args.output_dir / "fused-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} exact rational fused states per path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. Vulkan dispatch checked: {args.compute_tool is not None}.")


if __name__ == "__main__":
    main()
