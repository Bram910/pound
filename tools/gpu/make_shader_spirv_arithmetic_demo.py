#!/usr/bin/env python3
"""Check FP32 addition/multiplication using exact rational expected results."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess


def encode(opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
    return struct.pack("<4I", opcode | predicate << 12 | invert << 15
                       | destination << 16 | first << 24, second, third, 0xFC000)


def operand(bits, modifiers, ftz):
    if modifiers & 2:
        bits &= 0x7FFFFFFF
    if modifiers & 1:
        bits ^= 0x80000000
    if ftz and bits & 0x7F800000 == 0:
        bits &= 0x80000000
    return bits


def power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def finite(bits):
    exponent = bits >> 23 & 255
    mantissa = bits & 0x7FFFFF | (0x800000 if exponent else 0)
    return mantissa * power((exponent or 1) - 150) * (-1 if bits >> 31 else 1)


def rounded(value, sign, mode, ftz):
    magnitude = abs(value)
    if not magnitude:
        return sign
    if ftz and magnitude < power(-126):
        outward = (mode == 1 and sign != 0) or (mode == 2 and sign == 0)
        cutoff = power(-126) - power(-151 if mode == 0 else -150)
        normal = magnitude >= cutoff if mode == 0 else magnitude > cutoff if outward else False
        if not normal:
            return sign
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < power(exponent):
        exponent -= 1
    scaled = magnitude / power(max(exponent, -126) - 23)
    mantissa, remainder = divmod(scaled.numerator, scaled.denominator)
    increment = ((remainder * 2 > scaled.denominator
                  or (remainder * 2 == scaled.denominator and mantissa & 1)) if mode == 0
                 else bool(remainder) and ((mode == 1 and sign != 0) or (mode == 2 and sign == 0)))
    mantissa += int(increment)
    if mantissa >= 0x1000000:
        mantissa >>= 1
        exponent += 1
    if exponent >= 128:
        infinity = mode == 0 or (mode == 1 and sign != 0) or (mode == 2 and sign == 0)
        return sign | (0x7F800000 if infinity else 0x7F7FFFFF)
    exponent_bits = max(exponent, -126) + 127 if mantissa >= 0x800000 else 0
    return sign if ftz and exponent_bits == 0 else sign | exponent_bits << 23 | mantissa & 0x7FFFFF


def arithmetic(first, second, multiply, mode, ftz, saturate):
    left, right = first & 0x7FFFFFFF, second & 0x7FFFFFFF
    sign = (first ^ second) & 0x80000000
    if left > 0x7F800000 or right > 0x7F800000:
        result = 0x7FFFFFFF
    elif multiply and ((left == 0 and right == 0x7F800000) or (right == 0 and left == 0x7F800000)):
        result = 0x7FFFFFFF
    elif left == 0x7F800000 or right == 0x7F800000:
        if multiply:
            result = sign | 0x7F800000
        elif left == right and sign:
            result = 0x7FFFFFFF
        else:
            result = first if left == 0x7F800000 else second
    else:
        value = finite(first) * finite(second) if multiply else finite(first) + finite(second)
        if not multiply:
            sign = (0x80000000 if value < 0 else 0) if value else (
                first & 0x80000000 if (first ^ second) & 0x80000000 == 0 else 0x80000000 if mode == 1 else 0)
        result = rounded(value, sign, mode, ftz)
    if saturate:
        result = 0 if result >> 31 or result > 0x7F800000 else min(result, 0x3F800000)
    return result


def command(arguments):
    return subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-arithmetic-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions, checks = [], []

    def append(destination, multiply, mode=0, ftz=0, saturate=0, modifiers=0,
               immediate=None, first=0, second=1, predicate=7, invert=0):
        form = 1 if immediate is None else 4 if multiply else 2
        middle = second | (modifiers >> 2 & 1) << 31 | (modifiers >> 3 & 1) << 30
        instructions.append(encode((0x20 if multiply else 0x21) | form << 9,
                                   destination, first, middle if immediate is None else immediate,
                                   (modifiers & 3) << 8 | mode << 14 | ftz << 16 | saturate << 13
                                   | (0x400000 if multiply else 0), predicate, invert))
        checks.append((destination, multiply, mode, ftz, saturate, modifiers,
                       immediate, first, second, predicate, invert))

    for multiply in range(2):
        for mode in range(4):
            for ftz in range(2):
                for saturate in range(2):
                    append(32 + len(checks), multiply, mode, ftz, saturate)
    for modifiers in range(16):
        append(32 + len(checks), modifiers & 1, modifiers % 4, modifiers >> 1 & 1,
               modifiers >> 2 & 1, modifiers)
    for immediate in (1, 0x3F800001, 0xBF800000, 0x7F7FFFFF, 0x7F800000, 0x7F800001):
        for multiply in range(2):
            append(32 + len(checks), multiply, len(checks) % 4, immediate & 1, immediate >> 1 & 1,
                   immediate=immediate)
    append(0, 0, first=0, second=1, predicate=0)
    append(1, 1, mode=2, first=1, second=0, predicate=0, invert=1)
    append(255, 1, first=255, second=0)
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "arithmetic-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if (trace[0]["operation"] != "FADD" or trace[16]["operation"] != "FMUL"
            or trace[12]["float_rounding"] != 3 or trace[1]["float_saturate"] != 1):
        raise RuntimeError("Arithmetic trace metadata differs")
    edge = (0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
            0x80800000, 0x00800001, 0x80800001, 0x3F800000, 0xBF800000,
            0x3F7FFFFF, 0x3F7FFFFE, 0x3F800001, 0x3FFFFFFF, 0x40000000, 0x33800000, 0x32800000,
            0xB3800000, 0x33000000, 0x3F000000, 0x7F7FFFFF, 0xFF7FFFFF,
            0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC00000, 0x7F800001, 0xFF800001)
    inputs, expected = [], []
    for first in edge:
        for second in edge:
            for mask in (0, 127):
                record = [0] * 256
                record[0:2] = [first, second]
                record[255] = mask
                result = record.copy()
                for destination, multiply, mode, ftz, saturate, modifiers, immediate, left_register, right_register, predicate, invert in checks:
                    active = (True if predicate == 7 else bool(mask & (1 << predicate))) != bool(invert)
                    if active and destination != 255:
                        left = result[left_register] if left_register != 255 else 0
                        right = result[right_register] if right_register != 255 else 0
                        left = operand(left, modifiers & 3, ftz)
                        right = operand(right, modifiers >> 2, ftz) if immediate is None else operand(immediate, 0, ftz)
                        result[destination] = arithmetic(left, right, multiply, mode, ftz, saturate)
                inputs.extend(record)
                expected.extend(result)
    lanes = len(inputs) // 256
    initial_registers = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != initial_registers or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions)
                or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from exact rational results")
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
        module = args.output_dir / f"arithmetic-{mode}.spv"
        emission = [tool, "spirv" if mode == "straight" else "spirv-bounded", str(raw), str(module)]
        if mode == "bounded":
            emission.append(str(len(instructions)))
        command(emission)
        if args.spirv_val:
            command([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)])
        input_path, expected_path = args.output_dir / f"{mode}-input.bin", args.output_dir / f"{mode}-expected.bin"
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
    chain_report = None
    if args.compute_tool:
        chain = args.output_dir / "arithmetic-loop.bin"
        chain.write_bytes(b"".join((encode(0x221, 0, first=0, second=1),
                                    encode(0x220, 3, first=0, second=1, third=0x400000),
                                    encode(0x20B, first=0, second=2, third=1 << 12 | 1 << 20 | 7 << 23),
                                    encode(0x947, second=0xFFFFFFC0, third=0x0383FFFF, predicate=0),
                                    encode(0x94D, third=7 << 23))))
        source, wanted = [], []
        for mask in range(128):
            record = [0] * 261
            record[0:4] = [0, 0x3E800000, 0x3F800000, 0]
            record[255] = mask
            source.extend(record)
            record[0:4] = [0x3F800000, 0x3E800000, 0x3F800000, 0x3E800000]
            record[255:261] = [(mask & ~3) | 2, 80, 17, 0, 1, 0]
            wanted.extend(record)
        chain_input = args.output_dir / "chain-input.bin"
        chain_output = args.output_dir / "chain-actual.bin"
        chain_input.write_bytes(struct.pack(f"<{len(source)}I", *source))
        chain_report = command([str(args.compute_tool.resolve()), "bounded-chain", str(chain),
                                str(chain_input), str(chain_output), "3", "8", args.compute_device]).stdout.strip()
        if chain_output.read_bytes() != struct.pack(f"<{len(wanted)}I", *wanted):
            raise RuntimeError("Persistent arithmetic loop differs from independent expected states")
        if "6 dispatches" not in chain_report:
            raise RuntimeError(f"Arithmetic chain dispatch count differs: {chain_report}")
    rejected = (
        ("FADD-wrong-form", encode(0x821, 9, second=1)),
        ("FMUL-wrong-form", encode(0x420, 9, second=1, third=0x400000)),
        ("constant-buffer", encode(0x620, 9, second=1, third=0x400000)),
        ("FMUL-scale", encode(0x220, 9, second=1, third=0x300000)),
        ("FMUL-DNZ", encode(0x220, 9, second=1, third=0x401000)),
        ("FADD-extra-bits", encode(0x221, 9, second=1, third=1 << 17)),
    )
    sentinel = b"unsupported arithmetic must preserve this output"
    rejected_output = args.output_dir / "rejected.spv"
    for name, instruction in rejected:
        rejected_input = args.output_dir / f"rejected-{name}.bin"
        rejected_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            rejected_output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(rejected_input), str(rejected_output)], capture_output=True, timeout=30)
            if result.returncode == 0 or rejected_output.read_bytes() != sentinel:
                raise RuntimeError(f"Unsupported arithmetic changed output: {name}, {mode}")
    report = {"kind": "FP32 FADD/FMUL with exact rational expected states",
              "instruction_count": len(instructions), "independent_checks": reports,
              "rejections": [name for name, _ in rejected],
              "persistent_chain_lanes": 128 if chain_report is not None else 0,
              "persistent_chain_report": chain_report}
    (args.output_dir / "arithmetic-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} exact rational edge states for each SPIR-V path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. Vulkan dispatch checked: {args.compute_tool is not None}.")
    if args.compute_tool is None:
        print("No Vulkan dispatch performed. Use --compute-tool to run the emitted fixtures.")


if __name__ == "__main__":
    main()
