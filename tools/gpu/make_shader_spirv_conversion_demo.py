#!/usr/bin/env python3
"""Check FP32/32-bit integer conversions against exact rational expected states."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import struct
import subprocess

from make_shader_spirv_arithmetic_demo import command, encode, finite, operand, rounded


def convert(bits, to_integer, signed, mode, ftz=0, ntz=0, modifiers=0):
    if not to_integer:
        value = bits - (1 << 32) if signed and bits >> 31 else bits
        return rounded(Fraction(value), 0x80000000 if value < 0 else 0, mode, 0)
    bits = operand(bits, modifiers, ftz)
    magnitude = bits & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        return 0 if ntz else 0x80000000
    minimum, maximum = (-(1 << 31), (1 << 31) - 1) if signed else (0, (1 << 32) - 1)
    if magnitude == 0x7F800000:
        return (minimum if bits >> 31 else maximum) & 0xFFFFFFFF
    value = finite(bits)
    absolute = abs(value)
    integral, remainder = divmod(absolute.numerator, absolute.denominator)
    increment = ((remainder * 2 > absolute.denominator
                  or (remainder * 2 == absolute.denominator and integral & 1)) if mode == 0
                 else bool(remainder) and ((mode == 1 and value < 0) or (mode == 2 and value >= 0)))
    integral += int(increment)
    if value < 0:
        integral = -integral
    return max(minimum, min(maximum, integral)) & 0xFFFFFFFF


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-conversion-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions, checks = [], []

    def append(destination, to_integer, signed, mode, ftz=0, ntz=0, modifiers=0,
               immediate=None, source=0, predicate=7, invert=0):
        second = source | (modifiers & 1) << 31 | (modifiers >> 1) << 30
        third = 0x201000 | signed << (8 if to_integer else 10) | mode << 14 | ftz << 16 | ntz << 13
        instructions.append(encode((0x105 if to_integer else 0x106) | (1 if immediate is None else 4) << 9,
                                   destination, second=second if immediate is None else immediate,
                                   third=third, predicate=predicate, invert=invert))
        checks.append((destination, to_integer, signed, mode, ftz, ntz, modifiers,
                       immediate, source, predicate, invert))

    for signed in range(2):
        for mode in range(4):
            for ftz in range(2):
                for ntz in range(2):
                    for modifiers in range(4):
                        append(32 + len(checks), 1, signed, mode, ftz, ntz, modifiers)
            append(32 + len(checks), 0, signed, mode, source=1)
    for immediate in (0, 1, 0x80000000, 0xFFFFFFFF, 0x3FC00000, 0x7F800000, 0x7FC00000):
        for to_integer in range(2):
            append(32 + len(checks), to_integer, len(checks) % 2, len(checks) % 4,
                   ntz=to_integer, immediate=immediate)
    for predicate, invert in ((0, 0), (0, 1), (1, 0), (1, 1), (7, 1)):
        append(32 + len(checks), 1, 1, 2, ntz=1, predicate=predicate, invert=invert)
    append(32 + len(checks), 1, 0, 1, source=255)
    append(32 + len(checks), 0, 1, 2, source=255)
    append(0, 1, 1, 3, ntz=1, predicate=0)
    append(1, 0, 1, 0, source=1, predicate=0, invert=1)
    append(255, 1, 1, 0)
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "conversion-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if (trace[0]["operation"] != "F2I" or trace[16]["operation"] != "I2F"
            or trace[8]["float_nan_to_zero"] != 0 or trace[4]["float_nan_to_zero"] != 1
            or trace[68]["conversion_signed"] != 1):
        raise RuntimeError("Conversion trace fields differ from the encoded instructions")
    edge = (0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
            0x80800000, 0x3F000000, 0xBF000000, 0x3EFFFFFF, 0x3F000001, 0x3FC00000,
            0xBFC00000, 0x40200000, 0xC0200000, 0x4EFFFFFF, 0x4F000000, 0xCF000000,
            0xCF000001, 0x4F7FFFFF, 0x4F800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000,
            0xFF800000, 0x7F800001, 0xFF800001, 0x7FC00000, 0xFFC00000, 0xFFFFFFFF,
            0x7FFFFFFF, 0x01000001, 0xFEFFFFFF, 0x7FFFFF80, 0x80000080)
    randomizer = random.Random(0xC017E47)
    inputs, expected = [], []
    lanes = 512
    for lane in range(lanes):
        record = [0] * 256
        if lane:
            record[0] = edge[lane % len(edge)] if lane < 384 else randomizer.getrandbits(32)
            record[1] = edge[(lane * 7) % len(edge)] if lane < 384 else randomizer.getrandbits(32)
            record[255] = lane % 128
        inputs.extend(record)
        result = record.copy()
        for destination, to_integer, signed, mode, ftz, ntz, modifiers, immediate, source, predicate, invert in checks:
            active = (1 if predicate == 7 else result[255] >> predicate & 1) ^ invert
            bits = immediate if immediate is not None else 0 if source == 255 else result[source]
            if active and destination != 255:
                result[destination] = convert(bits, to_integer, signed, mode, ftz, ntz, modifiers)
        expected.extend(result)
    zero_result = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != zero_result or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions) or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from exact conversion results")
    reports = []
    for mode in ("straight", "bounded"):
        stride = 256 if mode == "straight" else 261
        source, wanted = [], []
        for lane in range(lanes):
            source.extend(inputs[lane * 256:(lane + 1) * 256])
            wanted.extend(expected[lane * 256:(lane + 1) * 256])
            if mode == "bounded":
                source.extend((0, 0, 0, 0, 0xDEADBEEF))
                wanted.extend((len(instructions) * 16, len(instructions), 0, 1, 0))
        module = args.output_dir / f"conversion-{mode}.spv"
        emission = [tool, "spirv" if mode == "straight" else "spirv-bounded", str(raw), str(module)]
        if mode == "bounded":
            emission.append(str(len(instructions)))
        command(emission)
        if args.spirv_val:
            command([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)])
        source_path = args.output_dir / f"{mode}-input.bin"
        source_path.write_bytes(struct.pack(f"<{len(source)}I", *source))
        (args.output_dir / f"{mode}-expected.bin").write_bytes(struct.pack(f"<{len(wanted)}I", *wanted))
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
        chain = args.output_dir / "conversion-loop.bin"
        chain.write_bytes(b"".join((encode(0x306, 1, second=0, third=0x201400),
                                    encode(0x421, 1, first=1, second=0x3F800000),
                                    encode(0x305, 0, second=1, third=0x20F100),
                                    encode(0x80C, first=0, second=4, third=0x70 | 1 << 12 | 1 << 20 | 7 << 23),
                                    encode(0x947, second=0xFFFFFFB0, third=0x0383FFFF, predicate=0),
                                    encode(0x94D, third=7 << 23))))
        source, wanted = [], []
        for mask in range(128):
            record = [0] * 261
            record[255] = mask
            source.extend(record)
            record[0:2] = [4, 0x40800000]
            record[255:261] = [(mask & ~3) | 2, 96, 21, 0, 1, 0]
            wanted.extend(record)
        chain_input, chain_output = args.output_dir / "chain-input.bin", args.output_dir / "chain-actual.bin"
        chain_input.write_bytes(struct.pack(f"<{len(source)}I", *source))
        chain_report = command([str(args.compute_tool.resolve()), "bounded-chain", str(chain),
                                str(chain_input), str(chain_output), "3", "8", args.compute_device]).stdout.strip()
        if chain_output.read_bytes() != struct.pack(f"<{len(wanted)}I", *wanted) or "7 dispatches" not in chain_report:
            raise RuntimeError(f"Persistent conversion loop differs: {chain_report}")
    rejected = (("f16", encode(0x305, second=0, third=0x101000)),
                ("s16", encode(0x305, second=0, third=0x200900)),
                ("f64", encode(0x311, second=0, third=0x301100)),
                ("integer64", encode(0x306, second=0, third=0x301400)),
                ("constant-buffer", encode(0x705, second=0, third=0x201000)),
                ("uniform", encode(0xB06, second=0, third=0x201000)),
                ("unused-source", encode(0x305, first=1, second=0, third=0x201000)),
                ("integer-absolute", encode(0x306, second=0x40000000, third=0x201000)),
                ("integer-negate", encode(0x306, second=0x80000000, third=0x201000)),
                ("integer-ftz", encode(0x306, second=0, third=0x211000)),
                ("integer-subop", encode(0x306, second=0x10000000, third=0x201000)),
                ("extra-bits", encode(0x305, second=0, third=0x221000)))
    sentinel = b"conversion rejection must preserve this output"
    output = args.output_dir / "rejected.spv"
    for name, instruction in rejected:
        bad_input = args.output_dir / f"rejected-{name}.bin"
        bad_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(bad_input), str(output)], capture_output=True, timeout=30)
            if result.returncode == 0 or output.read_bytes() != sentinel:
                raise RuntimeError(f"Conversion rejection changed output: {name}, {mode}")
    report = {"kind": "FP32/signed and unsigned 32-bit conversions with exact rational expected states",
              "instruction_count": len(instructions), "independent_checks": reports,
              "rejections": [name for name, _ in rejected],
              "persistent_chain_lanes": 128 if chain_report is not None else 0,
              "persistent_chain_report": chain_report}
    (args.output_dir / "conversion-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} exact rational conversion states per path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. Vulkan dispatch checked: {args.compute_tool is not None}.")


if __name__ == "__main__":
    main()
