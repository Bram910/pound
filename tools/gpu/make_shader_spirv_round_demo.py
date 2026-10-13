#!/usr/bin/env python3
"""Check FP32 integral rounding against exact rational expected states."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import struct
import subprocess

from make_shader_spirv_arithmetic_demo import command, encode, finite, operand, rounded


def round_integral(bits, mode, ftz=0, modifiers=0):
    bits = operand(bits, modifiers, ftz)
    magnitude = bits & 0x7FFFFFFF
    if magnitude > 0x7F800000:
        return 0x7FFFFFFF
    if magnitude == 0x7F800000:
        return bits
    value = finite(bits)
    absolute = abs(value)
    integral, remainder = divmod(absolute.numerator, absolute.denominator)
    increment = ((remainder * 2 > absolute.denominator
                  or (remainder * 2 == absolute.denominator and integral & 1)) if mode == 0
                 else bool(remainder) and ((mode == 1 and bits >> 31) or (mode == 2 and not bits >> 31)))
    integral += int(increment)
    if bits >> 31:
        integral = -integral
    return rounded(Fraction(integral), bits & 0x80000000, 0, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-round-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions, checks = [], []

    def append(destination, mode, ftz=0, modifiers=0, immediate=None,
               source=0, predicate=7, invert=0):
        second = source | (modifiers & 1) << 31 | (modifiers >> 1) << 30
        instructions.append(encode(0x107 | (1 if immediate is None else 4) << 9,
                                   destination, second=second if immediate is None else immediate,
                                   third=0x201000 | mode << 14 | ftz << 16, predicate=predicate, invert=invert))
        checks.append((destination, mode, ftz, modifiers, immediate, source, predicate, invert))

    for mode in range(4):
        for ftz in range(2):
            for modifiers in range(4):
                append(32 + len(checks), mode, ftz, modifiers)
    for immediate in (0, 0x80000000, 1, 0x80000001, 0x3F000000, 0xBF000000,
                      0x3FC00000, 0xBFC00000, 0x40200000, 0xC0200000, 0x4AFFFFFF,
                      0x4B000000, 0xCAFFFFFF, 0xCB000000, 0x7F800000, 0xFF800000,
                      0x7F800001, 0xFFC00000, 0x7FFFFFFF, 0xFFFFFFFF):
        append(32 + len(checks), len(checks) % 4, len(checks) % 2, immediate=immediate)
    for predicate, invert in ((0, 0), (0, 1), (1, 0), (1, 1), (7, 1)):
        append(32 + len(checks), 2, predicate=predicate, invert=invert)
    append(32 + len(checks), 1, source=255)
    append(32 + len(checks), 2, modifiers=1, source=255)
    append(0, 0, source=0, predicate=0)
    append(1, 1, modifiers=3, source=1, predicate=0, invert=1)
    append(255, 0)
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "round-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if (trace[0]["operation"] != "FRND" or trace[24]["float_rounding"] != 3
            or trace[4]["float_ftz"] != 1 or trace[3]["sources"][0]["absolute"] != 1
            or trace[32]["sources"][0]["kind"] != "immediate"):
        raise RuntimeError("Integral-rounding trace fields differ from the encodings")
    edge = (0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
            0x80800000, 0x3F000000, 0xBF000000, 0x3EFFFFFF, 0x3F000001, 0xBEFFFFFF,
            0xBF000001, 0x3FC00000, 0xBFC00000, 0x40200000, 0xC0200000, 0x40200001,
            0xC0200001, 0x4AFFFFFF, 0x4B000000, 0xCAFFFFFF, 0xCB000000, 0x4F800000,
            0xCF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7F800001,
            0xFF800001, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF, 0xFFFFFFFF)
    randomizer = random.Random(0xA017E47)
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
        for destination, mode, ftz, modifiers, immediate, source, predicate, invert in checks:
            active = (1 if predicate == 7 else result[255] >> predicate & 1) ^ invert
            bits = immediate if immediate is not None else 0 if source == 255 else result[source]
            if active and destination != 255:
                result[destination] = round_integral(bits, mode, ftz, modifiers)
        expected.extend(result)
    zero_result = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != zero_result or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions) or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from exact integral-rounding results")
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
        module = args.output_dir / f"round-{mode}.spv"
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
        chain = args.output_dir / "round-loop.bin"
        chain.write_bytes(b"".join((encode(0x307, 0, second=0, third=0x201000),
                                    encode(0x421, 0, first=0, second=0x3F400000),
                                    encode(0x80B, first=0, second=0x40800000, third=1 << 12 | 1 << 20 | 7 << 23),
                                    encode(0x947, second=0xFFFFFFC0, third=0x0383FFFF, predicate=0),
                                    encode(0x94D, third=7 << 23))))
        source, wanted = [], []
        for mask in range(128):
            record = [0] * 261
            record[0] = 0x3E800000
            record[255] = mask
            source.extend(record)
            record[0] = 0x40980000
            record[255:261] = [(mask & ~3) | 2, 80, 21, 0, 1, 0]
            wanted.extend(record)
        chain_input, chain_output = args.output_dir / "chain-input.bin", args.output_dir / "chain-actual.bin"
        chain_input.write_bytes(struct.pack(f"<{len(source)}I", *source))
        chain_report = command([str(args.compute_tool.resolve()), "bounded-chain", str(chain),
                                str(chain_input), str(chain_output), "3", "8", args.compute_device]).stdout.strip()
        if chain_output.read_bytes() != struct.pack(f"<{len(wanted)}I", *wanted) or "7 dispatches" not in chain_report:
            raise RuntimeError(f"Persistent integral-rounding loop differs: {chain_report}")
    rejected = (("source-f16", encode(0x307, second=0, third=0x101000)),
                ("destination-f16", encode(0x307, second=0, third=0x200800)),
                ("f64", encode(0x313, second=0, third=0x301800)),
                ("constant-buffer", encode(0x707, second=0, third=0x201000)),
                ("uniform", encode(0xB07, second=0, third=0x201000)),
                ("unused-source", encode(0x307, first=1, second=0, third=0x201000)),
                ("unused-source-bits", encode(0x307, second=0x100, third=0x201000)),
                ("unsupported-immediate-form", encode(0x507, second=0, third=0x201000)),
                ("flag-bit8", encode(0x307, second=0, third=0x201100)),
                ("flag-bit10", encode(0x307, second=0, third=0x201400)),
                ("flag-bit13", encode(0x307, second=0, third=0x203000)),
                ("extra-bits", encode(0x307, second=0, third=0x221000)))
    sentinel = b"rounding rejection must preserve this output"
    output = args.output_dir / "rejected.spv"
    for name, instruction in rejected:
        bad_input = args.output_dir / f"rejected-{name}.bin"
        bad_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(bad_input), str(output)], capture_output=True, timeout=30)
            if result.returncode == 0 or output.read_bytes() != sentinel:
                raise RuntimeError(f"Rounding rejection changed output: {name}, {mode}")
    report = {"kind": "FP32 integral rounding with exact rational expected states",
              "instruction_count": len(instructions), "independent_checks": reports,
              "rejections": [name for name, _ in rejected],
              "persistent_chain_lanes": 128 if chain_report is not None else 0,
              "persistent_chain_report": chain_report}
    (args.output_dir / "round-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} exact rational integral-rounding states per path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. Vulkan dispatch checked: {args.compute_tool is not None}.")


if __name__ == "__main__":
    main()
