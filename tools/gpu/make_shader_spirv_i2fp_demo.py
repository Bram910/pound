#!/usr/bin/env python3
"""Check the I2FP FP32/32-bit subset against exact integer-to-float results."""

import argparse
import json
from pathlib import Path
import random
import struct
import subprocess

from make_shader_spirv_arithmetic_demo import command, encode
from make_shader_spirv_conversion_demo import convert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-i2fp-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    instructions = [encode(0x845, 6, second=0xFFFFFFFF, third=0x201400),
                    encode(0x245, 6, second=255, third=0x20D000, predicate=1, invert=1),
                    encode(0x802, 7, second=0xBF800000, third=0xF00),
                    encode(0x802, 7, second=0, third=0xF00, predicate=0),
                    encode(0x906, 8, second=0xFFFFFFFF, third=0x201400),
                    encode(0x306, 8, second=255, third=0x20D400, predicate=2, invert=1)]
    checks = []

    def append(destination, signed, mode, immediate=None, source=0, predicate=7, invert=0, legacy=False):
        instructions.append(encode((0x106 if legacy else 0x45) | (1 if immediate is None else 4) << 9,
                                   destination, second=source if immediate is None else immediate,
                                   third=0x201000 | signed << 10 | mode << 14, predicate=predicate, invert=invert))
        checks.append((destination, signed, mode, immediate, source, predicate, invert))

    for signed in range(2):
        for mode in (0, 3):
            for source in (0, 1, 2, 255):
                for legacy in (False, True):
                    append(32 + len(checks), signed, mode, source=source, legacy=legacy)
            for predicate, invert in ((0, 0), (0, 1), (1, 0), (1, 1), (7, 0), (7, 1)):
                append(32 + len(checks), signed, mode, predicate=predicate, invert=invert)
            for immediate in (0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x01000000, 0x01000001,
                              0x01000003, 0x01FFFFFF, 0x02000001, 0x80000001, 0xFFFFFF7F,
                              0x7FFFFF80, 0x80000080, 0x10000000):
                append(32 + len(checks), signed, mode, immediate=immediate)
    append(0, 1, 0, source=0, predicate=0)
    append(1, 0, 3, source=1, predicate=0, invert=1)
    append(255, 1, 0, source=255)
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "i2fp-compute.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([tool, "trace", str(raw)]).stdout.splitlines()]
    if (trace[6]["operation"] != "I2FP" or trace[7]["operation"] != "I2F"
            or trace[35]["float_rounding"] != 3 or trace[64]["conversion_signed"] != 1
            or trace[20]["sources"][0]["kind"] != "immediate"):
        raise RuntimeError("I2FP trace fields differ from the encodings")
    edge = (0, 1, 2, 3, 0xFFFFFFFF, 0xFFFFFFFE, 0x7FFFFFFF, 0x80000000, 0x80000001,
            0x7FFFFF80, 0x80000080, 0x00FFFFFE, 0x00FFFFFF, 0x01000000, 0x01000001,
            0x01000002, 0x01000003, 0x01000005, 0x01FFFFFF, 0x02000001, 0xFEFFFFFF,
            0xFEFFFFFD, 0xFFFFFF7F, 0xFFFFFF80, 0x10000000, 0x10000008, 0x10000010,
            0x10000018, 0xE0000001, 0x40000001, 0xC0000001, 0xFF000001)
    randomizer = random.Random(0x12FC017)
    inputs, expected = [], []
    lanes = 512
    for lane in range(lanes):
        record = [0] * 256
        if lane:
            record[0] = edge[lane % len(edge)] if lane < 384 else randomizer.getrandbits(32)
            record[1] = edge[(lane * 7) % len(edge)] if lane < 384 else randomizer.getrandbits(32)
            record[2] = edge[(lane * 13) % len(edge)] if lane < 384 else randomizer.getrandbits(32)
            record[255] = lane % 128
        inputs.extend(record)
        result = record.copy()
        result[6] = 0xBF800000 if record[255] & 2 else 0
        result[7] = 0 if record[255] & 1 else 0xBF800000
        result[8] = 0xBF800000 if record[255] & 4 else 0
        for destination, signed, mode, immediate, source, predicate, invert in checks:
            active = (1 if predicate == 7 else result[255] >> predicate & 1) ^ invert
            bits = immediate if immediate is not None else 0 if source == 255 else result[source]
            if active and destination != 255:
                result[destination] = convert(bits, 0, signed, mode)
        expected.extend(result)
    zero_result = {f"R{index}": value for index, value in enumerate(expected[:255]) if value}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([tool, mode, str(raw)]).stdout)
        if (result["registers"] != zero_result or result["predicates"] != expected[255]
                or result["pc"] != len(instructions) * 16 or result["steps"] != len(instructions) or result["halted"] != 1):
            raise RuntimeError(f"CPU {mode} differs from exact I2FP results")
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
        module = args.output_dir / f"i2fp-{mode}.spv"
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
        chain = args.output_dir / "i2fp-loop.bin"
        chain.write_bytes(b"".join((encode(0x245, 1, second=0, third=0x201400),
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
            raise RuntimeError(f"Persistent I2FP loop differs: {chain_report}")
    rejected = (("round-down", encode(0x245, second=0, third=0x205000)),
                ("round-up", encode(0x245, second=0, third=0x209000)),
                ("destination-f16", encode(0x245, second=0, third=0x200800)),
                ("destination-f64", encode(0x245, second=0, third=0x201800)),
                ("source-s16", encode(0x245, second=0, third=0x101400)),
                ("source-s64", encode(0x245, second=0, third=0x301400)),
                ("constant-buffer", encode(0x645, second=0, third=0x201000)),
                ("uniform", encode(0xA45, second=0, third=0x201000)),
                ("unused-source", encode(0x245, first=1, second=0, third=0x201000)),
                ("unused-source-bits", encode(0x245, second=0x100, third=0x201000)),
                ("integer-absolute", encode(0x245, second=0x40000000, third=0x201000)),
                ("integer-negate", encode(0x245, second=0x80000000, third=0x201000)),
                ("ftz", encode(0x245, second=0, third=0x211000)),
                ("extra-bits", encode(0x245, second=0, third=0x203000)))
    sentinel = b"I2FP rejection must preserve this output"
    output = args.output_dir / "rejected.spv"
    for name, instruction in rejected:
        bad_input = args.output_dir / f"rejected-{name}.bin"
        bad_input.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            output.write_bytes(sentinel)
            result = subprocess.run([tool, mode, str(bad_input), str(output)], capture_output=True, timeout=30)
            if result.returncode == 0 or output.read_bytes() != sentinel:
                raise RuntimeError(f"I2FP rejection changed output: {name}, {mode}")
    report = {"kind": "I2FP FP32/signed and unsigned 32-bit RN/RZ conversions with exact rational expected states",
              "instruction_count": len(instructions), "independent_checks": reports,
              "predicated_zero_regressions": ["I2FP", "MOV", "I2F"],
              "rejections": [name for name, _ in rejected],
              "persistent_chain_lanes": 128 if chain_report is not None else 0,
              "persistent_chain_report": chain_report}
    (args.output_dir / "i2fp-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {lanes} exact rational I2FP states per path.")
    print(f"Khronos validation checked: {args.spirv_val is not None}. Vulkan dispatch checked: {args.compute_tool is not None}.")


if __name__ == "__main__":
    main()
