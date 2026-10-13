import argparse
import json
from pathlib import Path
import random
import struct
import subprocess

from make_shader_spirv_arithmetic_demo import command, encode


def expanded(word, ftz, modifiers):
    half = word & 0xFFFF
    if modifiers & 2:
        half &= 0x7FFF
    if modifiers & 1:
        half ^= 0x8000
    if half & 0x7C00 == 0x7C00 and half & 0x3FF:
        return 0x7FFFFFFF
    if ftz and half & 0x7C00 == 0:
        half &= 0x8000
    return struct.unpack("<I", struct.pack("<f", struct.unpack("<e", struct.pack("<H", half))[0]))[0]


def main():
    parser = argparse.ArgumentParser(description="Check scalar FP16-to-FP32 widening against Python binary16.")
    parser.add_argument("--shader-tool", type=Path, required=True)
    parser.add_argument("--compute-tool", type=Path)
    parser.add_argument("--compute-device", default="auto")
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/gpu-widen-demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instructions, checks = [], []
    for rounding in range(4):
        for ftz in range(2):
            for modifiers in range(4):
                destination = len(instructions) + 16
                instructions.append(encode(0x304, destination, second=modifiers % 2 << 31 | modifiers // 2 << 30,
                                           third=0x101000 | rounding << 14 | ftz << 16))
                checks.append((destination, 0, ftz, modifiers, 7, 0))
    for predicate, invert in ((0, 0), (0, 1), (1, 0), (1, 1), (7, 1)):
        destination = len(instructions) + 16
        instructions.append(encode(0x304, destination, second=1, third=0x101000,
                                   predicate=predicate, invert=invert))
        checks.append((destination, 1, 0, 0, predicate, invert))
    for destination, source in ((2, 255), (1, 1), (0, 0), (255, 1)):
        instructions.append(encode(0x304, destination, second=source, third=0x101000))
        checks.append((destination, source, 0, 0, 7, 0))
    instructions.append(encode(0x94D, third=7 << 23))
    raw = args.output_dir / "widen.bin"
    raw.write_bytes(b"".join(instructions))
    trace = [json.loads(line) for line in command([str(args.shader_tool.resolve()), "trace", str(raw)]).stdout.splitlines()]
    if any(item["float_source_bits"] != 16 or item["float_destination_bits"] != 32 for item in trace[:-1]):
        raise RuntimeError("Widening format metadata mismatch")
    randomizer = random.Random(0xF1632)
    inputs, expected = [], []
    edge = (0, 0x8000, 1, 0x8001, 0x3FF, 0x83FF, 0x400, 0x8400, 0x3C00, 0xBC00,
            0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF)
    lanes = 4096
    for lane in range(lanes):
        record = [0] * 256
        if lane:
            record[0] = randomizer.getrandbits(16) << 16 | (edge[lane % len(edge)] if lane < 512 else randomizer.getrandbits(16))
            record[1] = randomizer.getrandbits(16) << 16 | edge[lane // len(edge) % len(edge)]
            record[255] = lane % 128
        inputs.append(record)
        result = record.copy()
        for destination, source, ftz, modifiers, predicate, invert in checks:
            if destination != 255 and ((1 if predicate == 7 else result[255] >> predicate & 1) ^ invert):
                result[destination] = expanded(0 if source == 255 else result[source], ftz, modifiers)
        expected.append(result)
    zero = {f"R{index}": word for index, word in enumerate(expected[0][:255]) if word}
    for mode in ("run", "run-prepared"):
        result = json.loads(command([str(args.shader_tool.resolve()), mode, str(raw)]).stdout)
        if result["registers"] != zero or result["halted"] != 1:
            raise RuntimeError(f"CPU {mode} differs from Python binary16")
    for mode in ("straight", "bounded"):
        source, wanted = [], []
        for record, result in zip(inputs, expected):
            source.extend(record)
            wanted.extend(result)
            if mode == "bounded":
                source.extend((0, 0, 0, 0, 0xDEADBEEF))
                wanted.extend((len(instructions) * 16, len(instructions), 0, 1, 0))
        module = args.output_dir / f"{mode}.spv"
        emission = [str(args.shader_tool.resolve()), "spirv" if mode == "straight" else "spirv-bounded", str(raw), str(module)]
        if mode == "bounded":
            emission.append(str(len(instructions)))
        command(emission)
        if args.spirv_val:
            command([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)])
        input_path, output_path = args.output_dir / f"{mode}-input.bin", args.output_dir / f"{mode}-actual.bin"
        input_path.write_bytes(struct.pack(f"<{len(source)}I", *source))
        expected_bytes = struct.pack(f"<{len(wanted)}I", *wanted)
        (args.output_dir / f"{mode}-expected.bin").write_bytes(expected_bytes)
        if args.compute_tool:
            dispatch = [str(args.compute_tool.resolve()), mode, str(raw), str(input_path), str(output_path)]
            if mode == "bounded":
                dispatch.append(str(len(instructions)))
            command(dispatch + [args.compute_device])
            if output_path.read_bytes() != expected_bytes:
                raise RuntimeError(f"{mode} differs from Python binary16 expected states")
    sentinel = b"preserve rejected widening output"
    rejected = (encode(0x904, second=1, third=0x101000), encode(0x304, second=1, third=0x101100),
                encode(0x304, second=1, third=0x301000), encode(0x304, first=1, second=1, third=0x101000))
    for index, instruction in enumerate(rejected):
        bad = args.output_dir / f"bad-{index}.bin"
        bad.write_bytes(instruction + instructions[-1])
        for mode in ("spirv", "spirv-bounded"):
            output_path.write_bytes(sentinel)
            result = subprocess.run([str(args.shader_tool.resolve()), mode, str(bad), str(output_path)], capture_output=True, timeout=30)
            if result.returncode == 0 or output_path.read_bytes() != sentinel:
                raise RuntimeError("Unsupported widening modified output")
    report = {"lanes_per_path": lanes, "paths": 2, "rejections": 8,
              "khronos_validation_checked": args.spirv_val is not None,
              "vulkan_dispatch_checked": args.compute_tool is not None}
    (args.output_dir / "widen-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
