#!/usr/bin/env python3
"""Emit bounded SM86 branch/loop modules with independent control-state fixtures."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


def encode(opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
    return struct.pack("<4I", opcode | predicate << 12 | invert << 15
                       | destination << 16 | first << 24, second, third, 0xFC000)


def branch(index, target, condition=7):
    relative = ((target - index - 1) * 4) & ((1 << 48) - 1)
    return encode(0x947, second=(relative & ((1 << 30) - 1)) << 2,
                  third=relative >> 30 | condition << 23)


def lane(registers=None, predicates=0, pc=0, steps=0, halted=0, status=0):
    record = [0] * 261
    for register, value in (registers or {}).items():
        record[register] = value
    record[255:] = [predicates, pc, steps & 0xFFFFFFFF, steps >> 32, halted, status]
    return record


def write_lanes(path, records):
    values = [value for record in records for value in record]
    path.write_bytes(struct.pack(f"<{len(values)}I", *values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--spirv-val", type=Path)
    parser.add_argument("--compute-tool", type=Path, help="optionally dispatch with PoundGpuCompute")
    parser.add_argument("--compute-device", default="auto", help="auto or Vulkan physical-device index")
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-spirv-control-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    terminate = encode(0x94D, third=7 << 23)
    move = encode(0x802, 1, second=7, third=0xF00)
    diamond = branch(0, 2, condition=0) + branch(1, 3) + move + terminate
    guarded = encode(0x94D, third=7 << 23, predicate=0) + move + terminate
    counted = (encode(0x810, 1, first=1, second=1, third=0x07FFE0FF)
               + encode(0x80C, first=1, second=37, third=0x03F01070)
               + branch(2, 0, condition=0) + terminate)
    maximum_steps = (1 << 64) - 1
    cases = [
        ("diamond-first", diamond, 1, [lane(), lane(predicates=1)],
         [lane(pc=16, steps=1, status=10), lane(predicates=1, pc=32, steps=1, status=10)]),
        ("diamond-finished", diamond, 3, [lane(), lane(predicates=1)],
         [lane(pc=64, steps=3, halted=1), lane({1: 7}, predicates=1, pc=64, steps=3, halted=1)]),
        ("guarded-exit", guarded, 3, [lane(), lane(predicates=1)],
         [lane({1: 7}, pc=48, steps=3, halted=1), lane(predicates=1, pc=16, steps=1, halted=1)]),
        ("counted-partial", counted, 3, [lane()], [lane({1: 1}, predicates=1, steps=3, status=10)]),
        ("counted-resumed", counted, 109, [lane({1: 1}, predicates=1, steps=3, status=10)],
         [lane({1: 37}, pc=64, steps=112, halted=1)]),
        ("self-loop", branch(0, 0), 97, [lane()], [lane(steps=97, status=10)]),
        ("maximum-budget", branch(0, 0), 4096, [lane()], [lane(steps=4096, status=10)]),
        ("falloff-budget", move, 1, [lane()], [lane({1: 7}, pc=16, steps=1, status=10)]),
        ("falloff-fault", move, 3, [lane()], [lane({1: 7}, pc=16, steps=1, status=8)]),
        ("step-overflow", move + terminate, 3, [lane(steps=maximum_steps)],
         [lane(steps=maximum_steps, status=10)]),
        ("low-step-carry", terminate, 1, [lane(steps=0xFFFFFFFF)],
         [lane(pc=16, steps=0x100000000, halted=1)]),
        ("already-halted", terminate, 3, [lane(steps=maximum_steps, halted=1)],
         [lane(steps=maximum_steps, halted=1)]),
        ("invalid-state", terminate, 3, [lane({1: 123}, predicates=0x80000000, pc=3, steps=maximum_steps)],
         [lane({1: 123}, predicates=0x80000000, pc=3, steps=maximum_steps, status=1)]),
    ]
    reports = []
    for name, code, budget, initial, expected in cases:
        raw = args.output_dir / f"{name}.bin"
        module = args.output_dir / f"{name}.spv"
        raw.write_bytes(code)
        subprocess.run([tool, "spirv-bounded", str(raw), str(module), str(budget)],
                       capture_output=True, text=True, check=True, timeout=30)
        validated = False
        if args.spirv_val:
            subprocess.run([str(args.spirv_val.resolve()), "--target-env", "vulkan1.1", str(module)],
                           capture_output=True, text=True, check=True, timeout=30)
            validated = True
        if initial[0] == lane():
            reference = subprocess.run([tool, "run-prepared", str(raw), str(budget)],
                                       capture_output=True, text=True, timeout=30)
            wanted = expected[0]
            state = json.loads(reference.stdout)
            if (state["registers"] != {f"R{index}": value for index, value in enumerate(wanted[:255]) if value}
                    or state["predicates"] != wanted[255] or state["pc"] != wanted[256]
                    or state["steps"] != wanted[257] | wanted[258] << 32
                    or state["halted"] != wanted[259] or (reference.returncode == 0) != (wanted[260] == 0)):
                raise RuntimeError(f"Reference state differs: {name}: {state}")
        write_lanes(args.output_dir / f"{name}-input.bin", initial)
        write_lanes(args.output_dir / f"{name}-expected.bin", expected)
        dispatch_report = None
        if args.compute_tool:
            input_path = args.output_dir / f"{name}-input.bin"
            if name == "counted-resumed":
                partial = args.output_dir / "counted-partial-actual.bin"
                if partial.read_bytes() != input_path.read_bytes():
                    raise RuntimeError("Partial output does not match the resumption input")
                input_path = partial
            actual = args.output_dir / f"{name}-actual.bin"
            dispatch = subprocess.run(
                [str(args.compute_tool.resolve()), "bounded", str(raw), str(input_path), str(actual),
                 str(budget), args.compute_device], capture_output=True, text=True, check=True, timeout=60)
            if actual.read_bytes() != (args.output_dir / f"{name}-expected.bin").read_bytes():
                raise RuntimeError(f"Vulkan control fixture mismatch: {name}")
            dispatch_report = dispatch.stdout.strip()
        reports.append({"case": name, "budget": budget, "lanes": len(initial),
                        "khronos_validation_checked": validated,
                        "vulkan_dispatch_checked": dispatch_report is not None,
                        "vulkan_dispatch_report": dispatch_report})
    output = args.output_dir / "rejected-output.spv"
    sentinel = b"preserve output"
    raw = args.output_dir / "self-loop.bin"
    for budget in ("0", "4097", "-1", "1x"):
        output.write_bytes(sentinel)
        result = subprocess.run([tool, "spirv-bounded", str(raw), str(output), budget],
                                capture_output=True, timeout=30)
        if result.returncode == 0 or output.read_bytes() != sentinel:
            raise RuntimeError(f"Invalid budget changed output: {budget}")
    for name, code in (("unreachable-invalid", terminate + bytes(16)),
                       ("invalid-target", branch(0, 128) + terminate),
                       ("instruction-limit", encode(0x918) * 1025)):
        raw = args.output_dir / f"rejected-{name}.bin"
        raw.write_bytes(code)
        output.write_bytes(sentinel)
        result = subprocess.run([tool, "spirv-bounded", str(raw), str(output), "1"],
                                capture_output=True, timeout=30)
        if result.returncode == 0 or output.read_bytes() != sentinel:
            raise RuntimeError(f"Rejected program changed output: {name}")
    chains = []
    if args.compute_tool:
        chain_cases = [
            ("counted-chain", "counted-partial", 3, 40, 38, [lane({1: 37}, pc=64, steps=112, halted=1)]),
            ("self-loop-chain", "self-loop", 3, 4, 4, [lane(steps=12, status=10)]),
            ("overflow-chain", "step-overflow", 3, 40, 1, [lane(steps=maximum_steps, status=10)]),
        ]
        for name, source, budget, dispatch_limit, dispatch_count, expected in chain_cases:
            actual = args.output_dir / f"{name}-actual.bin"
            wanted = args.output_dir / f"{name}-expected.bin"
            write_lanes(wanted, expected)
            dispatch = subprocess.run(
                [str(args.compute_tool.resolve()), "bounded-chain",
                 str(args.output_dir / f"{source}.bin"), str(args.output_dir / f"{source}-input.bin"),
                 str(actual), str(budget), str(dispatch_limit), args.compute_device],
                capture_output=True, text=True, check=True, timeout=60)
            if actual.read_bytes() != wanted.read_bytes():
                raise RuntimeError(f"Persistent Vulkan chain mismatch: {name}")
            if f"for {dispatch_count} dispatches" not in dispatch.stdout:
                raise RuntimeError(f"Unexpected persistent dispatch count: {name}")
            chains.append({"case": name, "dispatches": dispatch_count,
                           "vulkan_dispatch_checked": True, "vulkan_dispatch_report": dispatch.stdout.strip()})
    report = {"kind": "bounded independent-lane SPIR-V fixtures, not rendering or SIMT reconvergence",
              "lane_words": 261, "state_words": {"predicates": 255, "pc": 256, "steps_low": 257,
                                                  "steps_high": 258, "halted": 259, "status": 260},
              "cases": reports, "persistent_chains": chains}
    (args.output_dir / "control-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Emitted 13 bounded control-flow fixtures: branches, loops, resumption, exits and fault states.")
    if args.compute_tool:
        print("All independent control fixtures passed Vulkan dispatch, including chained resumption.")
        print(f"Persistent session/kernel checks passed: {len(chains)} bounded chains.")
    else:
        print("No Vulkan dispatch performed. Use --compute-tool to check emitted fixtures on a device.")


if __name__ == "__main__":
    main()
