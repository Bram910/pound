#!/usr/bin/env python3
"""Check prepared SM86 execution, basic blocks, branches and strict validation."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


def encode(opcode, destination=0, first=0, second=0, third=0, predicate=7, invert=0):
    return struct.pack("<IIII", opcode | predicate << 12 | invert << 15
                       | destination << 16 | first << 24, second, third, 0xFC000)


def branch(offset, target, predicate=7, invert=0, condition=7, condition_invert=0):
    displacement = ((target - offset - 16) // 4) & ((1 << 48) - 1)
    return encode(0x947, second=(displacement & ((1 << 30) - 1)) << 2,
                  third=(displacement >> 30) | condition << 23 | condition_invert << 26,
                  predicate=predicate, invert=invert)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-tool", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-shader-program-demo"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tool = str(args.shader_tool.resolve())
    terminate = encode(0x94D, third=7 << 23)
    nop = encode(0x918)
    move = encode(0x802, 1, second=7, third=0xF00)
    reports = []

    def invoke(mode, path, budget=None):
        command = [tool, mode, str(path)] + ([] if budget is None else [str(budget)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return result.returncode, json.loads(result.stdout) if result.stdout else None

    cases = [
        ("straight", move + terminate, [(0, 2, [], 1, 0)]),
        ("diamond", branch(0, 32, condition=0) + branch(16, 48) + move + terminate,
         [(0, 1, [2, 1], 0, 0), (16, 1, [3], 0, 0), (32, 1, [3], 0, 0), (48, 1, [], 1, 0)]),
        ("constant-taken", branch(0, 32) + move + terminate,
         [(0, 1, [2], 0, 0), (16, 1, [2], 0, 0), (32, 1, [], 1, 0)]),
        ("constant-skipped", branch(0, 32, invert=1) + move + terminate,
         [(0, 1, [1], 0, 0), (16, 1, [2], 0, 0), (32, 1, [], 1, 0)]),
        ("contradictory-guards", branch(0, 32, predicate=0, condition=0, condition_invert=1) + move + terminate,
         [(0, 1, [1], 0, 0), (16, 1, [2], 0, 0), (32, 1, [], 1, 0)]),
        ("deduplicated", branch(0, 16, condition=0) + terminate,
         [(0, 1, [1], 0, 0), (16, 1, [], 1, 0)]),
        ("self-loop", branch(0, 0) + terminate,
         [(0, 1, [0], 0, 0), (16, 1, [], 1, 0)]),
        ("predicated-exit", encode(0x94D, third=7 << 23, predicate=0) + move + terminate,
         [(0, 1, [1], 1, 0), (16, 2, [], 1, 0)]),
        ("false-exit", encode(0x94D, third=7 << 23, invert=1) + nop,
         [(0, 1, [1], 0, 0), (16, 1, [], 0, 1)]),
        ("falloff", move, [(0, 1, [], 0, 1)]),
    ]
    for label, data, blocks in cases:
        path = args.output_dir / f"{label}.bin"
        path.write_bytes(data)
        status, graph = invoke("analyze", path)
        wanted = {"instruction_count": len(data) // 16, "block_count": len(blocks),
                  "blocks": [{"id": index, "offset": offset, "instruction_count": count,
                              "successors": successors, "exits": exits, "falls_off": falls_off}
                             for index, (offset, count, successors, exits, falls_off) in enumerate(blocks)]}
        if status or graph != wanted:
            raise RuntimeError(f"{label}: graph {graph}, expected {wanted}")
        for budget in (1, 2, 3, 97):
            raw = invoke("run", path, budget)
            prepared = invoke("run-prepared", path, budget)
            if raw != prepared:
                raise RuntimeError(f"{label}: raw/prepared execution differs at budget {budget}")
        reports.append({"case": label, "graph": graph, "result": prepared})
    for limit in (1, 2, 37, 1000):
        add = encode(0x810, 1, first=1, second=1, third=0x07FFE0FF)
        comparison = encode(0x80C, first=1, second=limit, third=0x03901070)
        data = add + comparison + branch(32, 0, predicate=0) + terminate
        path = args.output_dir / f"counted-{limit}.bin"
        path.write_bytes(data)
        for budget in (1, 4, 31, 3 * limit + 1):
            raw = invoke("run", path, budget)
            prepared = invoke("run-prepared", path, budget)
            if raw != prepared:
                raise RuntimeError(f"Counted loop {limit}: raw/prepared execution differs")
        if prepared[0] or prepared[1]["registers"] != {"R1": limit}:
            raise RuntimeError(f"Counted loop {limit}: wrong result")
        status, graph = invoke("analyze", path)
        if status or graph["blocks"][0]["successors"] != [0, 1]:
            raise RuntimeError(f"Counted loop {limit}: wrong control-flow graph")
        reports.append({"case": f"counted-{limit}", "graph": graph, "result": prepared})
    for label, data in (("unreachable-invalid", terminate + bytes(16)),
                        ("skipped-invalid-target", branch(0, 128, invert=1) + terminate),
                        ("negative-invalid-target", branch(0, -16) + terminate)):
        path = args.output_dir / f"{label}.bin"
        path.write_bytes(data)
        for mode in ("analyze", "run-prepared"):
            status, output = invoke(mode, path)
            if not status or output is not None:
                raise RuntimeError(f"{label}: preparation should reject the entire program")
    (args.output_dir / "program-report.json").write_text(
        json.dumps({"kind": "prepared reference programs, not native shader compilation", "programs": reports}, indent=2)
        + "\n", encoding="utf-8")
    print("Verified 14 prepared programs, control-flow graphs and raw/prepared budget results;")
    print("self/counting loops, predicate guards, exits/falloff and strict whole-program rejection.")


if __name__ == "__main__":
    main()
