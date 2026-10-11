#!/usr/bin/env python3
"""Verify timestamped semaphore records with an explicit synthetic replay clock."""

import argparse
import json
from pathlib import Path
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("build/gpu-completion-demo"), type=Path)
    parser.add_argument("--clock-step-ns", type=int, default=17)
    parser.add_argument("--segment-words", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.clock_step_ns <= 1_000_000_000:
        parser.error("--clock-step-ns must be between 1 and 1000000000")
    if not 1 <= args.segment_words <= 8191:
        parser.error("--segment-words must be between 1 and 8191")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = bytearray(b"\xA5") * 4096
    words = []
    records = (
        (0x0123456789ABCDEF, 0, 0x03100001, 0x0123456789ABCDEF, 64),
        (0xDEADBEEF, 0, 0x02100001, 0xDEADBEEF, 32),
        (9, 0xFFFFFFFF, 0xAA100006, 8, 32),
        (0xFEDCBA9876543210, 1, 0x8B100006, 0xFEDCBA9876543210, 64),
    )

    def methods(address, values):
        words.append((1 << 29) | (len(values) << 16) | (address // 4))
        words.extend(values)

    expected_records = []
    for index, (operand, initial, flags, value, bits) in enumerate(records):
        address = 2048 + index * 16
        struct.pack_into("<Q", image, address, initial)
        methods(0x28, [0, 0, index % 2, 0x28000000])
        methods(0x5C, [address, 0, operand & 0xFFFFFFFF, operand >> 32, flags])
        methods(0x64, [value & 0xFFFFFFFF, value >> 32, 0x01000000 if bits == 64 else 0])
        expected_records.append({"address": address, "payload_bits": bits,
                                 "payload": value, "timestamp_ns": (index + 1) * args.clock_step_ns})
    methods(0x78, [1])
    commands = struct.pack("<" + "I" * len(words), *words)
    command_address = 256
    if command_address + len(commands) > 2048:
        raise RuntimeError("Command buffer overlaps completion records")
    image[command_address:command_address + len(commands)] = commands
    entries = b"".join(struct.pack("<II", command_address + offset * 4,
                                   min(args.segment_words, len(words) - offset) << 10)
                       for offset in range(0, len(words), args.segment_words))
    memory_path = args.output_dir / "memory.bin"
    entries_path = args.output_dir / "gpfifo.bin"
    output_path = args.output_dir / "output-memory.bin"
    memory_path.write_bytes(image)
    entries_path.write_bytes(entries)
    subprocess.run([str(args.replay.resolve()), str(memory_path), str(entries_path),
                    str(output_path), f"--clock-step-ns={args.clock_step_ns}"], check=True)
    expected = bytearray(image)
    for record in expected_records:
        struct.pack_into("<QQ", expected, record["address"], record["payload"], record["timestamp_ns"])
    if output_path.read_bytes() != expected:
        raise RuntimeError("Completion record, zero padding, timestamp or untouched bytes differ")
    report = args.output_dir / "completion-records.json"
    report.write_text(json.dumps({"clock": "synthetic, not measured GPU time",
                                 "step_ns": args.clock_step_ns, "records": expected_records},
                                indent=2) + "\n", encoding="utf-8")
    print(f"Verified four timestamped release/reduction records, four barriers and {len(entries) // 8} segments.")
    print(f"Saved {report}; timestamps use the explicit synthetic replay clock.")


if __name__ == "__main__":
    main()
