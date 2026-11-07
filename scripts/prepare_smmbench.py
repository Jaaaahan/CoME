#!/usr/bin/env python3
"""Convert SMMBench into cluster-scoped, raw multimodal memory bundles.

Only reference labels are retained from supplied captions. Visual descriptions
are generated independently by the observation model. QA evidence stays private.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from mm_memory_bench.core import BundleWriter, content_asset, content_text, stable_id, validate_bundle


def convert(raw: Path, output: Path) -> dict:
    raw, output = raw.resolve(), output.resolve()
    clusters = sorted((raw / "Samples").glob("cluster_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not clusters:
        raise ValueError(f"No SMMBench clusters in {raw}")
    assets = set()
    manifest = {"benchmark": "smmbench", "source": str(raw),
                "converter_protocol": "smmbench-raw-1", "context_policy": "one cluster, all conversations",
                "caption_policy": "reference label only; descriptions regenerated from images",
                "question_policy": "open-ended; native QA retained in private metadata"}
    with BundleWriter(output, manifest) as writer:
        def image(name: str) -> dict:
            path = (raw / "Images" / name).resolve()
            if not path.is_relative_to(raw / "Images") or not path.is_file():
                raise ValueError(f"Missing or invalid image: {name}")
            aid = stable_id("smmbench", "image", name)
            if aid not in assets:
                writer.add_asset({"asset_id": aid, "media_type": "image",
                                  "path": os.path.relpath(path, output)})
                assets.add(aid)
            return content_asset("image", aid)

        for cluster in clusters:
            cid = stable_id("smmbench", cluster.name)
            writer.add_context({"context_id": cid, "benchmark": "smmbench", "title": cluster.name})
            rows = []
            for file in sorted(cluster.glob("*.json")):
                if file.name == "QA_sample.json":
                    continue
                for index, row in enumerate(json.loads(file.read_text())["conversation"]):
                    rows.append((str(row["timestamp"]), file.stem, index, row, file))
            for sequence, (timestamp, session, index, row, file) in enumerate(sorted(rows, key=lambda x: x[:3])):
                parts = []
                caption = row.get("caption", "")
                label = caption.splitlines()[0] if caption else ""
                if re.fullmatch(r"(?:Fig|Doc)\.\s+\S+", label):
                    parts.append(content_text(label))
                kind = row["content_type"]
                if kind == "text":
                    parts.append(content_text(row["content"]))
                elif kind == "image":
                    if row["content"]:
                        parts.append(content_text(row["content"]))
                    parts.append(image(row["image_path"]))
                elif kind == "json_evidence":
                    for part in row["content"]:
                        if part["type"] == "text":
                            parts.append(content_text(part["content"]["text"]))
                        elif part["type"] == "image":
                            parts.append(image(part["content"]["image_path"]))
                        else:
                            raise ValueError(f"Unknown document part: {part['type']}")
                else:
                    raise ValueError(f"Unknown message type: {kind}")
                writer.add_memory({"memory_id": stable_id(cid, session, index), "context_id": cid,
                    "kind": "dialogue_message", "sequence": sequence, "session_id": session,
                    "timestamp": timestamp, "speaker": row.get("sender_name", ""),
                    "role": "assistant" if row.get("sender_name") == "assistant" else "user",
                    "content": parts, "metadata": {"native_turn_index": index},
                    "provenance": {"source_file": str(file.relative_to(raw)),
                                   "json_pointer": f"/conversation/{index}"}})
            qa = json.loads((cluster / "QA_sample.json").read_text())
            if not qa:
                raise ValueError(f"Cluster has no questions and would be skipped by indexer: {cluster}")
            for question in qa:
                writer.add_question({"question_id": stable_id(cid, question["id"]), "context_id": cid,
                    "prompt": [content_text(question["question"])],
                    "answer": {"text": str(question["answer"])}, "memory_scope": {"mode": "all"},
                    "task": {"response_type": "text", "category": question.get("domain", ""),
                             "subcategory": question.get("category", "")},
                    "metadata": {"native_qa": question}})
    report = validate_bundle(output, check_assets=True)
    if report["missing_asset_count"]:
        raise ValueError(report)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.raw, args.output), indent=2))
