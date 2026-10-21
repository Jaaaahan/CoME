"""Summarize CoME cost and independently annotated evidence preservation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def summarize(predictions: list[dict], annotations: list[dict] | None = None) -> dict:
    by_id = {str(row["question_id"]): row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    metadata = [row["metadata"] for row in predictions]
    result = {"questions": len(predictions)}
    for field in ("reader_input_tokens", "reader_completion_tokens", "work_calls", "work_tokens",
                  "latency_seconds", "new_observations"):
        values = [m[field] for m in metadata if isinstance(m.get(field), (float, int))]
        result[f"mean_{field}"] = mean(values) if values else None
    closed = 0
    for row in metadata:
        selected = {n["node_id"] for n in row["selected_nodes"]}
        closed += all(set(supports) <= selected for supports in row["requires"].values())
    result["structural_dependency_closure_rate"] = closed / len(metadata) if metadata else None
    result["unresolved_question_rate"] = mean(bool(m["unresolved"]) for m in metadata) if metadata else None
    # These groups must come from an independent review, never from the system's
    # own generated requires map. Each group represents all supports for one fact.
    if annotations is not None:
        total, retained = 0, 0
        seen = set()
        for row in annotations:
            qid = str(row["question_id"])
            if qid not in by_id or qid in seen:
                raise ValueError("annotation IDs must be unique and have predictions")
            seen.add(qid)
            selected = {n["node_id"] for n in by_id[qid]["metadata"]["selected_nodes"]}
            for group in row["required_supports"]:
                if not isinstance(group, list) or not group or any(not isinstance(n, str) for n in group):
                    raise ValueError("required_supports needs nonempty lists of node IDs")
                total += 1
                retained += set(group) <= selected
        result.update(annotated_support_groups=total, retained_support_groups=retained,
                      complete_evidence_retention_rate=retained / total if total else None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def rows(path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    result = summarize(rows(args.predictions), rows(args.annotations) if args.annotations else None)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
