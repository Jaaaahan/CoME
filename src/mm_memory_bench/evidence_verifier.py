"""Shared GraphMemix ECV response validation."""

from __future__ import annotations

import json
from typing import Any, Mapping


POSITIVE_ROLES = {"new_fact", "clarification", "corroboration"}
ALL_ROLES = POSITIVE_ROLES | {"redundant", "conflict", "irrelevant"}
ECV_PARSER_PROTOCOL = "graphmemix-ecv-structured-normalize-2"


def direct_support_value(row: Mapping[str, Any], *, edge_only: bool) -> float:
    """Enforce the edge-only contract in code, not merely in the prompt."""
    if edge_only:
        return 0.0
    try:
        return max(0.0, min(5.0, float(row.get("direct_support", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def validate_candidate_rows(
    value: Any,
    *,
    expected_aliases: set[str],
    eligible_anchors: Mapping[str, set[str]],
) -> list[Mapping[str, Any]]:
    """Reject incomplete, duplicate, or schema-invalid structured ECV rows."""
    if not isinstance(value, list):
        raise ValueError("ECV candidates must be a list")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    ignored_unknown_rows = 0
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("ECV candidate row must be an object")
        alias = str(row.get("id", ""))
        if alias not in expected_aliases:
            ignored_unknown_rows += 1
            continue
        if alias in seen:
            raise ValueError(f"ECV returned duplicate candidate id: {alias!r}")
        seen.add(alias)
        role = str(row.get("role", "")).lower()
        invalid_role = role not in ALL_ROLES
        if invalid_role:
            role = "irrelevant"
        try:
            direct = float(row.get("direct_support", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ECV returned invalid direct support for {alias}") from exc
        if not 0.0 <= direct <= 5.0:
            raise ValueError(f"ECV direct support out of range for {alias}")
        anchor = row.get("best_anchor_id")
        ineligible_anchor = (
            anchor is not None
            and str(anchor) not in eligible_anchors.get(alias, set())
        )
        try:
            incremental = float(row.get("incremental_support", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ECV returned invalid incremental support for {alias}") from exc
        if not 0.0 <= incremental <= 5.0:
            raise ValueError(f"ECV incremental support out of range for {alias}")
        normalized = dict(row)
        normalizations: list[str] = []
        if invalid_role:
            normalized["role"] = "irrelevant"
            normalizations.append("invalid_role_replaced_with_irrelevant")
        if ineligible_anchor:
            normalized["best_anchor_id"] = None
            normalized["incremental_support"] = 0.0
            normalized["role"] = "irrelevant"
            normalizations.append("ineligible_anchor_cleared_edge_fields")
        elif anchor is None and (incremental != 0.0 or role != "irrelevant"):
            normalized["incremental_support"] = 0.0
            normalized["role"] = "irrelevant"
            normalizations.append("null_anchor_cleared_edge_fields")
        elif role in POSITIVE_ROLES and incremental <= 0.0:
            normalized["role"] = "irrelevant"
            normalizations.append("zero_incremental_cleared_positive_role")
        elif role not in POSITIVE_ROLES and incremental > 0.0:
            normalized["incremental_support"] = 0.0
            normalizations.append("nonpositive_role_cleared_incremental")
        if normalizations:
            normalized["_normalizations"] = normalizations
        rows.append(normalized)
    if seen != expected_aliases:
        missing = sorted(expected_aliases - seen)
        raise ValueError(f"ECV omitted {len(missing)} candidate rows: {missing[:3]}")
    if ignored_unknown_rows and rows:
        first = dict(rows[0])
        first["_normalizations"] = [
            *map(str, first.get("_normalizations", [])),
            *(["ignored_unknown_candidate_row"] * ignored_unknown_rows),
        ]
        rows[0] = first
    return rows


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise
        value = json.loads(text[left:right + 1])
    if not isinstance(value, dict):
        raise ValueError("verifier returned non-object JSON")
    return value


