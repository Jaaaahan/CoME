"""Question visibility controls; never part of model question text."""
from typing import Iterable, Mapping, Any


def visible_ids(question: Mapping[str, Any], memories: Iterable[Mapping[str, Any]]) -> set[str]:
    context = question.get("context_id")
    memories = [m for m in memories if context is None or m.get("context_id") == context]
    all_ids = {str(m["memory_id"]) for m in memories}
    scope = question.get("memory_scope", {"mode": "all"})
    mode = scope.get("mode", "all")
    if mode == "all":
        return all_ids
    if mode == "prefix":
        cutoff = scope.get("max_sequence")
        if type(cutoff) is not int:
            raise ValueError("prefix memory scope needs max_sequence")
        return {str(m["memory_id"]) for m in memories if int(m["sequence"]) <= cutoff}
    if mode == "ids":
        ids = set(map(str, scope.get("memory_ids", [])))
        if not ids <= all_ids:
            raise ValueError("question references an unavailable memory")
        return ids
    raise ValueError("unknown memory scope")
