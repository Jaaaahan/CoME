"""Shared GraphMemix forest solvers; independent of benchmark scripts."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def solve_forest(
    candidate_ids: Sequence[str], prizes: Mapping[str, float],
    adjacency: Mapping[str, Sequence[tuple[str, float, str]]], *,
    k: int, edge_weight: float, root_cost: float,
) -> tuple[list[str], dict[str, Any]]:
    # SciPy is only required by the optional MILP solver.  Keeping the import
    # local lets the deterministic greedy fixed-K solver run in lightweight
    # inference containers without changing either solver's mathematics.
    from scipy import sparse
    from scipy.optimize import Bounds, LinearConstraint, milp

    ids = list(dict.fromkeys(map(str, candidate_ids)))
    if len(ids) < k:
        raise ValueError("fewer candidates than requested evidence slots")
    node = {memory_id: index for index, memory_id in enumerate(ids)}
    undirected: dict[tuple[int, int], tuple[float, str]] = {}
    for left_id in ids:
        for right_id, similarity, relation in adjacency.get(left_id, []):
            if right_id not in node or right_id == left_id:
                continue
            key = tuple(sorted((node[left_id], node[right_id])))
            current = undirected.get(key)
            if current is None or similarity > current[0]:
                undirected[key] = (float(similarity), relation)
    arcs: list[tuple[int, int, float, str]] = []
    for (left, right), (similarity, relation) in sorted(undirected.items()):
        cost = -math.log(max(similarity, 1e-6))
        arcs.extend(((left, right, cost, relation), (right, left, cost, relation)))

    n, m = len(ids), len(arcs)
    # Variables: selected x[n], root arcs r[n], directed memory arcs y[m],
    # continuous root flows fr[n], and continuous memory-arc flows f[m].
    x0, r0, y0, fr0, f0 = 0, n, 2 * n, 2 * n + m, 3 * n + m
    total = 3 * n + 2 * m
    objective = np.zeros(total, dtype=np.float64)
    for memory_id, index in node.items():
        objective[x0 + index] = -float(prizes[memory_id])
        objective[r0 + index] = root_cost
    for index, (_, _, cost, _) in enumerate(arcs):
        objective[y0 + index] = edge_weight * cost

    rows: list[int] = []; cols: list[int] = []; values: list[float] = []
    lower: list[float] = []; upper: list[float] = []
    row = 0
    for index in range(n):
        rows.append(row); cols.append(x0 + index); values.append(1.0)
    lower.append(float(k)); upper.append(float(k)); row += 1

    incoming: dict[int, list[int]] = defaultdict(list)
    for arc_index, (_, child, _, _) in enumerate(arcs):
        incoming[child].append(arc_index)
    for index in range(n):
        # Every selected node has exactly one parent: root or another selected node.
        rows.extend((row, row)); cols.extend((r0 + index, x0 + index)); values.extend((1.0, -1.0))
        for arc_index in incoming.get(index, []):
            rows.append(row); cols.append(y0 + arc_index); values.append(1.0)
        lower.append(0.0); upper.append(0.0); row += 1
        # Root flow is available only when this root arc is selected.
        rows.extend((row, row)); cols.extend((fr0 + index, r0 + index)); values.extend((1.0, -float(k)))
        lower.append(-np.inf); upper.append(0.0); row += 1
        # Single-commodity conservation: each selected memory consumes one unit.
        rows.extend((row, row)); cols.extend((fr0 + index, x0 + index)); values.extend((1.0, -1.0))
        for arc_index, (parent, child, _, _) in enumerate(arcs):
            if child == index:
                rows.append(row); cols.append(f0 + arc_index); values.append(1.0)
            if parent == index:
                rows.append(row); cols.append(f0 + arc_index); values.append(-1.0)
        lower.append(0.0); upper.append(0.0); row += 1
    for arc_index, (parent, child, _, _) in enumerate(arcs):
        # Arc endpoints must both be selected.
        rows.extend((row, row)); cols.extend((y0 + arc_index, x0 + parent)); values.extend((1.0, -1.0))
        lower.append(-np.inf); upper.append(0.0); row += 1
        rows.extend((row, row)); cols.extend((y0 + arc_index, x0 + child)); values.extend((1.0, -1.0))
        lower.append(-np.inf); upper.append(0.0); row += 1
        # Memory-edge flow is available only on a selected directed arc.
        rows.extend((row, row)); cols.extend((f0 + arc_index, y0 + arc_index)); values.extend((1.0, -float(k)))
        lower.append(-np.inf); upper.append(0.0); row += 1

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(row, total))
    integrality = np.zeros(total, dtype=np.int8); integrality[:fr0] = 1
    lower_bounds = np.zeros(total); upper_bounds = np.full(total, float(k))
    upper_bounds[:fr0] = 1.0
    started = time.perf_counter()
    result = milp(
        c=objective, integrality=integrality, bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 5.0, "mip_rel_gap": 0.0},
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None:
        raise RuntimeError(f"GraphMemix MILP failed: {result.message}")
    selected = {index for index in range(n) if result.x[x0 + index] > 0.5}
    roots = [index for index in selected if result.x[r0 + index] > 0.5]
    chosen_edges: list[tuple[int, int, str]] = []
    for arc_index, (parent, child, _, relation) in enumerate(arcs):
        if result.x[y0 + arc_index] > 0.5:
            chosen_edges.append((parent, child, relation))
    key = lambda index: (-float(prizes[ids[index]]), ids[index])
    # Root-arc orientation is objective-equivalent in an undirected component
    # and may therefore be arbitrary.  Order each component from its highest
    # prize node instead of exposing a solver tie-break to the reader.
    chosen_undirected: dict[int, set[int]] = defaultdict(set)
    for parent, child, _ in chosen_edges:
        chosen_undirected[parent].add(child); chosen_undirected[child].add(parent)
    components: list[set[int]] = []
    unseen = set(selected)
    while unseen:
        start = min(unseen); component = {start}; queue = [start]; unseen.remove(start)
        while queue:
            current = queue.pop(0)
            for other in chosen_undirected.get(current, set()):
                if other in unseen:
                    unseen.remove(other); component.add(other); queue.append(other)
        components.append(component)
    ordered: list[int] = []
    for component in sorted(components, key=lambda values: key(min(values, key=key))):
        component_root = min(component, key=key)
        queue = [component_root]; visited = {component_root}
        while queue:
            current = queue.pop(0); ordered.append(current)
            neighbors = [value for value in chosen_undirected.get(current, set()) if value not in visited]
            for value in sorted(neighbors, key=key):
                visited.add(value); queue.append(value)
    if set(ordered) != selected:
        raise RuntimeError("selected forest traversal is incomplete")
    return [ids[index] for index in ordered], {
        "objective": float(result.fun), "candidate_nodes": n,
        "candidate_edges": len(undirected), "chosen_edges": len(chosen_edges),
        "explicit_chosen_edges": sum(relation == "explicit" for _, _, relation in chosen_edges),
        "semantic_chosen_edges": sum(relation == "semantic" for _, _, relation in chosen_edges),
        "components": len(roots), "solver_seconds": elapsed,
    }


def solve_forest_greedy(
    candidate_ids: Sequence[str], prizes: Mapping[str, float],
    adjacency: Mapping[str, Sequence[tuple[str, float, str]]], *,
    k: int, edge_weight: float, root_cost: float,
) -> tuple[list[str], dict[str, Any]]:
    """Deterministic 1-swap search; the forest for each node set is exact."""
    started = time.perf_counter()
    ids = list(dict.fromkeys(map(str, candidate_ids)))
    position = {memory_id: index for index, memory_id in enumerate(ids)}
    edges: dict[tuple[str, str], tuple[float, str]] = {}
    for left in ids:
        for right, similarity, relation in adjacency.get(left, []):
            if right not in position or right == left:
                continue
            key = tuple(sorted((left, right)))
            current = edges.get(key)
            if current is None or similarity > current[0]:
                edges[key] = (float(similarity), relation)

    def forest(selected: set[str]) -> tuple[float, list[tuple[str, str, str]]]:
        parent = {memory_id: memory_id for memory_id in selected}
        def find(value: str) -> str:
            while parent[value] != value:
                parent[value] = parent[parent[value]]; value = parent[value]
            return value
        weighted: list[tuple[float, str, str, str]] = []
        for (left, right), (similarity, relation) in edges.items():
            if left not in selected or right not in selected:
                continue
            benefit = root_cost - edge_weight * -math.log(max(similarity, 1e-6))
            if benefit > 0:
                weighted.append((benefit, left, right, relation))
        benefit_sum = 0.0; chosen: list[tuple[str, str, str]] = []
        for benefit, left, right, relation in sorted(weighted, key=lambda row: (-row[0], row[1], row[2])):
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                continue
            parent[right_root] = left_root; benefit_sum += benefit; chosen.append((left, right, relation))
        value = sum(float(prizes[mid]) for mid in selected) - len(selected) * root_cost + benefit_sum
        return value, chosen

    selected = set(sorted(ids, key=lambda mid: (-float(prizes[mid]), position[mid], mid))[:k])
    value, chosen = forest(selected); iterations = 0
    while True:
        best: tuple[float, str, str, list[tuple[str, str, str]]] | None = None
        for removed in sorted(selected):
            for added in ids:
                if added in selected:
                    continue
                proposal = set(selected); proposal.remove(removed); proposal.add(added)
                proposal_value, proposal_edges = forest(proposal)
                candidate = (proposal_value, removed, added, proposal_edges)
                if proposal_value > value + 1e-12 and (best is None or candidate[:3] > best[:3]):
                    best = candidate
        if best is None:
            break
        value, removed, added, chosen = best
        selected.remove(removed); selected.add(added); iterations += 1

    graph: dict[str, set[str]] = defaultdict(set)
    for left, right, _ in chosen:
        graph[left].add(right); graph[right].add(left)
    key = lambda mid: (-float(prizes[mid]), position[mid], mid)
    components: list[set[str]] = []
    unseen = set(selected)
    while unseen:
        start = min(unseen); unseen.remove(start); component = {start}; queue = [start]
        while queue:
            current = queue.pop(0)
            for other in graph.get(current, set()):
                if other in unseen:
                    unseen.remove(other); component.add(other); queue.append(other)
        components.append(component)
    ordered: list[str] = []
    for component in sorted(components, key=lambda values: key(min(values, key=key))):
        root = min(component, key=key); queue = [root]; visited = {root}
        while queue:
            current = queue.pop(0); ordered.append(current)
            for other in sorted((x for x in graph.get(current, set()) if x not in visited), key=key):
                visited.add(other); queue.append(other)
    return ordered, {
        "objective": -value, "candidate_nodes": len(ids), "candidate_edges": len(edges),
        "chosen_edges": len(chosen),
        "explicit_chosen_edges": sum(relation == "explicit" for _, _, relation in chosen),
        "semantic_chosen_edges": sum(relation == "semantic" for _, _, relation in chosen),
        "components": len(components), "solver_seconds": time.perf_counter() - started,
        "local_search_iterations": iterations, "exact": False,
    }


