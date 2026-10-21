from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..evidence_forest import solve_forest_greedy
from .types import Candidate, Claim, CoMEConfig, Verification


@dataclass
class Selection:
    node_ids: list[str]
    claims: list[Claim]
    tokens: int
    diagnostics: dict = field(default_factory=dict)


def closure(seeds: Iterable[str], requires: dict[str, frozenset[str]]) -> set[str]:
    result, pending = set(seeds), list(seeds)
    while pending:
        for dependency in requires.get(pending.pop(), ()):
            if dependency not in result:
                result.add(dependency)
                pending.append(dependency)
    return result


def remove_dependents(selected: set[str], removed: str,
                      requires: dict[str, frozenset[str]]) -> set[str]:
    remaining = selected - {removed}
    while True:
        invalid = {nid for nid in remaining if not requires.get(nid, frozenset()) <= remaining}
        if not invalid:
            return remaining
        remaining -= invalid


def active_claims(selected: set[str], claims: list[Claim]) -> list[Claim]:
    unique: dict[str, Claim] = {}
    for claim in claims:
        if claim.anchor_id in selected and claim.supports <= selected:
            if claim.fact_key not in unique or claim.utility > unique[claim.fact_key].utility:
                unique[claim.fact_key] = claim
    return sorted(unique.values(), key=lambda c: c.id)


def adjacency_for(verification: Verification, selected: set[str] | None = None) -> dict:
    adjacency: dict[str, list[tuple[str, float, str]]] = {}
    for edge in verification.edges:
        if edge.role not in {"new_fact", "clarification", "corroboration"}:
            continue
        if selected is not None and not ({edge.left, edge.right} | edge.supports) <= selected:
            continue
        adjacency.setdefault(edge.left, []).append((edge.right, edge.similarity, edge.role))
        adjacency.setdefault(edge.right, []).append((edge.left, edge.similarity, edge.role))
    return adjacency


def select_evidence(nodes: dict[str, Candidate], verification: Verification,
                    cost: Callable[[set[str], list[Claim]], int], budget: int,
                    config: CoMEConfig) -> Selection:
    """GraphMemix proposal followed by dependency-closed marginal-cost packing."""
    requires = {nid: nodes[nid].requires | verification.requires.get(nid, frozenset()) for nid in nodes}
    if not config.use_dependencies:
        requires = {nid: frozenset() for nid in nodes}
    claims = verification.claims
    available = set(nodes)
    valid = {nid for nid in nodes if closure([nid], requires) <= available}
    prizes = {nid: max(0.0, verification.prizes.get(nid, 0.0)) for nid in valid}
    proposal_prizes = dict(prizes)
    for claim in claims:
        if claim.anchor_id in valid and closure(claim.supports, requires) <= valid:
            proposal_prizes[claim.anchor_id] += claim.utility
    ids = sorted(valid)
    proposal, forest_info = solve_forest_greedy(ids, proposal_prizes, adjacency_for(verification),
        k=min(config.proposal_k, len(ids)), edge_weight=config.edge_weight, root_cost=config.root_cost)

    def bundle(seed: str) -> set[str]:
        expanded = closure([seed], requires)
        for claim in claims:
            if claim.anchor_id == seed and closure(claim.supports, requires) <= valid:
                expanded |= closure(claim.supports, requires)
        return expanded

    def utility(selected: set[str]) -> float:
        facts: dict[str, float] = {}
        for nid in selected:
            key = nodes[nid].fact_key or nid
            facts[key] = max(facts.get(key, 0.0), prizes.get(nid, 0.0))
        for claim in active_claims(selected, claims):
            facts[claim.fact_key] = max(facts.get(claim.fact_key, 0.0), claim.utility)
        # Reuse Kruskal inside the existing solver for each fixed selected set.
        _, info = solve_forest_greedy(sorted(selected), {nid: 0.0 for nid in selected},
            adjacency_for(verification, selected), k=len(selected),
            edge_weight=config.edge_weight, root_cost=config.root_cost)
        return sum(facts.values()) - info["objective"]

    selected: set[str] = set()
    for nid in proposal:
        selected |= bundle(nid)
    selected &= valid
    deleted = []
    # The reused proposal is fixed-K. Restore variable cardinality before packing
    # so irrelevant isolated nodes cannot survive merely to fill a slot.
    while selected:
        before_utility = utility(selected)
        improvements = []
        for nid in sorted(selected):
            remaining = remove_dependents(selected, nid, requires)
            gain = utility(remaining) - before_utility
            if gain > 1e-12:
                improvements.append((gain, nid, remaining))
        if not improvements:
            break
        _, removed, remaining = max(improvements, key=lambda row: (row[0], row[1]))
        deleted.append({"trigger": removed, "removed": sorted(selected - remaining), "reason": "utility"})
        selected = remaining
    while cost(selected, active_claims(selected, claims)) > budget:
        if not selected:
            raise ValueError("question/instructions alone exceed reader input budget")
        before_tokens = cost(selected, active_claims(selected, claims))
        before_utility = utility(selected)
        choices = []
        for nid in sorted(selected):
            remaining = remove_dependents(selected, nid, requires)
            saved = before_tokens - cost(remaining, active_claims(remaining, claims))
            if saved > 0:
                choices.append(((before_utility - utility(remaining)) / saved, nid, remaining))
        if not choices:
            # Shared media or integer rounding can make all singleton savings zero.
            selected = set()
            break
        _, removed, remaining = min(choices, key=lambda row: (row[0], row[1]))
        deleted.append({"trigger": removed, "removed": sorted(selected - remaining)})
        selected = remaining
    # Refill with complete support bundles. Every accepted move raises utility.
    while True:
        before_tokens = cost(selected, active_claims(selected, claims))
        before_utility = utility(selected)
        choices = []
        for nid in ids:
            expanded = selected | bundle(nid)
            if expanded == selected:
                continue
            tokens = cost(expanded, active_claims(expanded, claims))
            gain = utility(expanded) - before_utility
            if tokens <= budget and gain > 1e-12:
                choices.append((gain / max(1, tokens - before_tokens), nid, expanded))
        if not choices:
            break
        _, _, selected = max(choices, key=lambda row: (row[0], row[1]))
    if any(not requires[nid] <= selected for nid in selected):
        raise AssertionError("selection has dangling dependencies")
    ordered, final_info = solve_forest_greedy(sorted(selected), prizes,
        adjacency_for(verification, selected), k=len(selected),
        edge_weight=config.edge_weight, root_cost=config.root_cost)
    supported = active_claims(selected, claims)
    tokens = cost(selected, supported)
    if tokens > budget:
        raise ValueError("question/instructions alone exceed reader input budget")
    return Selection(ordered, supported, tokens, {"proposal": forest_info,
        "forest": final_info, "pruned": deleted, "invalid_dependencies": sorted(available - valid),
        "selected_claims": [c.id for c in supported], "dependency_closed": True})
