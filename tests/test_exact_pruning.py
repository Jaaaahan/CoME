"""Check exact pruning against exhaustive forest search and larger unit budgets."""
import itertools
import math
import random

import pytest

from scripts.evaluate_graphmemix_exact_pruning import exact_subset


def exhaustive(ids, prizes, edges, root_cost=.12, node_cost=0., edge_weight=.1):
    # Independent tiny-graph oracle: enumerate both node sets AND edge sets,
    # rejecting cycles. This does not rely on the production Kruskal routine.
    best = (-math.inf, 0, ())
    kept = []
    for count in range(1, len(ids) + 1):
        for members in itertools.combinations(range(len(ids)), count):
            nodes = {ids[i] for i in members}
            local = [e for e in edges if e[1] in nodes and e[2] in nodes]
            for mask in range(1 << len(local)):
                groups = {mid: {mid} for mid in nodes}
                cost, chosen, cycle = 0., 0, False
                for j, (similarity, left, right, _) in enumerate(local):
                    if not mask >> j & 1:
                        continue
                    if groups[left] is groups[right]:
                        cycle = True
                        break
                    merged = groups[left] | groups[right]
                    for mid in merged:
                        groups[mid] = merged
                    cost += edge_weight * -math.log(max(similarity, 1e-6))
                    chosen += 1
                if cycle:
                    continue
                value = sum(prizes[mid] - node_cost for mid in nodes) - root_cost * (count - chosen) - cost
                key = (round(value, 12), -count, tuple(-i for i in members))
                if key > best:
                    best, kept = key, [ids[i] for i in members]
    return kept, best[0]


def test_pruning_matches_exhaustive_forests_with_bridges_cycles_and_isolates():
    rng = random.Random(2710)
    ids = ['a', 'b', 'c', 'd', 'e']
    for _ in range(50):
        prizes = {mid: rng.choice([.01, .06, .12, .3, .8]) for mid in ids}
        edges = [(rng.choice([.2, .6, .99]), left, right, 'new_fact')
                 for left, right in itertools.combinations(ids, 2) if rng.random() < .3]
        expected, objective = exhaustive(ids, prizes, edges)
        actual, info = exact_subset(ids, prizes, edges, edge_weight=.1, root_cost=.12, node_cost=0.)
        assert actual == expected
        assert info['objective_value'] == pytest.approx(objective)


def test_twenty_unit_pruning_avoids_enumerating_positive_nodes():
    ids = [f'u{i}' for i in range(20)]
    edges = [(.99, ids[i], ids[i+1], 'new_fact') for i in range(19)]
    selected, info = exact_subset(ids, dict.fromkeys(ids, .8), edges,
                                  edge_weight=.1, root_cost=.12, node_cost=0.)
    assert selected == ids and info['chosen_edges'] == 19
    assert info['exact'] and info['evaluated_subsets'] == 1


@pytest.mark.parametrize('prize', [.01, .12])
def test_twenty_edgeless_units_keep_earliest_best_when_none_have_positive_net_value(prize):
    ids = [f'u{i}' for i in range(20)]
    selected, info = exact_subset(ids, dict.fromkeys(ids, prize), [],
                                  edge_weight=.1, root_cost=.12, node_cost=0.)
    assert selected == ['u0'] and info['evaluated_subsets'] == 1
