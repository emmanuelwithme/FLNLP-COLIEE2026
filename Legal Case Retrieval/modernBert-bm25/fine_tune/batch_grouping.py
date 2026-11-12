"""Pure-Python distinct-query grouping; no model/GPU state is touched.

Shuffle all pairs, then repair query collisions by whole-row swaps. A repair
strictly reduces the total number of duplicate query slots, so it terminates.
Local repair is approximately randomized, not uniform over all valid partitions.
"""
from collections import Counter, defaultdict, deque
import heapq
import random

from config import digest

SHUFFLE_REPAIR = "shuffle_repair_v1"
LEGACY_DEGREE = "legacy_degree_v1"


def _repair_distinct_queries(rows, rng, random_attempts=64):
    counts = [Counter(r["query_id"] for r in rows[i:i + 4]) for i in range(0, len(rows), 4)]
    bad = deque(i for i, c in enumerate(counts) if len(c) < 4)
    queued = set(bad)
    initial_slots = sum(4 - len(c) for c in counts)
    stats = {"initial_conflict_batches": len(bad), "initial_duplicate_slots": initial_slots,
             "repair_swaps": 0, "random_proposals": 0, "fallback_searches": 0}

    def legal(i, old, j):
        target = j // 4
        if target == i or old in counts[target]:
            return False
        candidate = rows[j]["query_id"]
        # If candidate already exists in the source, removing a duplicated
        # candidate from the target still improves the GLOBAL conflict count.
        return candidate not in counts[i] or counts[target][candidate] > 1

    def replace(counter, old, new):
        counter[old] -= 1
        if counter[old] == 0:
            del counter[old]
        counter[new] += 1

    while bad:
        i = bad.popleft()
        queued.remove(i)
        if len(counts[i]) == 4:
            continue
        old = rng.choice(sorted(q for q, n in counts[i].items() if n > 1))
        position = rng.choice([j for j in range(i * 4, i * 4 + 4) if rows[j]["query_id"] == old])
        destination = None
        for _ in range(random_attempts):
            stats["random_proposals"] += 1
            j = rng.randrange(len(rows))
            if legal(i, old, j):
                destination = j
                break
        if destination is None:
            stats["fallback_searches"] += 1
            start = rng.randrange(len(counts))
            for offset in range(len(counts)):
                target = (start + offset) % len(counts)
                for slot in range(4):
                    j = target * 4 + slot
                    if legal(i, old, j):
                        destination = j
                        break
                if destination is not None:
                    break
        if destination is None:
            raise AssertionError("No improving query swap despite feasible query counts")
        target = destination // 4
        candidate = rows[destination]["query_id"]
        before = 8 - len(counts[i]) - len(counts[target])
        rows[position], rows[destination] = rows[destination], rows[position]
        replace(counts[i], old, candidate)
        replace(counts[target], candidate, old)
        after = 8 - len(counts[i]) - len(counts[target])
        if not after < before:
            raise AssertionError("Repair must strictly reduce duplicate query slots")
        stats["repair_swaps"] += 1
        if stats["repair_swaps"] > initial_slots:
            raise AssertionError("Query repair exceeded its monotone termination bound")
        for index in (i, target):
            if len(counts[index]) < 4 and index not in queued:
                bad.append(index)
                queued.add(index)
    if any(len(c) != 4 for c in counts):
        raise AssertionError("Unrepaired query collision")
    stats["remaining_duplicate_slots"] = 0
    return stats


def shuffle_repair_rows(original, epoch, seed=289):
    """Preserve every original row, add <=3 audited rows, shuffle and repair.

    Feasibility: no query may occur more often than the final batch count.
    Padding prefers distinct queries; near-capacity artificial inputs may need
    multiple padding rows from the same unsaturated query. No original is lost.
    """
    if type(epoch) is not int or epoch < 0 or type(seed) is not int or seed < 0:
        raise ValueError("epoch and seed must be nonnegative integers")
    if not original:
        raise ValueError("Cannot group an empty training dataset")
    canonical = sorted(original, key=lambda row: (row["query_id"], row["positive_id"]))
    by_query = defaultdict(list)
    for row in canonical:
        by_query[row["query_id"]].append(row)
    if len(by_query) < 4:
        raise ValueError("Need at least 4 distinct training queries, including debug")
    batches = (len(canonical) + 3) // 4
    frequencies = Counter({q: len(v) for q, v in by_query.items()})
    if max(frequencies.values()) > batches:
        raise ValueError("Query degree too high to preserve all pairs in distinct-query batches")
    padding_rng = random.Random(int(digest([seed, epoch, SHUFFLE_REPAIR, "padding"]), 16))
    shuffle_seed = digest([seed, epoch, SHUFFLE_REPAIR, "shuffle"])
    repair_seed = digest([seed, epoch, SHUFFLE_REPAIR, "repair"])
    result = list(canonical)
    padding = []
    used_padding_queries = set()
    for _ in range(batches * 4 - len(result)):
        eligible = [q for q in sorted(by_query) if frequencies[q] < batches]
        preferred = [q for q in eligible if q not in used_padding_queries]
        q = padding_rng.choice(preferred or eligible)
        row = dict(padding_rng.choice(by_query[q]))
        result.append(row)
        frequencies[q] += 1
        used_padding_queries.add(q)
        padding.append({"query_id": q, "positive_id": row["positive_id"]})
    expected_objects = Counter(map(id, result))
    random.Random(int(shuffle_seed, 16)).shuffle(result)
    repair = _repair_distinct_queries(result, random.Random(int(repair_seed, 16)))
    if Counter(map(id, result)) != expected_objects:
        raise AssertionError("Grouping lost, duplicated or replaced a whole training row")
    if Counter(r["query_id"] for r in result) != frequencies:
        raise AssertionError("Grouping changed query multiplicities")
    groups = [sorted(r["query_id"] for r in result[i:i + 4]) for i in range(0, len(result), 4)]
    return result, {"original_pairs": len(original), "padded_pairs": len(padding),
        "padding_pairs": padding, "padding_distinct_queries": len(used_padding_queries),
        "logical_batches": batches, "policy": SHUFFLE_REPAIR, "grouping_hash": digest(groups),
        "shuffle_seed": shuffle_seed, "repair_seed": repair_seed, **repair}


def legacy_degree_rows(original, epoch, seed=289):
    """Exact v2 policy retained for old runs and reproducible comparisons."""
    by_query = defaultdict(list)
    for row in original:
        by_query[row["query_id"]].append(row)
    if len(by_query) < 4:
        raise ValueError("Need at least 4 distinct training queries, including debug")
    rng = random.Random(int(digest([seed, epoch, "logical_order"]), 16))
    for qid in sorted(by_query):
        rng.shuffle(by_query[qid])
    tie = {q: rng.random() for q in sorted(by_query)}
    heap = [(-len(values), tie[q], q) for q, values in by_query.items()]
    heapq.heapify(heap)
    if max(map(len, by_query.values())) > (len(original) + 3) // 4:
        raise ValueError("Query degree too high to preserve all pairs in distinct-query batches")
    available = {q: list(v) for q, v in by_query.items()}
    result, padding = [], []
    while heap:
        chosen = [heapq.heappop(heap) for _ in range(min(4, len(heap)))]
        batch = [by_query[q].pop() for _, _, q in chosen]
        for _, priority, q in chosen:
            if by_query[q]:
                heapq.heappush(heap, (-len(by_query[q]), priority, q))
        if len(batch) < 4:
            if heap:
                raise AssertionError("Internal packing error: padding before last batch")
            other = sorted(set(available) - {r["query_id"] for r in batch})
            rng.shuffle(other)
            for q in other[:4 - len(batch)]:
                row = dict(available[q][rng.randrange(len(available[q]))])
                batch.append(row)
                padding.append({"query_id": row["query_id"], "positive_id": row["positive_id"]})
        result.extend(batch)
    if len(result) != ((len(original) + 3) // 4) * 4:
        raise AssertionError("Unexpected logical epoch size")
    return result, {"original_pairs": len(original), "padded_pairs": len(padding),
                    "padding_pairs": padding, "logical_batches": len(result) // 4,
                    "policy": "all original positive pairs; max-remaining-query packing; tail padding"}
