"""Enumera famílias disponíveis, saturadas e shifted em sete coordenadas."""

import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path


def elements(mask, r):
    return [i + 1 for i in range(r) if mask & (1 << i)]


def mask_key(mask, r):
    return mask.bit_count(), tuple(elements(mask, r))


def masks_of_size(r, k):
    return [
        sum(1 << i for i in positions)
        for positions in combinations(range(r), k)
    ]


def active_masks(r):
    return [
        mask
        for k in range(2, r - 1)
        for mask in masks_of_size(r, k)
    ]


def complement_pairs(r):
    universe = (1 << r) - 1
    pairs = []
    for mask in active_masks(r):
        other = universe ^ mask
        if mask_key(mask, r) < mask_key(other, r):
            pairs.append((mask, other))
    return sorted(pairs, key=lambda pair: mask_key(pair[0], r))


def mandatory_family(r):
    return {0} | {1 << i for i in range(r)}


def submasks(mask):
    current = mask
    while True:
        yield current
        if current == 0:
            return
        current = (current - 1) & mask


def shifted_targets(mask, r):
    for j in range(r):
        if mask & (1 << j):
            for i in range(j):
                if not mask & (1 << i):
                    yield (mask ^ (1 << j)) | (1 << i)


def validate_family(family, r):
    universe = (1 << r) - 1

    if not mandatory_family(r) <= family:
        raise AssertionError("Faltam conjuntos obrigatórios.")
    if any(not 0 <= x <= universe for x in family):
        raise AssertionError("Máscara fora do universo.")
    if any(y not in family for x in family for y in submasks(x)):
        raise AssertionError("A família não é um downset.")
    if any(
        (x in family) == ((universe ^ x) in family)
        for x in range(universe + 1)
    ):
        raise AssertionError("A família não é saturada.")
    if any(y not in family for x in family for y in shifted_targets(x, r)):
        raise AssertionError("A família não é shifted.")

    # Downset + saturação impedem uniões completas: se x | y fosse
    # o universo, o complemento de x estaria contido em y e também
    # pertenceria à família.


def family_record(family, r):
    counts = Counter(x.bit_count() for x in family)
    maximals = sorted(
        (
            x for x in family
            if not any(x != y and (x & y) == x for y in family)
        ),
        key=lambda x: mask_key(x, r),
    )
    return {
        "cardinality": len(family),
        "active_profile": [counts[k] for k in range(2, r - 1)],
        "masks": sorted(family),
        "maximals": [elements(x, r) for x in maximals],
    }


def enumerate_shifted_saturated(r=7):
    """Retorna as famílias em ordem de busca e as estatísticas da enumeração."""
    pairs = complement_pairs(r)
    active = active_masks(r)

    # O literal (p, v) indica que pairs[p][v] pertence à família.
    literals = {
        mask: (p, v)
        for p, pair in enumerate(pairs)
        for v, mask in enumerate(pair)
    }
    implications = {literal: set() for literal in literals.values()}

    def negate(literal):
        p, v = literal
        return p, 1 - v

    def add_implication(source, target):
        implications[source].add(target)
        implications[negate(target)].add(negate(source))

    for x in active:
        # Relações de cobertura entre conjuntos ativos.
        if x.bit_count() >= 3:
            for i in range(r):
                bit = 1 << i
                if x & bit:
                    add_implication(literals[x], literals[x ^ bit])

        # Deslocamentos para coordenadas de menor índice.
        for y in shifted_targets(x, r):
            add_implication(literals[x], literals[y])

    implications = {
        literal: tuple(sorted(targets))
        for literal, targets in implications.items()
    }

    assignment = [-1] * len(pairs)
    families = []
    stats = Counter()
    started = time.perf_counter()

    def propagate(literal, trail):
        stack = [literal]
        while stack:
            p, v = stack.pop()
            if assignment[p] != -1:
                if assignment[p] != v:
                    return False
                continue

            assignment[p] = v
            trail.append(p)
            stack.extend(implications[p, v])

        return True

    def choose_variable():
        return max(
            (p for p, v in enumerate(assignment) if v == -1),
            key=lambda p: (
                len(implications[p, 0]) + len(implications[p, 1]),
                pairs[p][0].bit_count(),
                -p,
            ),
            default=None,
        )

    def search():
        stats["nodes"] += 1
        p = choose_variable()

        if p is None:
            family = mandatory_family(r) | {
                pairs[q][v] for q, v in enumerate(assignment)
            }
            validate_family(family, r)
            families.append(tuple(sorted(family)))
            stats["solutions"] += 1
            return

        for v in (0, 1):
            trail = []
            if propagate((p, v), trail):
                search()
            else:
                stats["contradictions"] += 1

            for q in reversed(trail):
                assignment[q] = -1

    search()

    if len(families) != len(set(families)):
        raise AssertionError("A enumeração produziu duplicatas.")

    return {
        "families": families,
        "stats": dict(stats),
        "seconds": time.perf_counter() - started,
        "implication_arcs": sum(map(len, implications.values())),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("shifted_candidates_r7.json"),
        help="Caminho do catálogo JSON (padrão: %(default)s).",
    )
    parser.add_argument(
        "--expected-shifted",
        type=int,
        default=134,
        help="Contagem esperada de famílias (padrão: %(default)s).",
    )
    args = parser.parse_args(argv)

    result = enumerate_shifted_saturated(r=7)
    families = result["families"]
    count = len(families)

    if count != args.expected_shifted:
        parser.exit(
            1,
            f"Esperadas {args.expected_shifted} famílias; encontradas {count}. "
            "O arquivo de saída não foi alterado.\n",
        )

    data = {
        "r": 7,
        "count": count,
        "stats": result["stats"],
        "seconds": result["seconds"],
        "implication_arcs": result["implication_arcs"],
        "candidates": [
            {"id": i} | family_record(family, 7)
            for i, family in enumerate(families)
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Famílias enumeradas: {count}")
    print(f"Nós: {result['stats']['nodes']}")
    print(f"Contradições: {result['stats'].get('contradictions', 0)}")
    print(f"Arcos de implicação: {result['implication_arcs']}")
    print(f"Tempo da enumeração: {result['seconds']:.6f} s")
    print(f"Arquivo gravado em: {args.output.resolve()}")


if __name__ == "__main__":
    main()
