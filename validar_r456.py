"""Compara as fórmulas de Hou–Hu com os MILPs original e reduzido."""

import argparse
import json
import math
import platform
import time
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from enumerar_r7 import (
    active_masks,
    complement_pairs,
    elements,
    family_record,
    mandatory_family,
    masks_of_size,
    submasks,
)

FULL_LIMITS = {4: 10, 5: 10, 6: 7}
FULL_COUNTS = {4: 495, 5: 1287, 6: 462}


def weight_table(n):
    a = tuple(ni - 1 for ni in n)
    weights = [1] * (1 << len(a))
    for x in range(1, len(weights)):
        bit = x & -x
        weights[x] = weights[x ^ bit] * a[bit.bit_length() - 1]
    return a, weights


def validate_solution(family, r):
    universe = (1 << r) - 1
    if not mandatory_family(r) <= family:
        raise AssertionError("Faltam conjuntos obrigatórios.")
    if any(not 0 <= x <= universe for x in family):
        raise AssertionError("Máscara fora do universo.")
    if any(y not in family for x in family for y in submasks(x)):
        raise AssertionError("A família não é um downset.")
    if any((x | y) == universe for x in family for y in family):
        raise AssertionError("A família contém uma união total.")
    if any(
        (x in family) == ((universe ^ x) in family)
        for x in range(universe + 1)
    ):
        raise AssertionError("A solução não é saturada.")


class DownsetMILP:
    def __init__(self, r, mode):
        if r not in FULL_LIMITS or mode not in ("original", "reduced"):
            raise ValueError("Dimensão ou modo não suportado.")
        self.r = r
        self.mode = mode
        self.active = active_masks(r)
        universe = (1 << r) - 1
        rows, rhs = [], []

        def add_row(coefficients, upper):
            rows.append(coefficients)
            rhs.append(upper)

        if mode == "original":
            index = {x: i for i, x in enumerate(self.active)}
            self.num_variables = len(index)

            # Todas as inclusões próprias entre conjuntos ativos.
            for x in self.active:
                for y in self.active:
                    if x != y and (x & y) == y:
                        add_row({index[x]: 1, index[y]: -1}, 0)

            # Todos os pares ativos cuja união é o universo.
            for x, y in combinations(self.active, 2):
                if (x | y) == universe:
                    add_row({index[x]: 1, index[y]: 1}, 1)
        else:
            self.pairs = complement_pairs(r)
            self.num_variables = len(self.pairs)
            affine = {}
            for i, (x, y) in enumerate(self.pairs):
                affine[x] = (i, 0, 1)   # z_x = b_i
                affine[y] = (i, 1, -1)  # z_y = 1 - b_i

            # Coberturas entre conjuntos ativos: z_x <= z_y.
            for x in self.active:
                if x.bit_count() < 3:
                    continue
                for i in range(r):
                    bit = 1 << i
                    if not (x & bit):
                        continue
                    y = x ^ bit
                    ix, cx, sx = affine[x]
                    iy, cy, sy = affine[y]
                    coefficients = {ix: sx}
                    coefficients[iy] = coefficients.get(iy, 0) - sy
                    add_row(coefficients, cy - cx)

        self.num_rows = len(rows)
        row_idx, col_idx, data = [], [], []
        for i, row in enumerate(rows):
            for j, value in row.items():
                if value:
                    row_idx.append(i)
                    col_idx.append(j)
                    data.append(float(value))

        matrix = coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(self.num_rows, self.num_variables),
        ).tocsc()
        self.constraint = LinearConstraint(
            matrix, np.full(self.num_rows, -np.inf), np.asarray(rhs, dtype=float)
        )
        self.bounds = Bounds(
            np.zeros(self.num_variables), np.ones(self.num_variables)
        )
        self.integrality = np.ones(self.num_variables, dtype=np.int32)

    def solve(self, n):
        a, weights = weight_table(n)
        constant = 1 + sum(a)
        if self.mode == "original":
            coefficients = [weights[x] for x in self.active]
        else:
            constant += sum(weights[y] for _, y in self.pairs)
            coefficients = [weights[x] - weights[y] for x, y in self.pairs]

        if abs(constant) + sum(map(abs, coefficients)) >= 2**52:
            raise ValueError("Pesos grandes demais para esta verificação numérica.")

        start = time.perf_counter()
        result = milp(
            c=-np.asarray(coefficients, dtype=float),
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=self.constraint,
            options={"disp": False, "presolve": True, "mip_rel_gap": 0.0},
        )
        elapsed = time.perf_counter() - start
        context = f"r={self.r}, n={n}, modelo={self.mode}"

        if result.status != 0 or result.x is None:
            raise RuntimeError(f"{context}: status={result.status}; {result.message}")

        rounded = np.rint(result.x)
        if not np.all(
            ((rounded == 0) | (rounded == 1))
            & (np.abs(result.x - rounded) <= 1e-6)
        ):
            raise RuntimeError(f"{context}: solução não binária.")

        if self.mode == "original":
            chosen = {x for x, b in zip(self.active, rounded) if b == 1}
        else:
            chosen = {x if b == 1 else y for (x, y), b in zip(self.pairs, rounded)}

        family = mandatory_family(self.r) | chosen
        validate_solution(family, self.r)
        value = sum(weights[x] for x in family)

        reported = constant - float(result.fun)
        if not math.isfinite(reported) or abs(reported - value) > 1e-5:
            raise RuntimeError(f"{context}: objetivo numérico difere do peso inteiro.")

        dual = getattr(result, "mip_dual_bound", None)
        upper = None if dual is None else constant - float(dual)
        if upper is None or not math.isfinite(upper) or abs(upper - value) > 1e-5:
            raise RuntimeError(f"{context}: limite dual numérico não fecha.")

        return {
            "value": value,
            "status": int(result.status),
            "message": str(result.message),
            "upper_bound_numeric": upper,
            "gap_numeric": float(result.mip_gap),
            "nodes": int(result.mip_node_count or 0),
            "seconds": elapsed,
        } | family_record(family, self.r)


class HouHuFormula:
    def __init__(self, r):
        self.r = r
        self.universe = (1 << r) - 1
        self.edges = masks_of_size(r, 2)
        self.graphs = []

        if r in (5, 6):
            for graph_mask in range(1 << len(self.edges)):
                graph = tuple(
                    edge for i, edge in enumerate(self.edges)
                    if graph_mask & (1 << i)
                )
                if all(e & f for e, f in combinations(graph, 2)):
                    self.graphs.append(graph)

            if len(self.graphs) != {5: 76, 6: 192}[r]:
                raise AssertionError("Contagem incorreta de grafos intersectantes.")

        if r == 6:
            self.triple_pairs = [
                (c, d) for c, d in complement_pairs(r) if c.bit_count() == 3
            ]
            if len(self.triple_pairs) != 10:
                raise AssertionError("Contagem incorreta de pares de triplas.")

    def evaluate(self, n):
        a, weights = weight_table(tuple(sorted(n, reverse=True)))
        base = 1 + sum(a)

        if self.r == 4:
            star = a[0] * (a[1] + a[2] + a[3])
            triangle = a[0] * a[1] + a[0] * a[2] + a[1] * a[2]
            return {
                "value": base + max(star, triangle),
                "star_gain": star,
                "triangle_gain": triangle,
            }

        base += sum(weights[e] for e in self.edges)
        best_gain, best_graph = None, None

        for graph in self.graphs:
            gain = sum(weights[self.universe ^ e] - weights[e] for e in graph)
            if self.r == 6:
                for c, d in self.triple_pairs:
                    choices = [0]
                    for triple in (c, d):
                        if all((e & triple) != e for e in graph):
                            choices.append(weights[triple])
                    gain += max(choices)

            if best_gain is None or gain > best_gain:
                best_gain, best_graph = gain, graph

        return {
            "value": base + best_gain,
            "graph_size": len(best_graph),
            "graph_edges": [elements(e, self.r) for e in best_graph],
        }


def run_validation(quick, output):
    rows = []
    limits = {4: 4, 5: 4, 6: 4} if quick else FULL_LIMITS

    for r, upper in limits.items():
        formula = HouHuFormula(r)
        original = DownsetMILP(r, "original")
        reduced = DownsetMILP(r, "reduced")
        count = 0

        for values in combinations_with_replacement(range(2, upper + 1), r):
            n = tuple(reversed(values))
            expected = formula.evaluate(n)
            red = reduced.solve(n)
            orig = original.solve(n)

            if not expected["value"] == red["value"] == orig["value"]:
                raise AssertionError(
                    f"Discordância em r={r}, n={n}: fórmula={expected['value']}, "
                    f"reduzido={red['value']}, original={orig['value']}."
                )

            rows.append({
                "r": r,
                "n": list(n),
                "formula": expected,
                "reduced_milp": red,
                "original_milp": orig,
            })
            count += 1

        expected_count = math.comb(upper + r - 2, r) if quick else FULL_COUNTS[r]
        if count != expected_count:
            raise AssertionError(
                f"Contagem incorreta em r={r}: {count}; esperadas {expected_count}."
            )

        print(f"r={r}: {count} instâncias, sem divergências de valor.", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Total: {len(rows)} instâncias, sem divergências de valor.")
    print(f"Arquivo: {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Usa a grade reduzida, com tamanhos de parte de 2 a 4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Caminho do JSON de saída; por padrão, grava em resultados/.",
    )
    args = parser.parse_args(argv)

    prefix = "validation_r456_quick" if args.quick else "validation_r456"
    output = args.output or Path("resultados") / f"{prefix}.json"

    print(
        f"Python {platform.python_version()} | "
        f"NumPy {np.__version__} | SciPy {scipy.__version__}"
    )
    print("Modo:", "rápido" if args.quick else "grade completa")
    start = time.perf_counter()
    run_validation(args.quick, output)
    print(f"Tempo total: {time.perf_counter() - start:.2f} s")
    print("Pesos e viabilidade verificados com inteiros Python.")
    print("A otimalidade dos MILPs é numérica, conforme o SciPy/HiGHS.")


if __name__ == "__main__":
    main()
