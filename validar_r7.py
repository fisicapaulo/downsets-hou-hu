"""Compara dois MILPs, o catálogo shifted e as 43 famílias canônicas para r=7."""

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

from enumerar_r7 import enumerate_shifted_saturated, family_record

R = 7
U = (1 << R) - 1
MASKS = set(range(U + 1))
MANDATORY = {0} | {1 << i for i in range(R)}
ACTIVE = tuple(x for x in range(U + 1) if 2 <= x.bit_count() <= 5)
PAIRS = tuple((x, U ^ x) for x in ACTIVE if x < (U ^ x))
CERTIFICATION_NOTE = (
    "Viabilidade e pesos verificados com inteiros Python. "
    "A otimalidade MILP é reportada numericamente pelo SciPy/HiGHS; "
    "não há certificado formal exato do limite dual."
)


def grid_r7(historical=None):
    if historical is None:
        grid = {
            tuple(reversed(v))
            for v in combinations_with_replacement(range(2, 6), R)
        }
        grid.update({
            (10, 2, 2, 2, 2, 2, 2),
            (10, 10, 10, 10, 10, 10, 10),
        })
        source = "grade autônoma: 120 vetores em [2,5] mais dois testes"
    else:
        document = json.loads(historical.read_text(encoding="utf-8"))
        vectors = []

        def collect(obj):
            if isinstance(obj, dict):
                n = obj.get("n")
                if (
                    isinstance(n, list)
                    and len(n) == R
                    and all(type(x) is int and x >= 2 for x in n)
                ):
                    vectors.append(tuple(n))
                    return
                for value in obj.values():
                    collect(value)
            elif isinstance(obj, list):
                for value in obj:
                    collect(value)

        collect(document)
        grid = set(vectors)
        if len(grid) != len(vectors):
            raise ValueError("Há vetores repetidos no arquivo histórico.")
        source = f"vetores históricos de {historical.name}"

    if len(grid) != 122:
        raise ValueError(f"Esperados 122 vetores distintos, encontrados {len(grid)}.")
    if any(n != tuple(sorted(n, reverse=True)) for n in grid):
        raise ValueError("A grade contém vetores fora da ordem não crescente.")

    return sorted(grid), source


def weights_r7(n):
    a = [ni - 1 for ni in n]
    weights = [1] * (U + 1)
    for x in range(1, U + 1):
        bit = x & -x
        weights[x] = weights[x ^ bit] * a[bit.bit_length() - 1]
    return weights


def check_family_r7(family):
    if not MANDATORY <= family or not family <= MASKS:
        raise AssertionError("Máscaras inválidas ou conjuntos obrigatórios ausentes.")

    for x in family:
        for i in range(R):
            bit = 1 << i
            if x & bit and (x ^ bit) not in family:
                raise AssertionError("A família não é um downset.")

    if any((x in family) == ((U ^ x) in family) for x in MASKS):
        raise AssertionError("A família não é saturada.")

    # Downset + saturação impedem união total.


def canonical_r7():
    families = {
        "lower_ball": {x for x in MASKS if x.bit_count() <= 3}
    }

    for i in range(R):
        bit = 1 << i
        full_subcube = U ^ bit
        families[f"subcube_omit_{i + 1}"] = {
            x for x in MASKS if not (x & bit) and x != full_subcube
        } | {bit}

    for triple in combinations(range(R), 3):
        t = sum(1 << i for i in triple)
        label = "4plus3_T_" + "_".join(str(i + 1) for i in triple)
        families[label] = {x for x in MASKS if (x & t).bit_count() <= 1}

    if len(families) != 43:
        raise AssertionError("Esperadas 43 famílias canônicas.")
    for family in families.values():
        check_family_r7(family)

    return families


class MILP7:
    def __init__(self, reduced):
        self.reduced = reduced
        if reduced:
            self.columns = PAIRS
            affine = {}
            for i, (x, y) in enumerate(PAIRS):
                affine[x] = (i, 0, 1)   # z_x = b_i
                affine[y] = (i, 1, -1)  # z_y = 1 - b_i
        else:
            self.columns = ACTIVE
            affine = {x: (i, 0, 1) for i, x in enumerate(ACTIVE)}

        rr, cc, data, upper = [], [], [], []

        def add(terms, rhs):
            coefficients = {}
            for mask, factor in terms:
                i, offset, slope = affine[mask]
                rhs -= factor * offset
                coefficients[i] = coefficients.get(i, 0) + factor * slope
            for i, coefficient in coefficients.items():
                if coefficient:
                    rr.append(len(upper))
                    cc.append(i)
                    data.append(float(coefficient))
            upper.append(rhs)

        # Todas as coberturas entre conjuntos ativos, nos dois modelos.
        for x in ACTIVE:
            for i in range(R):
                bit = 1 << i
                if x & bit and (x ^ bit) in affine:
                    add(((x, 1), (x ^ bit, -1)), 0)

        if not reduced:
            for x, y in combinations(ACTIVE, 2):
                if (x | y) == U:
                    add(((x, 1), (y, 1)), 1)

        self.shape = (len(upper), len(self.columns))
        expected_shape = (350, 56) if reduced else (896, 112)
        if self.shape != expected_shape:
            raise AssertionError(f"Dimensões inesperadas do MILP: {self.shape}.")

        matrix = coo_matrix((data, (rr, cc)), shape=self.shape).tocsc()
        self.constraint = LinearConstraint(matrix, -np.inf, upper)
        self.bounds = Bounds(np.zeros(self.shape[1]), np.ones(self.shape[1]))
        self.integrality = np.ones(self.shape[1], dtype=np.int32)

    def solve(self, n, weights):
        constant = sum(weights[x] for x in MANDATORY)
        if self.reduced:
            constant += sum(weights[y] for _, y in self.columns)
            coefficients = [weights[x] - weights[y] for x, y in self.columns]
        else:
            coefficients = [weights[x] for x in self.columns]

        if abs(constant) + sum(map(abs, coefficients)) >= 2**52:
            raise ValueError(f"Pesos fora da faixa numérica adotada: {n}.")

        start = time.perf_counter()
        result = milp(
            c=-np.asarray(coefficients, dtype=float),
            integrality=self.integrality,
            bounds=self.bounds,
            constraints=self.constraint,
            options={"mip_rel_gap": 0.0, "presolve": True},
        )
        elapsed = time.perf_counter() - start
        mode = "reduzido" if self.reduced else "original"
        context = f"n={n}, modelo={mode}"

        if result.status != 0 or result.x is None:
            raise RuntimeError(f"{context}: status={result.status}; {result.message}")

        rounded = np.rint(result.x)
        if not np.all(
            ((rounded == 0) | (rounded == 1))
            & (np.abs(result.x - rounded) <= 1e-6)
        ):
            raise RuntimeError(f"{context}: solução não binária.")

        if self.reduced:
            chosen = {
                x if b == 1 else y for (x, y), b in zip(self.columns, rounded)
            }
        else:
            chosen = {x for x, b in zip(self.columns, rounded) if b == 1}

        family = MANDATORY | chosen
        check_family_r7(family)
        value = sum(weights[x] for x in family)
        objective = constant - float(result.fun)
        dual = getattr(result, "mip_dual_bound", None)
        upper = math.nan if dual is None else constant - float(dual)

        if any(
            not math.isfinite(v) or abs(v - value) > 1e-5
            for v in (objective, upper)
        ):
            raise RuntimeError(f"{context}: objetivo ou limite dual numérico não fecha.")

        return {
            "value": value,
            "status": int(result.status),
            "message": str(result.message),
            "upper_bound_numeric": upper,
            "gap_numeric": float(result.mip_gap),
            "nodes": int(result.mip_node_count or 0),
            "seconds": elapsed,
        } | family_record(family, R)


def run_validation(historical, output):
    if historical is not None and historical.resolve() == output.resolve():
        raise ValueError("A saída não pode sobrescrever o arquivo histórico.")

    start = time.perf_counter()
    grid, source = grid_r7(historical)
    shifted = enumerate_shifted_saturated(r=R)["families"]
    if len(shifted) != 134:
        raise AssertionError(f"Esperadas 134 famílias shifted, encontradas {len(shifted)}.")

    canonical = canonical_r7()
    original = MILP7(reduced=False)
    reduced = MILP7(reduced=True)
    rows = []
    discrepancies = []

    print(f"Grade: {source}")
    print(f"Catálogo: {len(shifted)} shifted; {len(canonical)} canônicas.")
    print(f"Original: {original.shape[1]} variáveis, {original.shape[0]} linhas.")
    print(f"Reduzido: {reduced.shape[1]} variáveis, {reduced.shape[0]} linhas.")

    for j, n in enumerate(grid, 1):
        weights = weights_r7(n)
        orig = original.solve(n, weights)
        red = reduced.solve(n, weights)

        shifted_values = [sum(weights[x] for x in family) for family in shifted]
        best_shifted = max(shifted_values)
        shifted_ids = [i for i, v in enumerate(shifted_values) if v == best_shifted]

        canonical_values = {
            label: sum(weights[x] for x in family)
            for label, family in canonical.items()
        }
        best_canonical = max(canonical_values.values())
        canonical_ties = [
            label for label, v in canonical_values.items() if v == best_canonical
        ]

        agrees = orig["value"] == red["value"] == best_shifted == best_canonical
        if not agrees:
            discrepancies.append(list(n))

        rows.append({
            "r": R,
            "n": list(n),
            "original_milp": orig,
            "reduced_milp": red,
            "shifted": {"value": best_shifted, "maximizing_ids": shifted_ids},
            "canonical": {"value": best_canonical, "maximizing_labels": canonical_ties},
            "all_values_match": agrees,
        })

        if j % 25 == 0 or j == len(grid):
            print(f"{j}/{len(grid)} instâncias concluídas.", flush=True)

    data = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver_interface": "scipy.optimize.milp / HiGHS",
        },
        "grid_source": source,
        "instances": len(rows),
        "shifted_count": len(shifted),
        "canonical_count": len(canonical),
        "discrepancies": discrepancies,
        "seconds": time.perf_counter() - start,
        "certification_note": CERTIFICATION_NOTE,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Total: {len(rows)} instâncias; divergências: {len(discrepancies)}.")
    print(f"Tempo da validação: {data['seconds']:.2f} s")
    print(f"Arquivo: {output.resolve()}")
    if discrepancies:
        print("Vetores com divergência:", discrepancies)

    return 1 if discrepancies else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical",
        type=Path,
        help="JSON histórico do qual extrair os 122 vetores n.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Caminho do JSON de saída; por padrão, grava em resultados/.",
    )
    args = parser.parse_args(argv)

    filename = (
        "validation_r7.json" if args.historical is None
        else "validation_r7_historical.json"
    )
    output = args.output or Path("resultados") / filename
    return run_validation(args.historical, output)


if __name__ == "__main__":
    raise SystemExit(main())
