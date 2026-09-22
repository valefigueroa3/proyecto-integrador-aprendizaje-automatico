"""Pruebas pareadas sobre la misma prueba común.

Incluye DeLong para comparar AUC, McNemar exacta para las decisiones y un
bootstrap pareado de la diferencia de AUC. Los p-valores de cada familia se
ajustan con Holm.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import binomtest, norm, rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
SK = ROOT / "artifacts" / "sklearn" / "predictions"
SP = ROOT / "artifacts" / "spark"
OUT = ROOT / "results" / "estadisticas"
OUT.mkdir(parents=True, exist_ok=True)


def compute_midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = np.asarray(x)[order]
    j = 0
    ranks = np.empty(len(x), dtype=float)
    while j < len(z):
        k = j
        while k < len(z) and z[k] == z[j]:
            k += 1
        ranks[j:k] = 0.5 * (j + k - 1) + 1
        j = k
    midranks = np.empty(len(x), dtype=float)
    midranks[order] = ranks
    return midranks


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = compute_midrank(predictions_sorted_transposed[r, :m])
        ty[r] = compute_midrank(predictions_sorted_transposed[r, m:])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_paired(y, s1, s2):
    y = np.asarray(y, dtype=int)
    order = np.argsort(-y)
    m = int(y.sum())
    if m == 0 or m == len(y):
        return {"auc_1": np.nan, "auc_2": np.nan, "delta_auc": np.nan, "z": np.nan, "p_value": np.nan}
    scores = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    aucs, cov = fast_delong(scores, m)
    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = delta / np.sqrt(max(var, 1e-15))
    p_value = max(float(2 * norm.sf(abs(z))), float(np.finfo(float).tiny))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "delta_auc": delta, "z": float(z), "p_value": p_value}


def mcnemar(y, p1, p2):
    y = np.asarray(y, dtype=int)
    p1, p2 = np.asarray(p1, dtype=int), np.asarray(p2, dtype=int)
    b = int(np.sum((p1 == y) & (p2 != y)))
    c = int(np.sum((p1 != y) & (p2 == y)))
    p = float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue) if b + c else 1.0
    return {"b_model_1_correct_model_2_wrong": b, "c_model_1_wrong_model_2_correct": c, "p_value": p}


def bootstrap_auc_difference(y, s1, s2, n_boot=100, seed=42):
    y = np.asarray(y, dtype=int)
    s1, s2 = np.asarray(s1), np.asarray(s2)
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)

    def prepare(score):
        order = np.argsort(score, kind="mergesort")
        yo = y[order]
        return yo.astype(float), order

    y1, order1 = prepare(s1)
    y2, order2 = prepare(s2)

    def weighted_auc(score_y, order, weights):
        w = weights[order]
        pos = w * score_y
        neg = w * (1.0 - score_y)
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            return np.nan
        # Las puntuaciones se ordenan una sola vez; cada réplica usa pesos
        # multinomiales equivalentes a remuestrear con reemplazo.
        neg_before = np.cumsum(neg) - neg
        numerator = np.sum(pos * neg_before) + 0.5 * np.sum(pos * neg)
        return numerator / (n_pos * n_neg)

    for _ in range(n_boot):
        weights = rng.multinomial(n, np.full(n, 1.0 / n))
        a1 = weighted_auc(y1, order1, weights)
        a2 = weighted_auc(y2, order2, weights)
        if np.isfinite(a1) and np.isfinite(a2):
            deltas.append(a1 - a2)
    deltas = np.asarray(deltas)
    observed = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    tail_count = min(int(np.sum(deltas <= 0)), int(np.sum(deltas >= 0)))
    # Corrección Monte Carlo para que el p-valor no se reporte como cero.
    bootstrap_p = min(1.0, 2.0 * (tail_count + 1) / (len(deltas) + 1))
    return {
        "observed_delta_auc": float(observed),
        "bootstrap_replicates": int(len(deltas)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "p_value_two_sided": float(bootstrap_p),
    }


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[idx]))
        adjusted[idx] = running
    return adjusted


def load_predictions(folder, models, suffix=""):
    out = {}
    for model in models:
        p = folder / f"{suffix}{model}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            out[model] = df.sort_values("id")
    return out


def compare(name, pred, reference):
    models = sorted(pred)
    if len(models) < 2:
        return
    base = pred[models[0]]
    y = base["default"].to_numpy()
    rows = []
    # La familia principal contrasta el mejor AUC contra cada alternativa,
    # que es la comparación relevante para seleccionar el modelo final.
    pairs = [(reference, other) for other in models if other != reference]
    for a, b in pairs:
        left, right = pred[a], pred[b]
        if not np.array_equal(left["id"].to_numpy(), right["id"].to_numpy()):
            raise ValueError("Las predicciones no están alineadas por id")
        d = delong_paired(y, left["score"], right["score"])
        m = mcnemar(y, left["prediction"], right["prediction"])
        boot = bootstrap_auc_difference(y, left["score"], right["score"])
        rows.append({"model_1": a, "model_2": b, **{f"delong_{k}": v for k, v in d.items()}, **{f"mcnemar_{k}": v for k, v in m.items()}, **{f"bootstrap_{k}": v for k, v in boot.items()}})
    result = pd.DataFrame(rows)
    for family, col in [("delong", "delong_p_value"), ("mcnemar", "mcnemar_p_value"), ("bootstrap", "bootstrap_p_value_two_sided")]:
        result[f"{family}_holm_p_value"] = holm(result[col].to_numpy())
    result.to_csv(OUT / f"comparaciones_{name}.csv", index=False, encoding="utf-8-sig")


def compare_environments(sk, sp):
    """Compara implementaciones análogas sobre las mismas filas de prueba."""
    model_pairs = [
        ("logistic_regression", "logistic_regression", "Regresión logística"),
        ("decision_tree", "decision_tree", "Árbol de decisión"),
        ("random_forest", "random_forest", "Bosque aleatorio"),
        ("hist_gradient_boosting", "gradient_boosted_tree", "Boosting"),
        ("linear_svc", "linear_svc", "SVM lineal"),
        ("gaussian_nb", "naive_bayes", "Naive Bayes gaussiano"),
    ]
    rows = []
    for sk_name, sp_name, family in model_pairs:
        if sk_name not in sk or sp_name not in sp:
            continue
        left, right = sk[sk_name], sp[sp_name]
        if not np.array_equal(left["id"].to_numpy(), right["id"].to_numpy()):
            raise ValueError(f"Las predicciones no están alineadas por id: {family}")
        y = left["default"].to_numpy()
        if not np.array_equal(y, right["default"].to_numpy()):
            raise ValueError(f"Las etiquetas no coinciden entre entornos: {family}")
        d = delong_paired(y, left["score"], right["score"])
        m = mcnemar(y, left["prediction"], right["prediction"])
        boot = bootstrap_auc_difference(y, left["score"], right["score"])
        rows.append({
            "familia_modelo": family,
            "modelo_sklearn": sk_name,
            "modelo_pyspark": sp_name,
            **{f"delong_{k}": v for k, v in d.items()},
            **{f"mcnemar_{k}": v for k, v in m.items()},
            **{f"bootstrap_{k}": v for k, v in boot.items()},
        })
    result = pd.DataFrame(rows)
    for family, col in [
        ("delong", "delong_p_value"),
        ("mcnemar", "mcnemar_p_value"),
        ("bootstrap", "bootstrap_p_value_two_sided"),
    ]:
        result[f"{family}_holm_p_value"] = holm(result[col].to_numpy())
    result.to_csv(OUT / "comparaciones_entornos.csv", index=False, encoding="utf-8-sig")


def main():
    sk_models = ["decision_tree", "gaussian_nb", "hist_gradient_boosting", "linear_svc", "logistic_regression", "random_forest"]
    sp_models = ["decision_tree", "gradient_boosted_tree", "linear_svc", "logistic_regression", "naive_bayes", "random_forest"]
    sk = load_predictions(SK, sk_models, suffix="test_")
    # Spark files tienen el mismo esquema, pero no prefijo test_.
    sp = load_predictions(SP, sp_models, suffix="predictions_")
    compare("sklearn", sk, "hist_gradient_boosting")
    compare("spark", sp, "gradient_boosted_tree")
    compare_environments(sk, sp)
    (OUT / "manifest.json").write_text(json.dumps({"bootstrap_replicates": 100, "correction": "Holm within each family of comparisons", "pairs": "best AUC model versus alternatives within each environment, and six corresponding model families across environments on the common test set", "note": "Bootstrap pareado multinomial sobre todas las observaciones. Los p-valores usan corrección de Monte Carlo; la resolución es limitada por 100 réplicas y se interpretan junto con el intervalo de confianza."}, indent=2), encoding="utf-8")
    print({"sklearn_models": list(sk), "spark_models": list(sp)})


if __name__ == "__main__":
    main()
