"""Explicación local LIME del mejor modelo sklearn por ROC-AUC."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "lending_club_model_base.parquet"
ART = ROOT / "artifacts" / "sklearn"
OUT = ROOT / "results" / "lime"
FIG = ROOT / "figures" / "lime"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

NUM = ["loan_amnt", "int_rate", "fico_range_high", "annual_inc", "dti"]
CAT = ["emp_length", "purpose", "home_ownership", "addr_state", "verification_status", "grade", "term"]
FEATURES = NUM + CAT


def main():
    train = pd.read_parquet(DATA, filters=[("split", "==", "train")], columns=FEATURES).sample(n=5000, random_state=42)
    test = pd.read_parquet(DATA, filters=[("split", "==", "test")], columns=["id"] + FEATURES + ["default"])
    test_row = test.loc[test["default"].eq(1)].iloc[0]
    pre = joblib.load(ART / "preprocessor.joblib")
    model = joblib.load(ART / "models" / "hist_gradient_boosting.joblib")

    categories = {}
    for col in CAT:
        values = sorted(train[col].fillna("Missing").astype(str).unique().tolist())
        categories[col] = values
    def encode(df):
        out = pd.DataFrame(index=df.index)
        for col in NUM:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(train[col].median())
        for col in CAT:
            lookup = {v: i for i, v in enumerate(categories[col])}
            out[col] = df[col].fillna("Missing").astype(str).map(lookup).fillna(-1)
        return out[FEATURES].astype(float)

    background = encode(train).to_numpy()
    row = encode(pd.DataFrame([test_row[FEATURES]])).iloc[0].to_numpy()
    cat_idx = [FEATURES.index(c) for c in CAT]
    explainer = LimeTabularExplainer(
        background,
        feature_names=FEATURES,
        class_names=["Fully Paid", "Charged Off"],
        categorical_features=cat_idx,
        categorical_names={i: categories[c] for i, c in enumerate(CAT, start=len(NUM))},
        discretize_continuous=True,
        random_state=42,
    )

    def predict_fn(x):
        xdf = pd.DataFrame(x, columns=FEATURES)
        decoded = xdf.copy()
        for col in NUM:
            decoded[col] = pd.to_numeric(decoded[col], errors="coerce")
        for col in CAT:
            vals = categories[col]
            idx = np.clip(np.rint(decoded[col].to_numpy()).astype(int), 0, len(vals) - 1)
            decoded[col] = [vals[i] for i in idx]
        matrix = pre.transform(decoded[FEATURES]).toarray().astype(np.float32, copy=False)
        proba = model.predict_proba(matrix)
        return proba

    explanation = explainer.explain_instance(row, predict_fn, labels=(1,), num_features=len(FEATURES), num_samples=800)
    table = pd.DataFrame(explanation.as_list(label=1), columns=["feature", "weight"])
    table["abs_weight"] = table["weight"].abs()
    table = table.sort_values("abs_weight", ascending=False)
    table.to_csv(OUT / "explicacion_local_hist_gradient_boosting.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "caso_explicado.json", "w", encoding="utf-8") as f:
        json.dump({"id": str(test_row.get("id", "")), "target": int(test_row["default"]), "model": "HistGradientBoostingClassifier", "n_background": 5000, "n_lime_samples": 800}, f, indent=2)
    fig = explanation.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(FIG / "lime_hist_gradient_boosting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
