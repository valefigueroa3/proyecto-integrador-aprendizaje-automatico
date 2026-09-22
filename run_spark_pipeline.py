"""Entrenamiento reproducible de los seis modelos con PySpark ML.

El script usa la base parquet derivada del CSV completo y la partición común ya
construida. No usa toPandas/collect sobre la base completa; solo agrega métricas
y escribe las predicciones de prueba a parquet.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import pandas as pd
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LinearSVC,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import vector_to_array


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "lending_club_model_base.parquet"
OUT = ROOT / "results" / "spark"
ART = ROOT / "artifacts" / "spark"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)


def numeric_metrics(pred):
    evaluator = BinaryClassificationEvaluator(
        labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = float(evaluator.evaluate(pred))
    counts = pred.select(
        F.sum(F.when((F.col("prediction") == 0) & (F.col("default") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("prediction") == 0) & (F.col("default") == 1), 1).otherwise(0)).alias("fn"),
        F.sum(F.when((F.col("prediction") == 1) & (F.col("default") == 0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("prediction") == 1) & (F.col("default") == 1), 1).otherwise(0)).alias("tp"),
        F.count(F.lit(1)).alias("n"),
    ).first()
    tn, fn, fp, tp, n = [int(counts[x] or 0) for x in ("tn", "fn", "fp", "tp", "n")]
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_test": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    conf = (
        SparkConf()
        .setAppName("lending-club-integrador")
        .setMaster(os.environ.get("SPARK_MASTER", "local[4]"))
        .set("spark.driver.memory", "8g")
        .set("spark.sql.shuffle.partitions", "16")
        .set("spark.default.parallelism", "16")
        .set("spark.sql.execution.arrow.pyspark.enabled", "false")
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = spark.read.parquet(str(DATA))
    cat_cols = ["term", "grade", "emp_length", "home_ownership", "verification_status", "purpose", "addr_state"]
    num_cols = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_high"]
    train = raw.where(F.col("split") == "train").select(["id", "default"] + num_cols + cat_cols).cache()
    test = raw.where(F.col("split") == "test").select(["id", "default"] + num_cols + cat_cols).cache()
    train_n, test_n = train.count(), test.count()

    # Ajuste de transformaciones únicamente sobre train. Se reutiliza después
    # por todos los estimadores y se almacena tras VectorAssembler.
    imputed_num = [f"{c}_imp" for c in num_cols]
    imputer = Imputer(inputCols=num_cols, outputCols=imputed_num, strategy="median")
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_oh" for c in cat_cols], handleInvalid="keep")
    assembler = VectorAssembler(inputCols=imputed_num + [f"{c}_oh" for c in cat_cols], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)
    prep = Pipeline(stages=[imputer] + indexers + [encoder, assembler, scaler]).fit(train)
    train_v = prep.transform(train).select("id", "default", "features").cache()
    test_v = prep.transform(test).select("id", "default", "features").cache()
    train_v.count(); test_v.count()

    specs = {
        "logistic_regression": (LogisticRegression(featuresCol="features", labelCol="default", maxIter=100), {"regParam": [1e-6, 1e-4], "elasticNetParam": [0.0]}),
        "decision_tree": (DecisionTreeClassifier(featuresCol="features", labelCol="default", seed=42), {"maxDepth": [5, 10]}),
        "random_forest": (RandomForestClassifier(featuresCol="features", labelCol="default", seed=42, featureSubsetStrategy="sqrt"), {"numTrees": [10], "maxDepth": [10, 15]}),
        "gradient_boosted_tree": (GBTClassifier(featuresCol="features", labelCol="default", seed=42), {"maxIter": [50], "maxDepth": [3, 5]}),
        "linear_svc": (LinearSVC(featuresCol="features", labelCol="default", maxIter=100, tol=1e-4), {"regParam": [1e-6, 1e-4]}),
        "naive_bayes": (NaiveBayes(featuresCol="features", labelCol="default", modelType="gaussian"), {"smoothing": [0.1, 1.0]}),
    }
    evaluator = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    manifest = {
        "spark_version": spark.version,
        "python": platform.python_version(),
        "train_rows": train_n,
        "test_rows": test_n,
        "num_folds": 2,
        "random_seed": 42,
        "features": num_cols + cat_cols,
        "note": "Grid reducido y 2 folds por coste computacional; sin muestreo de filas.",
    }
    (OUT / "hardware.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_metrics = []
    for name, (estimator, grid_dict) in specs.items():
        print(f"[spark] iniciando {name}", flush=True)
        builder = ParamGridBuilder()
        for param_name, values in grid_dict.items():
            builder = builder.addGrid(getattr(estimator, param_name), values)
        cv = CrossValidator(
            estimator=estimator,
            estimatorParamMaps=builder.build(),
            evaluator=evaluator,
            numFolds=2,
            seed=42,
            parallelism=1,
        )
        t0 = time.perf_counter()
        fitted = cv.fit(train_v)
        fit_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = fitted.transform(test_v).select("id", "default", "rawPrediction", "prediction")
        metrics = numeric_metrics(pred)
        predict_seconds = time.perf_counter() - t1
        # Hadoop en Windows exige winutils.exe para escribir desde Spark. Para
        # evitar esa dependencia, solo las predicciones de prueba (no la base)
        # se materializan en pandas y se guardan con pyarrow.
        pred_out = pred.select(
            "id", "default", "prediction", vector_to_array("rawPrediction")[1].alias("score")
        ).toPandas()
        pred_out.to_parquet(ART / f"predictions_{name}.parquet", index=False)
        cv_rows = [{"model": name, "fold_cv_auc": float(x)} for x in fitted.avgMetrics]
        (OUT / f"cv_{name}.json").write_text(json.dumps(cv_rows, indent=2), encoding="utf-8")
        metrics.update({"model": name, "fit_cv_seconds": fit_seconds, "predict_seconds": predict_seconds})
        (OUT / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        all_metrics.append(metrics)
        print(json.dumps(metrics), flush=True)

    (OUT / "comparacion_metricas_spark.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    spark.stop()


if __name__ == "__main__":
    main()
