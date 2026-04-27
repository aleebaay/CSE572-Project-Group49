
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Airline delay prediction pipeline")
    parser.add_argument("--data", type=str, default="Combined_Flights_2022.csv", help="Path to the CSV dataset")
    parser.add_argument("--sample-size", type=int, default=50000, help="Rows to sample after cleaning")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for streaming CSV")
    parser.add_argument("--outdir", type=str, default="outputs", help="Directory for outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def load_sampled_data(csv_path: Path, sample_size: int, chunk_size: int, seed: int) -> pd.DataFrame:
    usecols = [
        "FlightDate", "Airline", "Origin", "Dest", "Cancelled", "Diverted",
        "CRSDepTime", "CRSElapsedTime", "Distance", "DepDelayMinutes",
        "Month", "DayOfWeek"
    ]

    rng = np.random.default_rng(seed)
    parts = []
    sample_per_chunk = max(200, sample_size // 25)

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        chunk = chunk.copy()
        chunk["FlightDate"] = pd.to_datetime(chunk["FlightDate"], errors="coerce")
        chunk["Cancelled"] = chunk["Cancelled"].astype(str).str.lower().map({"true": True, "false": False})
        chunk["Diverted"] = chunk["Diverted"].astype(str).str.lower().map({"true": True, "false": False})

        # Keep only completed flights to avoid label ambiguity.
        chunk = chunk[(chunk["Cancelled"] != True) & (chunk["Diverted"] != True)]

        chunk["dep_hour"] = (pd.to_numeric(chunk["CRSDepTime"], errors="coerce") // 100).astype("float32")
        chunk["delayed_15"] = (pd.to_numeric(chunk["DepDelayMinutes"], errors="coerce") > 15).astype("int8")

        chunk = chunk[[
            "FlightDate", "Airline", "Origin", "Dest", "Month", "DayOfWeek",
            "dep_hour", "CRSElapsedTime", "Distance", "delayed_15"
        ]].dropna(subset=["FlightDate", "Airline", "Origin", "Dest", "delayed_15"])

        if len(chunk) > 0:
            take_n = min(sample_per_chunk, len(chunk))
            parts.append(chunk.sample(n=take_n, random_state=seed))

        if sum(len(x) for x in parts) >= sample_size:
            break

    if not parts:
        raise ValueError("No usable rows were found in the dataset.")

    df = pd.concat(parts, ignore_index=True)
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df


def make_split(df: pd.DataFrame):
    df = df.sort_values("FlightDate").reset_index(drop=True)
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    X_train = train_df.drop(columns=["FlightDate", "delayed_15"])
    y_train = train_df["delayed_15"].astype(int)
    X_val = val_df.drop(columns=["FlightDate", "delayed_15"])
    y_val = val_df["delayed_15"].astype(int)
    X_test = test_df.drop(columns=["FlightDate", "delayed_15"])
    y_test = test_df["delayed_15"].astype(int)

    return (X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df)


def build_models(seed: int):
    num_cols = ["Month", "DayOfWeek", "dep_hour", "CRSElapsedTime", "Distance"]
    cat_cols = ["Airline", "Origin", "Dest"]

    logit_preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    tree_preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "Logistic Regression": Pipeline([
            ("prep", logit_preprocess),
            ("model", LogisticRegression(max_iter=200, solver="liblinear", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("prep", tree_preprocess),
            ("model", RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=seed,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", tree_preprocess),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_depth=4,
                max_iter=120,
                random_state=seed,
            )),
        ]),
    }
    return models


def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    for name, pipe in models.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        test_proba = pipe.predict_proba(X_test)[:, 1]

        val_pred = (val_proba >= 0.5).astype(int)
        test_pred = (test_proba >= 0.5).astype(int)

        results[name] = {
            "pipe": pipe,
            "val_f1": f1_score(y_val, val_pred),
            "val_auc": roc_auc_score(y_val, val_proba),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred),
            "test_auc": roc_auc_score(y_test, test_proba),
            "test_proba": test_proba,
            "test_pred": test_pred,
        }

    return results


def tune_threshold(pipe, X_val, y_val):
    val_proba = pipe.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    scores = []
    for t in thresholds:
        scores.append(f1_score(y_val, (val_proba >= t).astype(int)))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.3, 4.6))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = build_arg_parser().parse_args(args=[])
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = load_sampled_data(data_path, args.sample_size, args.chunk_size, args.seed)
    df.to_csv(outdir / "sampled_data.csv", index=False)

    print("Sample size:", len(df))
    print("Class distribution:")
    print(df["delayed_15"].value_counts(normalize=True).sort_index().rename("share").to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df["delayed_15"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["On-time (0)", "Delayed >15m (1)"], rotation=0)
    ax.set_title("Target Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(outdir / "target_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.groupby("dep_hour")["delayed_15"].mean().sort_index().plot(ax=ax)
    ax.set_title("Average Delay Rate by Scheduled Departure Hour")
    ax.set_xlabel("Scheduled departure hour")
    ax.set_ylabel("Delay rate")
    fig.tight_layout()
    fig.savefig(outdir / "delay_rate_by_hour.png", dpi=200)
    plt.close(fig)

    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = make_split(df)

    models = build_models(args.seed)
    results = evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test)

    best_name = max(results, key=lambda k: results[k]["val_f1"])
    best_pipe = results[best_name]["pipe"]
    best_threshold, best_val_f1 = tune_threshold(best_pipe, X_val, y_val)

    best_test_proba = results[best_name]["test_proba"]
    tuned_pred = (best_test_proba >= best_threshold).astype(int)

    results[best_name]["tuned_threshold"] = best_threshold
    results[best_name]["test_precision_tuned"] = precision_score(y_test, tuned_pred, zero_division=0)
    results[best_name]["test_recall_tuned"] = recall_score(y_test, tuned_pred, zero_division=0)
    results[best_name]["test_f1_tuned"] = f1_score(y_test, tuned_pred)
    results[best_name]["test_auc_tuned"] = roc_auc_score(y_test, best_test_proba)

    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "model": name,
            "val_f1": round(r["val_f1"], 4),
            "val_auc": round(r["val_auc"], 4),
            "test_precision@0.5": round(r["test_precision"], 4),
            "test_recall@0.5": round(r["test_recall"], 4),
            "test_f1@0.5": round(r["test_f1"], 4),
            "test_auc": round(r["test_auc"], 4),
        })

    summary = pd.DataFrame(summary_rows).sort_values("test_f1@0.5", ascending=False)
    summary.to_csv(outdir / "model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(summary.to_string(index=False))
    print(f"\nBest model by validation F1: {best_name}")
    print(f"Best threshold on validation: {best_threshold:.2f}")
    print("Tuned test metrics:")
    print({
        "precision": round(results[best_name]["test_precision_tuned"], 4),
        "recall": round(results[best_name]["test_recall_tuned"], 4),
        "f1": round(results[best_name]["test_f1_tuned"], 4),
        "roc_auc": round(results[best_name]["test_auc_tuned"], 4),
    })

    with open(outdir / "metrics.json", "w") as f:
        json.dump({
            "sample_size": int(len(df)),
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "best_model": best_name,
            "best_threshold": best_threshold,
            "results": {
                k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                    for kk, vv in v.items()
                    if kk not in {"pipe", "test_proba", "test_pred"}}
                for k, v in results.items()
            },
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["test_f1@0.5", "test_auc"]:
        ax.plot(summary["model"], summary[metric], marker="o", label=metric)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison on Test Split")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "model_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["test_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curves on Test Split")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "roc_curves.png", dpi=200)
    plt.close(fig)

    save_confusion_matrix(
        y_test,
        results[best_name]["test_pred"],
        f"Confusion Matrix - {best_name} @0.5",
        outdir / "cm_default.png",
    )
    save_confusion_matrix(
        y_test,
        tuned_pred,
        f"Confusion Matrix - {best_name} @tuned={best_threshold:.2f}",
        outdir / "cm_tuned.png",
    )

    # Small subset for fast feature importance
    sample_n = min(5000, len(X_test))
    subset = X_test.sample(n=sample_n, random_state=args.seed)
    subset_y = y_test.loc[subset.index]
    perm = permutation_importance(best_pipe, subset, subset_y, n_repeats=2, random_state=args.seed, scoring="f1", n_jobs=-1)
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(outdir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    top_imp = imp.head(8).iloc[::-1]
    ax.barh(top_imp["feature"], top_imp["importance_mean"])
    ax.set_title(f"Permutation Importance - {best_name}")
    ax.set_xlabel("Mean importance (F1 decrease)")
    fig.tight_layout()
    fig.savefig(outdir / "feature_importance.png", dpi=200)
    plt.close(fig)

    print("\nTop feature importance:")
    print(imp.head(10).to_string(index=False))
    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
