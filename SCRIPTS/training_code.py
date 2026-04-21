import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42


def is_categorical(series: pd.Series) -> bool:
    dtype = series.dtype
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype)


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert target to binary labels where 1 means churn."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "yes": 1,
        "true": 1,
        "churn": 1,
        "1": 1,
        "no": 0,
        "false": 0,
        "no churn": 0,
        "0": 0,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        unknown_values = series[normalized.isna()].dropna().unique().tolist()
        raise ValueError(f"Unsupported target values found: {unknown_values}")
    return normalized.astype(int)


def build_pipeline(categorical_features, numeric_features, n_estimators: int, n_jobs: int):
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        max_depth=20,
        min_samples_leaf=5,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_metadata(df: pd.DataFrame, feature_columns):
    metadata = {
        "feature_columns": feature_columns,
        "categorical_features": [],
        "numeric_features": [],
        "categorical_values": {},
        "numeric_defaults": {},
    }

    for col in feature_columns:
        if is_categorical(df[col]):
            metadata["categorical_features"].append(col)
            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .head(50)
                .index
                .tolist()
            )
            metadata["categorical_values"][col] = values
        else:
            metadata["numeric_features"].append(col)
            metadata["numeric_defaults"][col] = float(df[col].median(skipna=True))

    return metadata


def train_model(
    input_csv: Path,
    output_model: Path,
    output_metrics: Path,
    max_rows: int,
    n_estimators: int,
    n_jobs: int,
):
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv, encoding="latin1")
    df = df.drop_duplicates().reset_index(drop=True)

    if max_rows and len(df) > max_rows:
        print(f"Dataset has {len(df):,} rows. Sampling down to {max_rows:,} rows for faster training.")
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    if "CHURN" not in df.columns:
        raise ValueError("The dataset must contain a 'CHURN' target column.")

    y = normalize_target(df["CHURN"])
    X = df.drop(columns=["CHURN"])

    categorical_features = [
        col
        for col in X.columns
        if is_categorical(X[col])
    ]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
    }

    metadata = build_metadata(X, X.columns.tolist())
    bundle = {
        "pipeline": pipeline,
        "metadata": metadata,
        "target_mapping": {"no_churn": 0, "churn": 1},
    }

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, output_model)
    with output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model bundle saved to: {output_model}")
    print(f"Metrics saved to: {output_metrics}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn model and export a Streamlit-ready artifact.")
    parser.add_argument(
        "--input",
        default="DATA/Expresso_churn_dataset.csv",
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--model-output",
        default="MODEL/churn_model_bundle.joblib",
        help="Path for saved model bundle.",
    )
    parser.add_argument(
        "--metrics-output",
        default="MODEL/metrics.json",
        help="Path for saved model metrics JSON.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=400000,
        help="Maximum number of rows to use for training (speeds up large datasets).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=120,
        help="Number of trees in the Random Forest.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for model training (-1 uses all cores).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        input_csv=Path(args.input),
        output_model=Path(args.model_output),
        output_metrics=Path(args.metrics_output),
        max_rows=args.max_rows,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
    )
