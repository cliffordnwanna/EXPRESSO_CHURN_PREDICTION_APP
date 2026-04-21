import argparse
from pathlib import Path

import pandas as pd


def clean_dataset(input_csv: Path, output_csv: Path):
    print(f"Loading dataset from: {input_csv}")
    df = pd.read_csv(input_csv, encoding="latin1")

    print("Initial dataset shape:", df.shape)
    df = df.drop_duplicates().reset_index(drop=True)

    required_columns = ["CHURN"]
    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from dataset: {missing_required}")

    df = df.dropna(subset=["CHURN"])

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            mode_values = df[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "UNKNOWN"
            df[column] = df[column].fillna(fill_value)
        else:
            df[column] = df[column].fillna(df[column].median(skipna=True))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("Cleaned dataset shape:", df.shape)
    print(f"Cleaned dataset saved to: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Clean churn dataset for model training.")
    parser.add_argument(
        "--input",
        default="DATA/Expresso_churn_dataset.csv",
        help="Path to raw CSV dataset.",
    )
    parser.add_argument(
        "--output",
        default="DATA/cleaned_dataset.csv",
        help="Path to cleaned CSV output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_dataset(input_csv=Path(args.input), output_csv=Path(args.output))
