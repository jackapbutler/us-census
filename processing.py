"""Module for cleaning and processing US census data"""
from typing import Tuple
import argparse
import pathlib
import pandas as pd
import sklearn.preprocessing as sk_pre
import sklearn.model_selection as mod_sel

RAW_DATA = "data/raw/us_census.csv"
PROCESSED_DIR = "data/processed"
LABEL = "label"
LABEL_DICT = {" - 50000.": 0, " 50000+.": 1}


def missing_values_report(df: pd.DataFrame):
    """Create missing value report for a DataFrame"""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )

    print(
        "\n There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values. \n",
        mis_val_table_ren_columns,
    )


def encode_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Function for encoding categorical labels according to a mapping dictionary"""
    # encode labels
    print(f"\n Replacing the column {LABEL} according to mapping {LABEL_DICT}")
    df = data.replace({LABEL: LABEL_DICT})
    return df


def one_hot_encode(data: pd.DataFrame) -> pd.DataFrame:
    """Encodes any non-numerical columns via one-hot encoding"""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    non_numeric = data.select_dtypes(exclude=numerics)

    one_hot_cols = list(non_numeric.columns)
    df = pd.get_dummies(data=data, columns=one_hot_cols, dtype=float)

    print("\n One-hot encoded columns; \n", one_hot_cols)
    print("Increased dimensions from ", len(data.columns), " to ", len(df.columns))
    return df


def split_features_and_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Splitting a dataframe into features and labels"""
    # split features and labels
    X = data.drop([LABEL], axis=1)
    y = data[LABEL]

    print("\n Splitting the dataframe into features and labels")
    print("Features: ", list(X.columns))
    print("Label: ", LABEL)
    return X, y


def scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Scale all feature variables"""
    print("\n Scaling feature variables.")
    scaler = sk_pre.StandardScaler()
    scaled_X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return scaled_X


def save_data(
    dir: str, X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, y_te: pd.Series
) -> None:
    """Save a set of training and test data to the data/processing directory"""
    file_path: str = f"{PROCESSED_DIR}/{dir}/"
    output_dir = pathlib.Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n Saving training and testing datasets to {file_path}")

    X_tr.to_csv(file_path + "X_tr.csv", index=False)
    y_tr.to_csv(file_path + "y_tr.csv", index=False)
    X_te.to_csv(file_path + "X_te.csv", index=False)
    y_te.to_csv(file_path + "y_te.csv", index=False)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Provide the tag name for this processed train/test split (to be saved within the data/processed directory).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # read data
    df = pd.read_csv(RAW_DATA)
    missing_values_report(df)

    # encode columns
    df = encode_labels(df)

    # split
    X, y = split_features_and_labels(df)

    # process
    X = one_hot_encode(X)
    X = scale_features(X)
    X_tr, X_te, y_tr, y_te = mod_sel.train_test_split(
        X, y, test_size=0.3, random_state=0, shuffle=False
    )

    # save data
    save_data(args.tag, X_tr, y_tr, X_te, y_te)
