"""Regenerate `data/<name>.parquet` + `data/<name>.yaml` from raw CSVs.

Usage:
    uv run python scripts/prepare_datasets.py path/to/raw/csvs/
    # or, with a default of ./raw_csvs/:
    uv run python scripts/prepare_datasets.py

For each dataset:

  - read raw CSV with the per-dataset recipe (target column, cleaning),
  - encode categorical columns as integer codes using a fixed category order,
  - emit `<name>.parquet` (numerical columns float64; categorical columns int16),
  - emit `<name>.yaml` with target name, n_classes, and per-feature metadata
    (kind, lo/hi for numerical, categories for categorical).

The library only ever reads parquet+yaml; the runtime path never touches CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Recipe types
# ---------------------------------------------------------------------------

class Recipe:
    """A single dataset's prep recipe.

    Subclasses override `read(raw_dir) -> (df, target, categorical_columns)`.
    `df` must already be cleaned (no missing rows). The driver writes parquet
    + yaml from there.
    """

    name: str

    def read(self, raw_dir: Path) -> tuple[pd.DataFrame, str, list[str]]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

class Pima(Recipe):
    name = "pima"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "pima.csv")
        cols = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age",
        ]
        df = df[cols + ["y"]].copy()
        for c in cols[:-2]:  # all int-coded except bmi, dpf
            df[c] = df[c].astype(int)
        df["bmi"] = df["bmi"].astype(float)
        df["diabetes_pedigree_function"] = df["diabetes_pedigree_function"].astype(float)
        df["age"] = df["age"].astype(int)
        df["y"] = df["y"].astype(int)
        return df, "y", []


class Banknote(Recipe):
    name = "banknote"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "banknote.csv")
        df["y"] = df["y"].astype(int)
        return df, "y", []


class Wine(Recipe):
    name = "wine"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "wine.csv", index_col=None)
        # Original task is multiclass {1,2,3}; the reference paper restricts to
        # the binary {1,2} subset. Re-encode to {0,1}.
        df = df[df["y"].isin([1, 2])].copy()
        df["y"] = (df["y"] == 2).astype(int)
        return df, "y", []


class Abalone(Recipe):
    name = "abalone"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "abalone.csv")
        # Binarise rings (target column is named 'y' in this snapshot of the
        # CSV) at the median, matching the reference paper.
        df["y"] = (df["y"] <= df["y"].median()).astype(int)
        return df, "y", ["sex"]


class Ionosphere(Recipe):
    name = "ionosphere"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "ionosphere.csv")
        # Target is {'g','b'} — encode as {0,1}.
        df["y"] = (df["y"] == "g").astype(int)
        return df, "y", []


class Occupancy(Recipe):
    name = "occupancy"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "occupancy.csv", index_col=0)
        cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
        df = df[cols + ["y"]].copy()
        df["y"] = df["y"].astype(int)
        for c in cols:
            df[c] = df[c].astype(float)
        return df, "y", []


class MammographicMasses(Recipe):
    name = "mammographic_masses"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "mammographic_masses.csv")
        df = df[(df == "?").sum(axis=1) == 0].copy()
        for c in df.columns:
            df[c] = df[c].astype(int)
        return df, "y", []


class Compas(Recipe):
    name = "compas"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "compas-scores-two-years.csv")
        df = df.dropna(subset=["days_b_screening_arrest"])
        keep = (
            (df["days_b_screening_arrest"] <= 30)
            & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "NA")
            & ((df["race"] == "African-American") | (df["race"] == "Caucasian"))
        )
        df = df[keep]
        out = pd.DataFrame()
        out["y"] = df["two_year_recid"].astype(int)
        out["AgeGroup"] = df["age_cat"].map(
            {"Less than 25": 1, "25 - 45": 2, "Greater than 45": 3}
        ).astype(int)
        out["Race"] = df["race"].map({"African-American": 0, "Caucasian": 1}).astype(int)
        out["Sex"] = df["sex"].map({"Male": 0, "Female": 1}).astype(int)
        out["PriorsCount"] = df["priors_count"].astype(int)
        out["ChargeDegree"] = df["c_charge_degree"].map({"M": 0, "F": 1}).astype(int)
        return out.reset_index(drop=True), "y", ["AgeGroup", "Race", "Sex", "ChargeDegree"]


class Postoperative(Recipe):
    name = "post_operative"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "post-operative.csv")
        df = df[(df == "?").sum(axis=1) == 0].copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.strip()
        # "comfort" is numeric; the rest categorical (and so is y).
        df["comfort"] = df["comfort"].astype(int)
        df["y"] = df["y"].astype("category").cat.codes.astype(int)
        cat_cols = [c for c in df.columns if c not in ("comfort", "y")]
        return df.reset_index(drop=True), "y", cat_cols


class Seismic(Recipe):
    name = "seismic"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "seismic.csv")
        cat_cols = ["seismic", "seismoacoustic", "shift", "ghazard"]
        df["y"] = df["y"].astype(int)
        for c in cat_cols:
            df[c] = df[c].astype(str)
        for c in df.columns:
            if c not in cat_cols + ["y"]:
                df[c] = df[c].astype(int)
        return df.reset_index(drop=True), "y", cat_cols


class Adult(Recipe):
    name = "adult"

    def read(self, raw_dir):
        df = pd.read_csv(raw_dir / "adult.data")
        df = df.copy()
        for c in df.select_dtypes(include="object"):
            df[c] = df[c].str.strip()
        # Drop noted nuisance columns (sex, race, fnlwgt) per reference recipe.
        for c in ("sex", "race", "fnlwgt"):
            if c in df.columns:
                df = df.drop(columns=[c])
        # Collapse native-country into US / non-US, harmonise education.
        df.loc[df["native-country"] != "United-States", "native-country"] = "Non-United-States"
        edu_collapse = {
            "Preschool": "prim-middle-school",
            "1st-4th": "prim-middle-school",
            "5th-6th": "prim-middle-school",
            "7th-8th": "prim-middle-school",
            "9th": "high-school",
            "10th": "high-school",
            "11th": "high-school",
            "12th": "high-school",
        }
        df["education"] = df["education"].apply(lambda v: edu_collapse.get(v, v))
        out = pd.DataFrame()
        out["y"] = df["y"].astype("category").cat.codes.astype(int)  # <=50K vs >50K → 0/1
        out["age"] = df["age"].astype(int)
        out["NativeCountry"] = df["native-country"].astype(str)
        out["WorkClass"] = df["workclass"].astype(str)
        out["EducationNumber"] = df["education-num"].astype(int)
        out["EducationLevel"] = df["education"].astype(str)
        out["MaritalStatus"] = df["marital-status"].astype(str)
        out["Occupation"] = df["occupation"].astype(str)
        out["Relationship"] = df["relationship"].astype(str)
        out["CapitalGain"] = df["capital-gain"].astype(float)
        out["CapitalLoss"] = df["capital-loss"].astype(float)
        out["HoursPerWeek"] = df["hours-per-week"].astype(int)
        cat_cols = [
            "NativeCountry", "WorkClass", "EducationLevel",
            "MaritalStatus", "Occupation", "Relationship",
        ]
        return out.reset_index(drop=True), "y", cat_cols


class Credit(Recipe):
    name = "credit"

    def read(self, raw_dir):
        raw = pd.read_csv(raw_dir / "credit.data", index_col=0)
        # Convert NTD monetary columns to USD (consistent with the reference
        # recipe in research_paper/dataset_reader.py).
        ntd_to_usd = 32.75
        money = [c for c in raw.columns if any(k in c for k in ("BILL_AMT", "PAY_AMT", "LIMIT_BAL"))]
        for c in money:
            raw[c] = (raw[c] / ntd_to_usd).round(-1).astype(int)

        out = pd.DataFrame()
        out["y"] = (1 - raw["default payment next month (label)"]).astype(int)
        out["isMale"] = (raw["SEX"] == 1).astype(int)
        # MARRIAGE: 1=married, 2=single — drop other (3,0) rows
        keep = raw["MARRIAGE"].isin([1, 2])
        raw = raw[keep]
        out = out[keep.values].copy()
        out["isMarried"] = (raw["MARRIAGE"] == 1).astype(int).values
        # AgeGroup
        ag = pd.Series(0, index=raw.index, dtype=int)
        ag[raw["AGE"] < 25] = 1
        ag[(raw["AGE"] >= 25) & (raw["AGE"] <= 40)] = 2
        ag[(raw["AGE"] > 40) & (raw["AGE"] <= 59)] = 3
        ag[raw["AGE"] >= 60] = 4
        out["AgeGroup"] = ag.values
        # EducationLevel: {others=1, HS=2, Univ=3, Grad=4}
        el = pd.Series(1, index=raw.index, dtype=int)
        el[raw["EDUCATION"] == 3] = 2
        el[raw["EDUCATION"] == 2] = 3
        el[raw["EDUCATION"] == 1] = 4
        out["EducationLevel"] = el.values

        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
        pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
        out["MaxBillAmountOverLast6Months"] = np.maximum(raw[bill_cols].max(axis=1), 0).astype(float).values
        out["MaxPaymentAmountOverLast6Months"] = np.maximum(raw[pay_cols].max(axis=1), 0).astype(float).values
        out["MonthsWithZeroBalanceOverLast6Months"] = np.sum(
            np.greater(raw[pay_cols].values, raw[bill_cols].values), axis=1
        ).astype(int)
        out["MonthsWithLowSpendingOverLast6Months"] = np.sum(
            raw[bill_cols].div(raw["LIMIT_BAL"], axis=0) < 0.20, axis=1
        ).astype(int).values
        out["MonthsWithHighSpendingOverLast6Months"] = np.sum(
            raw[bill_cols].div(raw["LIMIT_BAL"], axis=0) > 0.80, axis=1
        ).astype(int).values
        out["MostRecentBillAmount"] = np.maximum(raw[bill_cols[0]], 0).astype(float).values
        out["MostRecentPaymentAmount"] = np.maximum(raw[pay_cols[0]], 0).astype(float).values

        overdue_cols = [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
        overdue = raw[overdue_cols].replace({-2: 0, -1: 0})
        out["TotalMonthsOverdue"] = overdue.sum(axis=1).astype(int).values
        out["HasHistoryOfOverduePayments"] = (overdue.sum(axis=1) > 0).astype(int).values

        cat_cols = ["isMale", "isMarried", "AgeGroup", "EducationLevel", "HasHistoryOfOverduePayments"]
        return out.reset_index(drop=True), "y", cat_cols


RECIPES: list[Recipe] = [
    Pima(), Banknote(), Wine(), Abalone(), Ionosphere(), Occupancy(),
    MammographicMasses(), Compas(), Postoperative(), Seismic(), Adult(), Credit(),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def write_dataset(
    df: pd.DataFrame,
    target: str,
    categorical_columns: list[str],
    out_dir: Path,
    name: str,
    notes: str = "",
) -> None:
    """Encode categoricals as int codes, write parquet + yaml."""
    feat_specs = []
    enc = pd.DataFrame(index=df.index)
    feature_cols = [c for c in df.columns if c != target]
    for c in feature_cols:
        if c in categorical_columns:
            cats = sorted(df[c].dropna().unique().tolist(), key=lambda v: str(v))
            mapping = {v: i for i, v in enumerate(cats)}
            enc[c] = df[c].map(mapping).astype(np.int16)
            feat_specs.append({
                "name": c,
                "kind": "categorical",
                "categories": [str(v) for v in cats],
            })
        else:
            col = df[c].astype(np.float64)
            lo = float(col.min())
            hi = float(col.max())
            if lo == hi:
                # constant feature — give it a nominal width so the splitter
                # never divides by zero. Records still bound it usefully.
                hi = lo + 1.0
            enc[c] = col.astype(np.float64)
            feat_specs.append({
                "name": c,
                "kind": "numerical",
                "lo": lo,
                "hi": hi,
            })
    # target column
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in df.columns")
    y = df[target].astype(np.int64)
    n_classes = int(y.nunique())
    enc[target] = y.values

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{name}.parquet"
    yaml_path = out_dir / f"{name}.yaml"

    enc.to_parquet(parquet_path, index=False)

    meta = {
        "name": name,
        "target": target,
        "n_classes": n_classes,
        "n_samples": int(len(enc)),
        "features": feat_specs,
    }
    if notes:
        meta["notes"] = notes
    with yaml_path.open("w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    print(f"  -> {parquet_path}  ({len(enc)} rows, {len(feature_cols)} features, {n_classes} classes)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("raw_dir", nargs="?", default="raw_csvs",
                   help="directory containing the raw CSV files")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--only", nargs="*", default=None,
                   help="only build these dataset names")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out)

    if not raw_dir.exists():
        raise SystemExit(f"raw CSV directory not found: {raw_dir}")

    selected = RECIPES
    if args.only:
        wanted = set(args.only)
        selected = [r for r in RECIPES if r.name in wanted]
        missing = wanted - {r.name for r in RECIPES}
        if missing:
            raise SystemExit(f"unknown dataset names: {sorted(missing)}")

    for recipe in selected:
        print(f"[{recipe.name}]")
        try:
            df, target, cats = recipe.read(raw_dir)
        except FileNotFoundError as e:
            print(f"  skip: {e}")
            continue
        write_dataset(df, target, cats, out_dir, recipe.name)


if __name__ == "__main__":
    main()
