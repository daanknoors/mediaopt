"""Load the Cosenza et al. (2022) culture-media data.

The published files have no column names. Inputs are the 14 components scaled to 0-1
(0 = none, 1 = maximum concentration in Table 1 of the paper). Outputs are standardised
(mean 0, sd 1): higher = more cells, but original units are lost.
"""

from pathlib import Path

import pandas as pd

URL = ("https://raw.githubusercontent.com/ZacharyCosenza/GradStuff_Cosenza/"
       "18e1173b8370a3981e4ee945028cb939465d8011/DBO_Data/")  # pinned version
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cosenza2022" / "raw"

# Column order of the file = Table 1 order: (abbreviation, name, max concentration, unit)
COMPONENTS = [
    ("T", "Transferrin", 0.026, "mg/ml"),
    ("I", "Insulin", 0.035, "mg/ml"),
    ("SS", "Sodium selenite", 1.75e-5, "mg/ml"),
    ("AA", "Ascorbic acid", 8.75e-3, "mg/ml"),
    ("Glu", "Glucose", 15.75, "mg/ml"),
    ("Gluta", "Glutamine", 1.519, "mg/ml"),
    ("Albu", "Albumin", 1.4, "mg/ml"),
    ("FBS", "Fetal bovine serum", 17.5, "% v/v"),
    ("H", "Hydrocortisone", 1.75e-5, "mg/ml"),
    ("D", "Dexamethasone", 7.0e-4, "mg/ml"),
    ("P", "Progesterone", 1.75e-5, "mg/ml"),
    ("Esd", "Estradiol", 8.75e-6, "mg/ml"),
    ("Ethan", "Ethanolamine", 6.65e-3, "mg/ml"),
    ("Glutath", "Glutathione", 3.5e-3, "mg/ml"),
]
COLS = [c[0] for c in COMPONENTS]
MAX_CONC = pd.Series({c[0]: c[2] for c in COMPONENTS})


def _read(name: str) -> pd.DataFrame:
    """Read a file from the local cache, downloading it the first time."""
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(URL + name, sep="\t", header=None).to_csv(path, sep="\t", header=False, index=False)
    return pd.read_csv(path, sep="\t", header=None)


def load_recipes() -> pd.DataFrame:
    """The 25 recipes measured with the expensive, reliable assay (Passage-2 cell count).

    Columns: the 14 scaled components, y (cell count, standardised), y_var (replicate variance).
    """
    X, Y = _read("BO_data.txt"), _read("BO_outputs.txt")
    df = X.iloc[:, :14].set_axis(COLS, axis=1)
    df["y"], df["y_var"] = Y[0], Y[1]
    is_passage2 = X[14] == 0  # last column = assay code; 0 = Passage-2 count
    return df[is_passage2].reset_index(drop=True)


def to_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Scaled 0-1 values -> lab units (Table 1)."""
    return df[COLS] * MAX_CONC
