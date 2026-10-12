"""Media cost: the linear model of Cosenza et al. (2022), Table 1.

cost = sum over components of (coefficient x scaled amount) + fixed cost of the DMEM base.
A coefficient is the relative cost of adding that component at its maximum amount (unitless).
"""

import pandas as pd

from .data import COLS, MAX_CONC

COEF = pd.Series({
    "T": 6.53e-3, "I": 1.43e-2, "SS": 6.4e-9, "AA": 9.8e-6, "Glu": 0.2, "Gluta": 2.09e-2, "Albu": 4.94,
    "FBS": 14.0, "H": 1.1e-5, "D": 7.2e-3, "P": 4.0e-7, "Esd": 1.6e-6, "Ethan": 6.1e-6, "Glutath": 6.0e-4,
})
BASE_COST = 0.021  # DMEM base present in every medium

# Commercial baseline: DMEM + 10 % FBS (Table 3), converted to the scaled 0-1 units.
_baseline_conc = pd.Series(0.0, index=COLS)
_baseline_conc[["Glu", "Gluta", "FBS"]] = [4.50, 0.43, 10.0]
BASELINE = (_baseline_conc / MAX_CONC).to_frame("baseline").T


def media_cost(df: pd.DataFrame) -> pd.Series:
    """Cost of each recipe (row)."""
    return df[COLS] @ COEF[COLS] + BASE_COST


def baseline_cost() -> float:
    """Cost of the commercial baseline: our cost ceiling."""
    return float(media_cost(BASELINE).iloc[0])
