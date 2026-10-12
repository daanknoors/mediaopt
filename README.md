# Cost-aware culture-media experiment recommendation

Technical assessment: recommend the next 3-5 media formulations to test, maximising cell count under a cost ceiling.

## Structure

```
src/mediaopt/
  data.py          load the Cosenza et al. (2022) data (25 recipes, expensive assay)
  cost.py          the paper's linear cost model; the baseline medium (cost ceiling)
  model.py         average, random forest and Gaussian process; leave-one-out evaluation
  acquisition.py   candidates, Expected Improvement, batch selection, recommendation
notebooks/
  01_data.ipynb        the data and its limits
  02_objective.ipynb   objective and cost ceiling
  03_models.ipynb      model comparison
  04_scoring.ipynb     candidates and Expected Improvement
  05_batch.ipynb       the recommended batch vs a random batch
  06_ceiling.ipynb     sensitivity to the cost ceiling
docs/
  Qorium - Technical Case Study.pdf   technical memo
```

## Setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for Python and packages. The Python version is pinned in `.python-version` (3.12), and exact package versions are pinned in `uv.lock`.

```bash
# 1. Install uv once (macOS); alternatively: brew install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the project folder: install Python 3.12 if needed, create .venv,
#    and install the locked dependencies plus this package (editable)
uv sync

# 3. Open the notebooks
uv run jupyter lab notebooks/
```

Run all notebooks end to end (headless):

```bash
``uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb``
```

Day-to-day commands:

| Task | Command |
|---|---|
| Add a runtime dependency | `uv add <package>` |
| Add a notebook/dev-only dependency | `uv add --dev <package>` |
| Run any command in the environment | `uv run <command>` (e.g. `uv run python`) |
| Update the lockfile after editing `pyproject.toml` | `uv lock` |

In VS Code or another editor, select `.venv/bin/python` as the interpreter or kernel.

The Cosenza data is downloaded automatically on first use from
[ZacharyCosenza/GradStuff_Cosenza](https://github.com/ZacharyCosenza/GradStuff_Cosenza)
(commit `18e1173`) and cached in `data/cosenza2022/raw/`. It is not redistributed in this repo.

## Data source

Cosenza, Z., Astudillo, R., Frazier, P. I., Baar, K., & Block, D. E. (2022).
Multi-information source Bayesian optimization of culture media for cellular agriculture.
*Biotechnology and Bioengineering*, 119(9), 2447-2458. https://doi.org/10.1002/bit.28132
