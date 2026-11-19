from pathlib import Path
import json
import numpy as np
import pandas as pd
from model_estimation import fit_native, valid
from model_design import pool_design, save_model

B = Path(__file__).resolve().parent
R = B / "results"
keys = [
    f"{mode}{year}" for mode in ("cyclist", "escooter") for year in (2022, 2023, 2024)
]
data = {key: pd.read_csv(B / "prepared" / f"{key}.csv") for key in keys}

specifications = json.loads((B / "specifications.json").read_text())
for name, config in specifications.items():
    mode = config["mode"]
    if "annual_specs" in config:
        specs = config["annual_specs"]
        groups = config["groups"]
        datasets = [data[f"{mode}{y}"] for y in (2022, 2023, 2024)]
        frame, sp, blocks, mapping = pool_design(datasets, specs, groups)
    else:
        sp = config["spec"]
        frame = data[f"{mode}{config['year']}"]
        blocks = None
    names = sp["terms"] + ["sd." + t for t in sp["terms"] if t in sp["random"]]
    start = np.array([config["warm_start"][t] for t in names])
    model = fit_native(
        frame, sp, start, pooled=blocks, maxiter=1000, optim_method="BFGS"
    )
    if not valid(model):
        raise RuntimeError(f"Estimation failed for {name}: {model.estimation_message}")
    info = {k: v for k, v in config.items() if k != "warm_start"}
    save_model(model, R / name, dict(info, spec=sp, n=len(frame)))
