import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, chi2

B = Path(__file__).resolve().parent
from model_estimation import fit_native, valid, YearDrawMixedLogit
from model_design import long_data, values

O = B / "results" / "combined_test"
O.mkdir(exist_ok=True)
modes = ["cyclist", "escooter"]
specs = {
    m: json.loads((B / "results" / f"{m}_partial.json").read_text()) for m in modes
}


starting_values = json.loads((B / "specifications.json").read_text())


class SharedDrawMixedLogit(YearDrawMixedLogit):
    def _generate_draws(self, sample_size, n_draws, halton=True, halton_opts=None):
        q = int(self._rvidx.sum())
        out = np.zeros((sample_size, q, n_draws))
        a = 0
        for n, cols in zip(self.blocks, self.random_blocks):
            if cols:
                u = super(YearDrawMixedLogit, self)._generate_halton_draws(
                    n, n_draws, len(cols), shuffle=False, drop=100
                )
                for j, c in enumerate(cols):
                    out[a : a + n, c, :] = norm.ppf(u[:, j, :])
            a += n
        return out


def save(m, label, sp, n):
    info = {
        "label": label,
        "ll": float(m.loglikelihood),
        "k": len(m.coeff_),
        "n": n,
        "spec": sp,
        "coefficients": dict(zip(m.coeff_names, map(float, m.coeff_))),
        "gradient_max": float(abs(m.grad_n.sum(0)).max()),
        "convergence": str(m.estimation_message),
    }
    (O / f"{label}.json").write_text(json.dumps(info, indent=2))
    with (O / f"{label}.pkl").open("wb") as f:
        pickle.dump(m, f)
    print(label, info["ll"], info["k"], info["gradient_max"], flush=True)
    return info


def fit(d, sp, start, blocks=None):
    if blocks is None:
        try:
            m = fit_native(
                d, sp, start, inference=False, maxiter=1000, optim_method="BFGS"
            )
            if valid(m, False):
                return m
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass
        return fit_native(
            d, sp, start, inference=False, maxiter=3000, optim_method="L-BFGS-B"
        )
    if not sp["random"]:
        return fit_native(
            d, sp, start, inference=False, maxiter=1000, optim_method="BFGS"
        )
    m = SharedDrawMixedLogit(blocks, [list(range(len(sp["random"]))) for _ in blocks])
    m.fit(
        **long_data(d, sp["terms"]),
        randvars={c: "n" for c in sp["random"]},
        n_draws=1000,
        halton=True,
        halton_opts={"drop": 100, "shuffle": False},
        base_alt=0,
        fit_intercept=False,
        init_coeff=start,
        maxiter=1000,
        random_state=20261009,
        tol_opts={"ftol": 1e-10, "gtol": 1e-5},
        verbose=0,
        batch_size=32,
        mnl_init=False,
        num_hess=False,
        skip_std_errs=True,
        optim_method="BFGS",
    )
    return m


for yi, year in enumerate([2022, 2023, 2024]):
    terms = list(
        dict.fromkeys(
            t for mode in modes for t in specs[mode]["annual_specs"][yi]["terms"]
        )
    )
    random = list(
        dict.fromkeys(
            t for mode in modes for t in specs[mode]["annual_specs"][yi]["random"]
        )
    )
    frames = {mode: pd.read_csv(B / "prepared" / f"{mode}{year}.csv") for mode in modes}
    results = {}
    fitted = {}
    sps = {}
    for mode in modes:
        d = frames[mode]
        active = [t for t in terms if np.any(values(d, t) != 0)]
        sp = {"terms": active, "random": [t for t in active if t in random]}
        sps[mode] = sp
        X = long_data(d, active)["X"][1::2]
        assert np.linalg.matrix_rank(X) == len(active), (mode, year, "rank")
        label = f"{mode}{year}_common"
        old = starting_values[f"{mode}{year}_unrestricted"]["warm_start"]
        names = active + ["sd." + t for t in sp["random"]]
        start = np.array(
            [old.get(t, 0.5 if t.startswith("sd.") else 0.0) for t in names]
        )
        print("FITTING", label, len(d), len(names), flush=True)
        m = fit(d, sp, start)
        assert valid(m, False), (label, m.estimation_message)
        if sp["random"]:
            alt = m.coeff_.copy()
            alt[-len(sp["random"]) :] = 1.5
            m2 = fit(d, sp, alt)
            assert valid(m2, False), (label, "second start", m2.estimation_message)
            if m2.loglikelihood > m.loglikelihood:
                m = m2
        results[mode] = save(m, label, sp, len(d))
    label = f"combined{year}_common"
    d = pd.concat([frames[m] for m in modes], ignore_index=True)
    sp = {"terms": terms, "random": [t for t in terms if t in random]}
    names = terms + ["sd." + t for t in sp["random"]]
    start = np.array(
        [
            sum(results[m]["coefficients"].get(t, 0.0) * len(frames[m]) for m in modes)
            / len(d)
            for t in names
        ]
    )
    print("FITTING", label, len(d), len(names), flush=True)
    m = fit(d, sp, start, [len(frames[m]) for m in modes])
    assert valid(m, False), (label, m.estimation_message)
    if sp["random"]:
        alt = m.coeff_.copy()
        alt[-len(sp["random"]) :] = 1.5
        m2 = fit(d, sp, alt, [len(frames[m]) for m in modes])
        assert valid(m2, False)
        if m2.loglikelihood > m.loglikelihood:
            m = m2
    save(m, label, sp, len(d))

out = {}
for mode in ["cyclist", "escooter", "combined"]:
    rows = [
        json.loads((O / f"{mode}{y}_common.json").read_text())
        for y in [2022, 2023, 2024]
    ]
    out[mode] = {
        "ll": sum(x["ll"] for x in rows),
        "k": sum(x["k"] for x in rows),
        "n": sum(x["n"] for x in rows),
    }
out["LR"] = -2 * (out["combined"]["ll"] - out["cyclist"]["ll"] - out["escooter"]["ll"])
out["df"] = out["cyclist"]["k"] + out["escooter"]["k"] - out["combined"]["k"]
out["p"] = float(chi2.sf(out["LR"], out["df"]))
out["settings"] = {
    "n_draws": 1000,
    "halton": True,
    "drop": 100,
    "shuffle": False,
    "random_distribution": "normal",
    "specification": "within-year union of final partially constrained model terms; no cross-year equalities in this auxiliary test; no significance pruning",
    "zero_columns": "pd_ftpth is structurally zero for bicycles and excluded from their estimable parameter count",
}
(O / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2), flush=True)
