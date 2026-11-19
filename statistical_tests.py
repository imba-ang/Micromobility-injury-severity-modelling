from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from model_estimation import load, probabilities
from model_design import pool_design

B = Path(__file__).resolve().parent
R = B / "results"
keys = [
    f"{mode}{year}" for mode in ("cyclist", "escooter") for year in (2022, 2023, 2024)
]
data = {key: pd.read_csv(B / "prepared" / f"{key}.csv") for key in keys}
fits = {name: load(R / f"{name}_annual") for name in keys}
samples = data
out = R
records = []
for target, d in samples.items():
    for source, (info, model) in fits.items():
        p = probabilities(model, d, info["spec"])
        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
        ll = float(np.log(np.where(d.severity == 1, p, 1 - p)).sum())
        own = fits[target][0]["ll"]
        if source == target:
            assert abs(ll - own) < 1e-5, (target, ll, own)
            continue
        stat = 2 * (own - ll)
        records.append(
            dict(
                source=source,
                target=target,
                n=len(d),
                df=info["k"],
                own_LL=own,
                transferred_LL=ll,
                LR=stat,
                confidence=chi2.cdf(stat, info["k"]) if stat >= 0 else None,
            )
        )
    print("Verified", target, flush=True)
assert len(records) == 30
pd.DataFrame(records).to_csv(out / "table3.csv", index=False)
print("Range", min(r["LR"] for r in records), max(r["LR"] for r in records))

comparisons = []
for y in [2022, 2023, 2024]:
    comparisons.append(
        (f"escooter{y}", f"cyclist{y}", "cross_mode", f"Transferability {y}")
    )
for mode in ["escooter", "cyclist"]:
    for a, b in [(2022, 2023), (2022, 2024), (2023, 2024)]:
        comparisons.append(
            (
                f"{mode}{a}",
                f"{mode}{b}",
                "temporal",
                f"Temporal prediction {'e-scooter' if mode == 'escooter' else 'bicycle'} {a} to {b}",
            )
        )
summary = []
allrows = []
for i, (source, target, kind, title) in enumerate(comparisons):
    o, m = fits[source]
    own, om = fits[target]
    d = data[target]
    p = probabilities(m, d, o["spec"])
    ownp = probabilities(om, d, own["spec"])
    delta = p - ownp
    allrows.append(
        pd.DataFrame(
            {
                "panel": chr(97 + i),
                "source_model": source,
                "target_sample": target,
                "observation": np.arange(1, len(d) + 1),
                "transferred_probability": p,
                "own_probability": ownp,
                "difference": delta,
            }
        )
    )
    summary.append(
        {
            "panel": chr(97 + i),
            "source": source,
            "target": target,
            "type": kind,
            "title": title,
            "n": len(d),
            "mean": float(delta.mean()),
            "sd": float(delta.std(ddof=1)),
            "min": float(delta.min()),
            "max": float(delta.max()),
            "positive_fraction": float(np.mean(delta > 0)),
        }
    )
pd.concat(allrows).to_csv(R / "figure3_observation_differences.csv", index=False)
(R / "figure3_summary.json").write_text(json.dumps(summary, indent=2))

print([(v["panel"], v["min"], v["max"]) for v in summary])
for mode in ("cyclist", "escooter"):
    o, m = load(R / f"{mode}_partial")
    data_mode = [data[f"{mode}{year}"] for year in (2022, 2023, 2024)]
    annual = [
        json.loads((R / f"{mode}{year}_unrestricted.json").read_text())
        for year in (2022, 2023, 2024)
    ]
    lr = 2 * (sum(a["ll"] for a in annual) - o["ll"])
    df = sum(a["k"] for a in annual) - o["k"]
    (R / f"{mode}_temporal_test.json").write_text(
        json.dumps(dict(LR=lr, df=df, p=chi2.sf(lr, df)), indent=2)
    )
    specs = o["annual_specs"]
    groups = o["groups"]
    frame, sp, blocks, mapping = pool_design(data_mode, specs, groups)
    rows = []
    for name, b, se, t in zip(m.coeff_names, m.coeff_, m.stderr, m.zvalues):
        sd = name.startswith("sd.")
        base = name[3:] if sd else name
        active = [(i, term) for (i, term), v in mapping.items() if v == base]
        term = active[0][1]
        effect = np.nan
        pe = np.nan
        if not sd and term != "constant" and "*" not in term:
            hi = [d.copy() for d in data_mode]
            lo = [d.copy() for d in data_mode]
            for j, _ in active:
                hi[j][term] = 1.0
                lo[j][term] = 0.0
            fp, _, _, _ = pool_design(hi, specs, groups)
            fm, _, _, _ = pool_design(lo, specs, groups)
            delta = probabilities(m, fp, sp) - probabilities(m, fm, sp)
            effect = float(delta.mean())
            present = np.concatenate(
                [
                    d[term].to_numpy() if (j, term) in active else np.zeros(len(d))
                    for j, d in enumerate(data_mode)
                ]
            )
            pe = float(np.mean(delta * present))
        rows.append(
            dict(
                parameter=("sd." if sd else "") + term,
                years=",".join(str(2022 + j) for j, _ in active),
                estimate=float(b),
                se=float(se),
                t=float(t),
                marginal_effect=effect,
                prevalence_weighted_effect=pe,
            )
        )
    pd.DataFrame(rows).to_csv(
        R / ("table4.csv" if mode == "cyclist" else "table5.csv"), index=False
    )

for name, (info, model) in fits.items():
    d = data[name]
    rows = []
    for term in model.coeff_names:
        effect = None
        if term != "constant" and not term.startswith("sd.") and "*" not in term:
            high = d.assign(**{term: 1.0})
            low = d.assign(**{term: 0.0})
            effect = float(
                (
                    probabilities(model, high, info["spec"])
                    - probabilities(model, low, info["spec"])
                ).mean()
            )
        rows.append(
            dict(
                parameter=term,
                estimate=info["coefficients"][term],
                se=info["se"][term],
                t=info["t"][term],
                marginal_effect=effect,
            )
        )
    pd.DataFrame(rows).to_csv(R / f"{name}_appendix.csv", index=False)
