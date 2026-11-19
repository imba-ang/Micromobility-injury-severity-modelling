import json, pickle
import numpy as np
import pandas as pd


def values(d, c):
    if c == "constant":
        return np.ones(len(d))
    a = np.ones(len(d))
    for key in c.split("*"):
        a *= d[key].to_numpy(float)
    return a


def long_data(d, terms):
    X = np.zeros((2 * len(d), len(terms)))
    X[1::2] = np.column_stack([values(d, c) for c in terms])
    y = d.severity.to_numpy(int)
    return dict(
        X=X,
        y=np.column_stack([1 - y, y]).ravel(),
        varnames=terms,
        alts=np.tile([0, 1], len(d)),
        ids=np.repeat(np.arange(len(d)), 2),
    )


def save_model(model, path, extra):
    data = {
        **extra,
        "coefficients": dict(zip(model.coeff_names, map(float, model.coeff_))),
        "se": dict(zip(model.coeff_names, map(float, model.stderr))),
        "t": dict(zip(model.coeff_names, map(float, model.zvalues))),
        "ll": float(model.loglikelihood),
        "k": len(model.coeff_),
        "convergence": bool(model.convergence),
        "message": str(model.estimation_message),
        "iterations": int(model.total_iter),
        "gradient_max": float(abs(model.grad_n.sum(axis=0)).max()),
        "covariance_min_eigenvalue": float(np.linalg.eigvalsh(model.covariance).min()),
        "seconds": float(model.estim_time_sec),
    }
    path.with_suffix(".json").write_text(json.dumps(data, indent=2))
    with path.with_suffix(".pkl").open("wb") as f:
        pickle.dump(model, f)
    return data


def pool_design(data, specs, groups):
    n = sum(map(len, data))
    columns = {}
    random = []
    mapping = {}
    offset = 0
    for i, (d, spec) in enumerate(zip(data, specs)):
        for term in spec["terms"]:
            name = groups[i][term]
            if name not in columns:
                columns[name] = np.zeros(n)
            columns[name][offset : offset + len(d)] = values(d, term)
            mapping[(i, term)] = name
            if term in spec["random"] and name not in random:
                random.append(name)
        offset += len(d)
    frame = pd.DataFrame(columns)
    frame["severity"] = np.concatenate([d.severity for d in data])
    terms = list(columns)
    random_order = [c for c in terms if c in random]
    blocks = []
    for i, s in enumerate(specs):
        blocks.append(
            [random_order.index(groups[i][c]) for c in s["terms"] if c in s["random"]]
        )
    return (
        frame,
        {"terms": terms, "random": random_order},
        ([len(d) for d in data], blocks),
        mapping,
    )
