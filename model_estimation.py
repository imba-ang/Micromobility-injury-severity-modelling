import json, pickle
import numpy as np
from scipy.stats import norm
from scipy.special import expit
from xlogit import MixedLogit, MultinomialLogit
from model_design import long_data

DRAW_CACHE = {}


class SparseMixedLogit(MixedLogit):
    """Use unchanged native kernels for stochastic and deterministic rows."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_sparse_signature", None)
        state.pop("_sparse_blocks", None)
        return state

    def _loglik_gradient(
        self,
        betas,
        Xd,
        panels,
        draws,
        weights,
        avail,
        scale_d,
        addit_d,
        batch_size,
        return_gradient=True,
    ):
        if panels is not None or scale_d is not None:
            return super()._loglik_gradient(
                betas,
                Xd,
                panels,
                draws,
                weights,
                avail,
                scale_d,
                addit_d,
                batch_size,
                return_gradient,
            )
        signature = (id(Xd), id(draws))
        if getattr(self, "_sparse_signature", None) != signature:
            self._sparse_signature = signature
            self._sparse_blocks = []
            p = Xd.shape[2]
            ri = np.flatnonzero(self._rvidx)
            stochastic = np.any(Xd[:, :, ri] != 0, axis=(1, 2))
            for mask in [stochastic, ~stochastic]:
                rows = np.flatnonzero(mask)
                if not len(rows):
                    continue
                xx = Xd[rows]
                active = np.flatnonzero(np.any(xx != 0, axis=(0, 1)))
                local = np.flatnonzero(np.isin(ri, active))
                kernel = MixedLogit()
                kernel._rvidx = self._rvidx[active].copy()
                kernel._rvdist = np.asarray(self._rvdist)[local]
                indices = np.r_[active, p + local]
                z = (
                    draws[rows][:, local, :]
                    if len(local)
                    else np.zeros((len(rows), 0, 1))
                )
                self._sparse_blocks.append((rows, kernel, indices, xx[:, :, active], z))
        objective = 0.0
        gradient = np.zeros(len(betas))
        scores = np.zeros((len(Xd), len(betas))) if return_gradient else None
        sub = lambda a, rows: None if a is None else a[rows]
        for rows, kernel, indices, xx, z in self._sparse_blocks:
            r = kernel._loglik_gradient(
                betas[indices],
                xx,
                None,
                z,
                sub(weights, rows),
                sub(avail, rows),
                None,
                sub(addit_d, rows),
                batch_size,
                return_gradient,
            )
            if return_gradient:
                v, g, gn = r
                objective += v
                gradient[indices] += g
                scores[np.ix_(rows, indices)] = gn
            else:
                objective += r
        return (objective, gradient, scores) if return_gradient else objective


class YearDrawMixedLogit(MixedLogit):
    def __init__(self, blocks, random_blocks):
        super().__init__()
        self.blocks = blocks
        self.random_blocks = random_blocks

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_native_signature", None)
        state.pop("_native_blocks", None)
        return state

    def _generate_draws(self, sample_size, n_draws, halton=True, halton_opts=None):
        key = (tuple(self.blocks), tuple(tuple(b) for b in self.random_blocks), n_draws)
        if key not in DRAW_CACHE:
            DRAW_CACHE.clear()
            q = sum(len(b) for b in self.random_blocks)
            draws = np.zeros((sample_size, q, n_draws))
            row = 0
            for n, cols in zip(self.blocks, self.random_blocks):
                if cols:
                    u = super()._generate_halton_draws(
                        n, n_draws, len(cols), shuffle=False, drop=100
                    )
                    for local, global_col in enumerate(cols):
                        draws[row : row + n, global_col, :] = norm.ppf(u[:, local, :])
                row += n
            DRAW_CACHE[key] = draws
        return DRAW_CACHE[key]

    def _loglik_gradient(
        self,
        betas,
        Xd,
        panels,
        draws,
        weights,
        avail,
        scale_d,
        addit_d,
        batch_size,
        return_gradient=True,
    ):
        """Sum native year kernels without repeating deterministic draws.

        This is algebraically the native pooled call. Inactive year columns
        are omitted locally and their native gradients mapped back to shared
        coefficient indices. Years without random terms require only one draw.
        """
        if panels is not None or scale_d is not None:
            return super()._loglik_gradient(
                betas,
                Xd,
                panels,
                draws,
                weights,
                avail,
                scale_d,
                addit_d,
                batch_size,
                return_gradient,
            )
        signature = (id(Xd), id(draws))
        if getattr(self, "_native_signature", None) != signature:
            self._native_signature = signature
            self._native_blocks = []
            p = Xd.shape[2]
            random_indices = np.flatnonzero(self._rvidx)
            a = 0
            for n in self.blocks:
                b = a + n
                active = np.flatnonzero(np.any(Xd[a:b] != 0, axis=(0, 1)))
                random_local = np.flatnonzero(np.isin(random_indices, active))
                kernel = SparseMixedLogit()
                kernel._rvidx = self._rvidx[active].copy()
                kernel._rvdist = np.asarray(self._rvdist)[random_local]
                indices = np.r_[active, p + random_local]
                z = (
                    draws[a:b][:, random_local, :]
                    if len(random_local)
                    else np.zeros((n, 0, 1))
                )
                self._native_blocks.append(
                    (a, b, kernel, indices, Xd[a:b][:, :, active], z)
                )
                a = b
        objective = 0.0
        grad = np.zeros(len(betas))
        grad_n = np.zeros((len(Xd), len(betas))) if return_gradient else None
        part = lambda v, a, b: None if v is None else v[a:b]
        for a, b, kernel, indices, xx, z in self._native_blocks:
            result = kernel._loglik_gradient(
                betas[indices],
                xx,
                None,
                z,
                part(weights, a, b),
                part(avail, a, b),
                None,
                part(addit_d, a, b),
                batch_size,
                return_gradient,
            )
            if return_gradient:
                value, g, gn = result
                objective += value
                grad[indices] += g
                grad_n[a:b, indices] = gn
            else:
                objective += result
        return (objective, grad, grad_n) if return_gradient else objective


def observed_information(m, d, spec):
    """Exact derivative of the simulated binary probability for xlogit draws.

    This computes standard errors only. Coefficients, optimization, simulation
    draws and likelihood remain xlogit's. Uses O(NR) work arrays instead of NRK.
    """
    X = long_data(d, spec["terms"])["X"][1::2]
    y = d.severity.to_numpy(float)
    theta = m.coeff_
    p = X.shape[1]
    q = len(spec["random"])
    indices = [spec["terms"].index(c) for c in spec["terms"] if c in spec["random"]]
    H = np.zeros((p + q, p + q))
    if q:
        z = m._generate_draws(len(d), 1000, True, {"drop": 100, "shuffle": False})
    for a in range(0, len(d), 128):
        b = min(a + 128, len(d))
        xx = X[a:b]
        yy = y[a:b]
        zz = z[a:b] * xx[:, indices, None] if q else np.zeros((b - a, 0, 1))
        eta = xx @ theta[:p]
        sim = expit(eta[:, None] + np.einsum("nqr,q->nr", zz, theta[p:]))
        prob = sim.mean(axis=1)
        g0 = sim * (1 - sim)
        g1 = g0 * (1 - 2 * sim)
        dp = np.column_stack(
            [xx * g0.mean(axis=1)[:, None], np.mean(g0[:, None, :] * zz, axis=2)]
        )
        first = (prob - yy) / (prob * (1 - prob))
        second = yy / prob**2 + (1 - yy) / (1 - prob) ** 2
        H += dp.T @ (second[:, None] * dp)
        H[:p, :p] += xx.T @ ((first * g1.mean(axis=1))[:, None] * xx)
        if q:
            cross = xx.T @ (first[:, None] * np.mean(g1[:, None, :] * zz, axis=2))
            H[:p, p:] += cross
            H[p:, :p] += cross.T
            for j in range(q):
                for k in range(q):
                    H[p + j, p + k] += np.sum(
                        first * np.mean(g1 * zz[:, j] * zz[:, k], axis=1)
                    )
    ev = np.linalg.eigvalsh(H)
    m.information_min_eigenvalue = float(ev[0])
    if ev[0] > 1e-8:
        m.covariance = np.linalg.inv(H)
        m.hess_inv = m.covariance
        m.stderr = np.sqrt(np.diag(m.covariance))
        m.zvalues = theta / m.stderr
        m.pvalues = 2 * norm.sf(abs(m.zvalues))
    else:
        m.stderr = np.full(len(theta), np.nan)
        m.zvalues = m.stderr.copy()
        m.pvalues = m.stderr.copy()
        m.covariance = np.full((len(theta), len(theta)), np.nan)
    return m


def fit_native(
    d,
    spec,
    start=None,
    inference=True,
    pooled=None,
    maxiter=500,
    optim_method="L-BFGS-B",
):
    assert all("hm_urbn" not in term.split("*") for term in spec["terms"])
    data = long_data(d, spec["terms"])
    if spec["random"]:
        m = SparseMixedLogit() if pooled is None else YearDrawMixedLogit(*pooled)
        kwargs = dict(
            randvars={c: "n" for c in spec["random"]},
            n_draws=1000,
            halton=True,
            halton_opts={"drop": 100, "shuffle": False},
            batch_size=32,
            mnl_init=start is None,
            num_hess=False,
            skip_std_errs=True,
            optim_method=optim_method,
        )
    else:
        m = MultinomialLogit()
        kwargs = dict(num_hess=inference, skip_std_errs=not inference)
    m.fit(
        **data,
        base_alt=0,
        fit_intercept=False,
        init_coeff=start,
        maxiter=maxiter,
        random_state=20261009,
        tol_opts={"ftol": 1e-10, "gtol": 1e-5},
        verbose=0,
        **kwargs,
    )
    if inference and spec["random"]:
        observed_information(m, d, spec)
    return m


def warm(spec, old):
    previous = dict(zip(old.coeff_names, old.coeff_))
    return np.array(
        [
            previous.get(c, 0.5 if c.startswith("sd.") else 0.0)
            for c in spec["terms"]
            + ["sd." + c for c in spec["terms"] if c in spec["random"]]
        ]
    )


def valid(m, inference=True):
    g = np.max(abs(m.grad_n.sum(axis=0)))
    return (
        np.isfinite(m.coeff_).all()
        and g < 0.02
        and (
            not inference
            or (
                np.isfinite(m.stderr).all()
                and np.isfinite(m.covariance).all()
                and np.linalg.eigvalsh(m.covariance).min() > 0
            )
        )
    )


def probabilities(model, d, spec):
    data = long_data(d, spec["terms"])
    data.pop("y")
    kwargs = dict(return_proba=True, verbose=0)
    if spec["random"]:
        kwargs.update(
            n_draws=1000,
            halton=True,
            halton_opts={"drop": 100, "shuffle": False},
            batch_size=64,
        )
    return model.predict(**data, **kwargs)[1][:, 1]


def load(path):
    info = json.loads(path.with_suffix(".json").read_text())
    with path.with_suffix(".pkl").open("rb") as f:
        model = pickle.load(f)
    return info, model
