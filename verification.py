"""Deterministic numerical verification for the repaired manuscript.

Requires NumPy, SciPy, and Matplotlib. Running this file writes sections/numerics.tex,
writes sections/numeric_supplement.tex, and regenerates every external PDF and EPS figure.
It includes multi-density construction benchmarks, forward W1 reduction, Monte Carlo and
Sobol baselines, and six-item inventory applications with up to seven stages. All reported
tables are generated from the computations below; assertions check the theorem identities,
certificates, reference-grid stability, and Bellman decompositions. The COR_USE_CACHE=1
mode is solely for rebuilding a document from existing local outputs; ordinary execution
recomputes everything and needs no cache files.
"""
from pathlib import Path
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar, differential_evolution

# ===== COR-focused computational extension =====
import math
import time
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, minimize_scalar, linprog
from scipy.special import ndtr
from scipy.stats import qmc


class Distribution:
    def __init__(self, name, left=0.0, right=10.0):
        self.name, self.A, self.B = name, left, right

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        flat = u.ravel()
        out = np.array([brentq(lambda x: self.cdf(x) - float(v), self.A, self.B,
                               xtol=2e-12, rtol=8.9e-16) for v in flat])
        return out.reshape(u.shape)

    def hinge(self, a):
        """E[(X-a)+], vectorized over a."""
        a = np.asarray(a, dtype=float)
        return np.vectorize(self._hinge_scalar)(a)

    def _hinge_scalar(self, a):
        if a <= self.A:
            return self.mean - a
        if a >= self.B:
            return 0.0
        return self.mean - a + quad(self.cdf, self.A, a, epsabs=2e-11,
                                    epsrel=2e-11, limit=100)[0]

    def moment(self, x):
        """Integral from A to x of t f(t) dt."""
        if x <= self.A:
            return 0.0
        if x >= self.B:
            return self.mean
        return quad(lambda t: t * self.pdf(t), self.A, x, epsabs=2e-11,
                    epsrel=2e-11, limit=100)[0]


class UniformDist(Distribution):
    def __init__(self):
        super().__init__('Uniform')
        self.mean = 5.0

    def pdf(self, x):
        return 0.1

    def cdf(self, x):
        return np.clip(x / 10.0, 0.0, 1.0)

    def ppf(self, u):
        return 10.0 * np.asarray(u)

    def _hinge_scalar(self, a):
        if a <= 0:
            return 5.0 - a
        if a >= 10:
            return 0.0
        return (10.0 - a) ** 2 / 20.0

    def moment(self, x):
        return np.clip(x, 0.0, 10.0) ** 2 / 20.0


class SineDist(Distribution):
    def __init__(self):
        super().__init__('Sine')
        self.k, self.w = 0.3, 0.6 * math.pi
        self.mean = 5.0 - self.k / self.w

    def pdf(self, x):
        return (1.0 + self.k * math.sin(self.w * x)) / 10.0

    def cdf(self, x):
        x = min(10.0, max(0.0, x))
        return (x + self.k * (1.0 - math.cos(self.w * x)) / self.w) / 10.0

    def ppf(self, u):
        return super().ppf(u)

    def _hinge_scalar(self, a):
        if a <= 0:
            return self.mean - a
        if a >= 10:
            return 0.0
        intcdf = a * a / 20.0 + self.k / (10.0 * self.w) * (
            a - math.sin(self.w * a) / self.w)
        return self.mean - a + intcdf

    def moment(self, x):
        x = min(10.0, max(0.0, x))
        return (0.5 * x * x + self.k * (-x * math.cos(self.w * x) / self.w
                + math.sin(self.w * x) / self.w ** 2)) / 10.0


class TruncNormal(Distribution):
    def __init__(self, mu, sigma, name):
        super().__init__(name)
        self.mu, self.sigma = mu, sigma
        self.alpha, self.beta = (self.A - mu) / sigma, (self.B - mu) / sigma
        self.Z = ndtr(self.beta) - ndtr(self.alpha)
        self.mean = mu + sigma * (self._phi(self.alpha) - self._phi(self.beta)) / self.Z

    @staticmethod
    def _phi(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def pdf(self, x):
        if x < self.A or x > self.B:
            return 0.0
        return self._phi((x - self.mu) / self.sigma) / (self.sigma * self.Z)

    def cdf(self, x):
        if x <= self.A:
            return 0.0
        if x >= self.B:
            return 1.0
        return (ndtr((x - self.mu) / self.sigma) - ndtr(self.alpha)) / self.Z

    def _hinge_scalar(self, a):
        if a <= self.A:
            return self.mean - a
        if a >= self.B:
            return 0.0
        aa, bb = (a - self.mu) / self.sigma, self.beta
        numerator = self.sigma * (self._phi(aa) - self._phi(bb)) + (self.mu - a) * (
            ndtr(bb) - ndtr(aa))
        return numerator / self.Z

    def moment(self, x):
        if x <= self.A:
            return 0.0
        if x >= self.B:
            return self.mean
        xx = (x - self.mu) / self.sigma
        return (self.mu * (ndtr(xx) - ndtr(self.alpha)) + self.sigma * (
            self._phi(self.alpha) - self._phi(xx))) / self.Z


class MixtureDist(Distribution):
    def __init__(self):
        super().__init__('Bimodal')
        self.components = [TruncNormal(2.2, 0.85, 'component'),
                           TruncNormal(7.3, 1.15, 'component')]
        self.weights = [0.42, 0.58]
        self.mean = sum(w * d.mean for w, d in zip(self.weights, self.components))

    def pdf(self, x):
        return sum(w * d.pdf(x) for w, d in zip(self.weights, self.components))

    def cdf(self, x):
        return sum(w * d.cdf(x) for w, d in zip(self.weights, self.components))

    def _hinge_scalar(self, a):
        return sum(w * d._hinge_scalar(a) for w, d in zip(self.weights, self.components))

    def moment(self, x):
        return sum(w * d.moment(x) for w, d in zip(self.weights, self.components))


def integrate_dist(dist, func, a, b, eps=2e-11):
    return quad(lambda x: func(x) * dist.pdf(x), a, b, epsabs=eps,
                epsrel=eps, limit=150)[0]


def cell_data_dist(dist, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        p = dist.cdf(b) - dist.cdf(a)
        m = dist.moment(b) - dist.moment(a)
        c = m / p
        g = 0.5 * (integrate_dist(dist, lambda x: c - x, a, c)
                   + integrate_dist(dist, lambda x: x - c, c, b))
        second = integrate_dist(dist, lambda x: (x - c) ** 2, a, b)
        out.append(dict(a=a, b=b, p=p, c=c, g=g, second=second))
    return out


def equal_mass_dist(dist, cells, power):
    total = integrate_dist(dist, lambda x: dist.pdf(x) ** (power - 1.0),
                           dist.A, dist.B)
    target = total / cells
    edges = [dist.A]
    for _ in range(cells - 1):
        a = edges[-1]
        edges.append(brentq(lambda x: integrate_dist(
            dist, lambda y: dist.pdf(y) ** (power - 1.0), a, x) - target,
            a, dist.B, xtol=2e-11, rtol=8.9e-15))
    edges.append(dist.B)
    return np.array(edges)


def single_g_dist(dist, a, b):
    if b - a < 1e-12:
        return 0.0
    return cell_data_dist(dist, [a, b])[0]['g']


def optimal_tree_dist(dist, cells, iterations=46):
    if cells == 1:
        return np.array([dist.A, dist.B])

    def maximal(a, level):
        if single_g_dist(dist, a, dist.B) <= level:
            return dist.B
        return brentq(lambda b: single_g_dist(dist, a, b) - level,
                      a, dist.B, xtol=3e-11, rtol=8.9e-15)

    def reachable(level):
        a = dist.A
        for _ in range(cells):
            a = maximal(a, level)
            if a >= dist.B - 2e-8:
                return True
        return False

    high = max(z['g'] for z in cell_data_dist(dist, np.linspace(dist.A, dist.B, cells + 1)))
    low = 0.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if reachable(mid):
            high = mid
        else:
            low = mid
    edges, a = [dist.A], dist.A
    for _ in range(cells - 1):
        a = maximal(a, high)
        edges.append(a)
    edges.append(dist.B)
    return np.array(edges)


def cm_atoms(dist, edges):
    cells = cell_data_dist(dist, edges)
    return np.array([z['c'] for z in cells]), np.array([z['p'] for z in cells])


def transport_w1(dist, atoms, probs=None):
    """Exact one-dimensional W1, including arbitrary finite-support weights.

    Monotone optimal transport assigns consecutive quantile intervals with the
    prescribed atom probabilities. If probs is omitted, nearest-representative
    reassignment determines those probabilities.
    """
    atoms = np.asarray(atoms, dtype=float)
    order = np.argsort(atoms)
    atoms = atoms[order]
    if probs is None:
        edges = np.r_[dist.A, 0.5 * (atoms[:-1] + atoms[1:]), dist.B]
    else:
        probs = np.asarray(probs, dtype=float)[order]
        assert len(probs) == len(atoms) and np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 2e-10
        cumulative = np.cumsum(probs)[:-1]
        edges = np.r_[dist.A, dist.ppf(cumulative), dist.B]
    total = 0.0
    for z, a, b in zip(atoms, edges[:-1], edges[1:]):
        lo, hi = max(a, z), min(b, z)
        if a < lo:
            total += z * (dist.cdf(lo) - dist.cdf(a)) - (dist.moment(lo) - dist.moment(a))
        if hi < b:
            total += (dist.moment(b) - dist.moment(hi)) - z * (dist.cdf(b) - dist.cdf(hi))
    return total


def forward_reduction(dist, cells, grid_size=768):
    """Forward scenario selection under L1 distance with probability reassignment."""
    grid = dist.ppf((np.arange(grid_size) + 0.5) / grid_size)
    weights = np.full(grid_size, 1.0 / grid_size)
    distances = np.abs(grid[:, None] - grid[None, :])
    current = np.full(grid_size, np.inf)
    selected = []
    for _ in range(cells):
        # Rows are candidate representatives and columns are sampled points.
        candidate_objectives = np.minimum(current[None, :], distances) @ weights
        j = int(np.argmin(candidate_objectives))
        selected.append(j)
        current = np.minimum(current, distances[:, j])
    atoms = grid[selected]
    assignment = np.argmin(np.abs(grid[:, None] - atoms[None, :]), axis=1)
    probs = np.array([weights[assignment == j].sum() for j in range(cells)])
    order = np.argsort(atoms)
    return atoms[order], probs[order]


def make_tree(dist, method, cells, seed=0):
    if method == 'Finite-B optimal':
        edges = optimal_tree_dist(dist, cells)
        atoms, probs = cm_atoms(dist, edges)
        return atoms, probs, 'CM', max(z['g'] for z in cell_data_dist(dist, edges))
    if method == 'Equal width':
        edges = np.linspace(dist.A, dist.B, cells + 1)
        atoms, probs = cm_atoms(dist, edges)
        return atoms, probs, 'CM', max(z['g'] for z in cell_data_dist(dist, edges))
    if method == 'Equal probability':
        edges = equal_mass_dist(dist, cells, 1.0)
        atoms, probs = cm_atoms(dist, edges)
        return atoms, probs, 'CM', max(z['g'] for z in cell_data_dist(dist, edges))
    if method == 'Equal sqrt-f':
        edges = equal_mass_dist(dist, cells, 0.5)
        atoms, probs = cm_atoms(dist, edges)
        return atoms, probs, 'CM', max(z['g'] for z in cell_data_dist(dist, edges))
    if method == 'Equal f^(1/3)':
        edges = equal_mass_dist(dist, cells, 1.0 / 3.0)
        atoms, probs = cm_atoms(dist, edges)
        return atoms, probs, 'CM', max(z['g'] for z in cell_data_dist(dist, edges))
    if method == 'Forward W1':
        atoms, probs = forward_reduction(dist, cells)
        return atoms, probs, 'transport', float('nan')
    if method == 'Monte Carlo':
        rng = np.random.default_rng(seed)
        atoms = np.sort(dist.ppf(rng.random(cells)))
        return atoms, np.full(cells, 1.0 / cells), 'transport', float('nan')
    if method == 'Sobol':
        sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
        atoms = np.sort(dist.ppf(sampler.random_base2(
            m=int(math.ceil(math.log2(cells))))[:cells].ravel()))
        return atoms, np.full(cells, 1.0 / cells), 'transport', float('nan')
    raise ValueError(method)


class PLFunction:
    """Convex piecewise-linear function, linearly extended beyond its knots."""
    def __init__(self, knots=None, values=None):
        if knots is None:
            self.x = np.array([0.0])
            self.y = np.array([0.0])
            self.left_slope = self.right_slope = 0.0
        else:
            self.x = np.asarray(knots, dtype=float)
            self.y = np.asarray(values, dtype=float)
            self.left_slope = (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
            self.right_slope = (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        out = np.interp(z, self.x, self.y)
        left = z < self.x[0]
        right = z > self.x[-1]
        out = np.where(left, self.y[0] + self.left_slope * (z - self.x[0]), out)
        out = np.where(right, self.y[-1] + self.right_slope * (z - self.x[-1]), out)
        return float(out) if out.ndim == 0 else out

    def minimizer_interval(self, slope):
        """Minimizers of slope*z + self(z)."""
        vals = slope * self.x + self.y
        best = vals.min()
        ids = np.flatnonzero(np.abs(vals - best) <= 2e-10 * max(1.0, abs(best)))
        return float(self.x[ids[0]]), float(self.x[ids[-1]])


def expected_hinge_tree(z, atoms, probs, r, h):
    return r * np.sum(probs * np.maximum(atoms - z, 0.0)) + h * np.sum(
        probs * np.maximum(z - atoms, 0.0))


def procurement_transform(K, order_cost, capacity):
    """H(y)=min_{0<=u<=U} order_cost*u+K(y+u), exactly for convex PL K."""
    zl, zr = K.minimizer_interval(order_cost)
    left_boundary, right_boundary = zl - capacity, zr
    knots = [left_boundary, right_boundary]
    knots += [float(z - capacity) for z in K.x if z < zl - 1e-9]
    knots += [float(z) for z in K.x if z > zr + 1e-9]
    knots = np.array(sorted(set(knots)))

    def raw(y):
        candidates = np.r_[y, y + capacity, np.clip(K.x, y, y + capacity)]
        vals = order_cost * (candidates - y) + K(candidates)
        return np.min(vals)

    vals = np.array([raw(float(y)) for y in knots])
    H = PLFunction(knots, vals)
    # Cache the transform data for exact policy recovery.
    H.zl, H.zr, H.order_cost, H.capacity, H.K = zl, zr, order_cost, capacity, K
    return H


def procurement_policy(H, y):
    candidates = np.r_[0.0, H.capacity, np.clip(H.K.x - y, 0.0, H.capacity)]
    vals = H.order_cost * candidates + H.K(y + candidates)
    return float(candidates[int(np.argmin(vals))])


def next_K(H, atoms, probs, r, h):
    knots = np.concatenate([atoms] + [atoms + k for k in H.x])
    knots = np.array(sorted(set(np.round(knots, 11))))
    vals = np.array([expected_hinge_tree(z, atoms, probs, r, h)
                     + np.sum(probs * H(z - atoms)) for z in knots])
    return PLFunction(knots, vals)


class InventoryInstance:
    def __init__(self, name, dist, horizon, seed=0):
        self.name, self.dist, self.T = name, dist, horizon
        rng = np.random.default_rng(seed)
        self.n = 6
        self.loads = np.array([0.55, 0.75, 0.9, 1.05, 1.25, 1.45])
        self.r = np.array([[5.0 + 0.35 * j + 0.15 * (t % 2)
                            for j in range(self.n)] for t in range(2, self.T + 1)])
        self.h = np.array([[0.55 + 0.12 * j + 0.08 * ((t + 1) % 3)
                            for j in range(self.n)] for t in range(2, self.T + 1)])
        self.order = np.array([[0.75 + 0.08 * j + 0.025 * t
                                for j in range(self.n)] for t in range(2, self.T)])
        self.initial_cost = 0.9 + 0.07 * np.arange(self.n)
        self.capacity = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        self.shared_capacity = 34.0 if seed == 0 else (31.0 if seed == 1 else 37.0)
        self.L = np.zeros((self.T - 1, self.n))
        running = np.zeros(self.n)
        for ti in range(self.T - 2, -1, -1):
            running = running + self.r[ti] + self.h[ti]
            self.L[ti] = running

    def scaled_atoms(self, atoms, probs, j):
        return self.loads[j] * atoms, probs

    def continuous_ell(self, t, j, state):
        d = self.loads[j]
        hinge = self.dist._hinge_scalar(float(state / d))
        return d * (self.r[t, j] + self.h[t, j]) * hinge + self.h[t, j] * (
            state - d * self.dist.mean)

    def tree_dp(self, atoms, probs):
        all_H, all_K = [], []
        for j in range(self.n):
            z, q = self.scaled_atoms(atoms, probs, j)
            K = PLFunction(z, [expected_hinge_tree(float(y), z, q,
                                                   self.r[-1, j], self.h[-1, j]) for y in z])
            # Terminal K must include the linear tails, not only values at demand atoms.
            terminal_knots = np.r_[z[0] - 1.0, z, z[-1] + 1.0]
            K = PLFunction(terminal_knots, [expected_hinge_tree(float(y), z, q,
                                                               self.r[-1, j], self.h[-1, j])
                                            for y in terminal_knots])
            Hs = [None] * (self.T - 1)  # H_2,...,H_T; H_T=0
            Ks = [None] * (self.T - 1)  # K_2,...,K_T
            Ks[-1] = K
            for ti in range(self.T - 3, -1, -1):
                H = procurement_transform(K, self.order[ti, j], self.capacity[j])
                Hs[ti] = H
                K = next_K(H, z, q, self.r[ti, j], self.h[ti, j])
                Ks[ti] = K
            all_H.append(Hs)
            all_K.append(Ks)
        return all_H, all_K

    def first_stage_tree(self, Ks, c_vec=None):
        n = self.n
        # Variables u_j and theta_j; theta >= all affine pieces of K2_j.
        c = np.r_[self.initial_cost if c_vec is None else c_vec, np.ones(n)]
        A_ub, b_ub = [], []
        for j in range(n):
            K = Ks[j][0]
            # Only the feasible first-stage interval is needed in this LP.
            # Retaining the full global knot list would create thousands of
            # redundant epigraph rows on long-horizon instances.
            knots = np.r_[0.0, K.x[(K.x > 0.0) & (K.x < self.capacity[j])],
                          self.capacity[j]]
            knots = np.unique(knots)
            vals = K(knots)
            slopes = np.diff(vals) / np.diff(knots)
            for k in range(len(knots) - 1):
                row = np.zeros(2 * n)
                # slope*u - theta <= slope*knot - value
                row[j] = slopes[k]
                row[n + j] = -1.0
                A_ub.append(row)
                b_ub.append(slopes[k] * knots[k] - vals[k])
        row = np.zeros(2 * n)
        row[:n] = 1.0
        A_ub.append(row)
        b_ub.append(self.shared_capacity)
        bounds = [(0.0, float(self.capacity[j])) for j in range(n)] + [(None, None)] * n
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds,
                      method='highs')
        if not res.success:
            raise RuntimeError(res.message)
        return res.x[:n], float(res.fun)

    def continuous_reference(self, grid_step=0.04):
        """Converged grid DP reference; expectations use fixed Gauss-Legendre quadrature."""
        lo, hi = -35.0, 105.0
        grid = np.arange(lo, hi + grid_step, grid_step)
        nodes, weights = np.polynomial.legendre.leggauss(48)
        x = 5.0 * (nodes + 1.0)
        q = 5.0 * weights * np.array([self.dist.pdf(float(z)) for z in x])
        q = q / q.sum()
        all_H, all_K = [], []
        for j in range(self.n):
            z = self.loads[j] * x
            Kvals = np.array([self.continuous_ell(self.T - 2, j, float(y)) for y in grid])
            # E[(dX-s)+]=d E[(X-s/d)+]; E[(s-dX)+]=s-dE[X]+dE[(X-s/d)+].
            Kinterp = lambda v, vals=Kvals: np.interp(v, grid, vals)
            Hs = [None] * (self.T - 1)
            Ks = [None] * (self.T - 1)
            Ks[-1] = Kinterp
            Hvals = None
            for ti in range(self.T - 3, -1, -1):
                Hprev = None if Hvals is None else (lambda v, vals=Hvals: np.interp(v, grid, vals))
                Kcurrent = Kinterp
                def objective(u, y):
                    return self.order[ti, j] * u + float(Kcurrent(y + u))
                Hnew = np.empty_like(grid)
                policies = np.empty_like(grid)
                for iy, y in enumerate(grid):
                    res = minimize_scalar(lambda u: objective(u, y), bounds=(0.0, self.capacity[j]),
                                          method='bounded', options={'xatol': 2e-8})
                    candidates = [0.0, self.capacity[j], res.x]
                    vals = [objective(u, y) for u in candidates]
                    k = int(np.argmin(vals))
                    policies[iy], Hnew[iy] = candidates[k], vals[k]
                Hvals = Hnew
                Hs[ti] = (lambda v, vals=Hnew: np.interp(v, grid, vals))
                Hpolicy = (lambda v, vals=policies: np.interp(v, grid, vals))
                Hs[ti] = (Hs[ti], Hpolicy)
                Knew = np.empty_like(grid)
                for iy, y in enumerate(grid):
                    ell = self.continuous_ell(ti, j, float(y))
                    Knew[iy] = ell + np.sum(q * Hs[ti][0](y - z))
                Kinterp = lambda v, vals=Knew: np.interp(v, grid, vals)
                Ks[ti] = Kinterp
            all_H.append(Hs)
            all_K.append(Ks)
        return all_H, all_K, dict(grid=grid, quadrature_nodes=x, quadrature_weights=q)

    def first_stage_continuous(self, Ks):
        def objective(u):
            return float(np.dot(self.initial_cost, u) + sum(Ks[j][0](u[j]) for j in range(self.n)))
        cons = {'type': 'ineq', 'fun': lambda u: self.shared_capacity - np.sum(u)}
        res = minimize(objective, np.full(self.n, self.shared_capacity / self.n),
                       method='SLSQP', bounds=[(0.0, self.capacity[j]) for j in range(self.n)],
                       constraints=cons, options={'ftol': 1e-11, 'maxiter': 500})
        if not res.success:
            raise RuntimeError(res.message)
        return res.x, objective(res.x)

    def evaluate_tree(self, atoms, probs, continuous):
        Hc, Kc, reference_data = continuous
        if 'value' not in reference_data:
            uc, vc = self.first_stage_continuous(Kc)
            reference_data.update(value=vc, initial=uc)
        vc, uc = reference_data['value'], reference_data['initial']
        Hb, Kb = self.tree_dp(atoms, probs)
        ub, vb = self.first_stage_tree(Kb)

        def continuous_K_on_tree(t, j, state):
            z, q = self.scaled_atoms(atoms, probs, j)
            ans = expected_hinge_tree(state, z, q, self.r[t, j], self.h[t, j])
            if t < self.T - 2:
                ans += sum(qk * Hc[j][t][0](state - zk) for zk, qk in zip(z, q))
            return float(ans)

        # Integration-only stage-2 error at the fixed tree decision. Unlike
        # Kc-Kb, this excludes the downstream value-function gap.
        eps1 = sum(Kc[j][0](ub[j]) - continuous_K_on_tree(0, j, ub[j])
                   for j in range(self.n))
        sigma1 = float(np.dot(self.initial_cost, ub) + sum(Kc[j][0](ub[j]) for j in range(self.n)) - vc)

        def future_gap(t, j, state, xi):
            # G_t(s,xi), where t=0 denotes stage 2. The terminal gap is zero.
            y = state - xi
            if t == self.T - 2:
                return 0.0
            Htree = Hb[j][t]
            Hcont = Hc[j][t][0]
            ut = procurement_policy(Htree, y)
            Knext_c = Kc[j][t + 1]
            sigma = self.order[t, j] * ut + Knext_c(y + ut) - Hcont(y)
            z, q = self.scaled_atoms(atoms, probs, j)
            next_state = y + ut
            eps = Knext_c(next_state) - continuous_K_on_tree(t + 1, j, next_state)
            continuation = sum(qk * future_gap(t + 1, j, next_state, zk)
                               for zk, qk in zip(z, q))
            return eps + continuation - sigma

        total_future = 0.0
        for j in range(self.n):
            zj, qj = self.scaled_atoms(atoms, probs, j)
            total_future += sum(qk * future_gap(0, j, ub[j], zk)
                                for zk, qk in zip(zj, qj))
        decomposed_gap = eps1 + total_future - sigma1
        direct_gap = vc - vb
        return dict(value=vb, reference=vc, gap=direct_gap,
                    residual=abs(decomposed_gap - direct_gap), sigma1=sigma1,
                    initial=ub, continuous_initial=uc)


def application_certificate(inst, atoms, probs, kind, profile, dist):
    base = transport_w1(dist, atoms, probs) if kind == 'transport' else 2.0 * profile
    weight = sum(inst.loads[j] * inst.L[:, j].sum() for j in range(inst.n))
    return weight * base


METHODS = ['Finite-B optimal', 'Equal width', 'Equal probability', 'Equal sqrt-f',
           'Equal f^(1/3)', 'Forward W1', 'Monte Carlo', 'Sobol']


def run_density_benchmark(replications=6):
    distributions = [UniformDist(), SineDist(), TruncNormal(5.4, 1.45, 'TruncNorm'),
                     MixtureDist()]
    rows = []
    for dist in distributions:
        optimal_profiles = {}
        for b in [4, 8, 16, 32]:
            start = time.perf_counter()
            atoms0, probs0, kind0, profile0 = make_tree(dist, 'Finite-B optimal', b)
            optimal_seconds = time.perf_counter() - start
            optimal_profiles[b] = profile0
            # Use the exact tree's actual CM cell probabilities, not a second
            # nearest-representative approximation with the same atoms.
            optimal_w1 = transport_w1(dist, atoms0, probs0)
            for method in METHODS:
                reps = replications if method in ('Monte Carlo', 'Sobol') else 1
                for rep in range(reps):
                    if method == 'Finite-B optimal' and rep == 0:
                        atoms, probs, kind, profile = atoms0, probs0, kind0, profile0
                        elapsed = optimal_seconds
                    else:
                        start_method = time.perf_counter()
                        atoms, probs, kind, profile = make_tree(dist, method, b,
                                                                 seed=7100 + 97 * rep + b)
                        elapsed = time.perf_counter() - start_method
                    assert abs(probs.sum() - 1.0) < 2e-12
                    assert np.all(probs > 0)
                    assert len(np.unique(atoms)) == b
                    w1 = transport_w1(dist, atoms, probs)
                    rows.append(dict(density=dist.name, B=b, method=method, rep=rep,
                                     profile=profile, W1=w1, seconds=elapsed,
                                     kind=kind, profile_ratio=profile / optimal_profiles[b]
                                     if kind == 'CM' else np.nan,
                                     W1_ratio=w1 / optimal_w1))
            # The exact optimizer dominates every barycentric competitor at its objective.
            for method in ['Equal width', 'Equal probability', 'Equal sqrt-f', 'Equal f^(1/3)']:
                other = next(r for r in rows if r['density'] == dist.name and r['B'] == b
                             and r['method'] == method)
                assert profile0 <= other['profile'] + 3e-8
    return rows


def run_application_benchmark(replications=4, grid_step=0.04):
    instances = [
        InventoryInstance('Sine-3', SineDist(), 3, seed=0),
        InventoryInstance('TruncNorm-5', TruncNormal(5.4, 1.45, 'TruncNorm'), 5, seed=1),
        InventoryInstance('Bimodal-5', MixtureDist(), 5, seed=2),
        InventoryInstance('Uniform-7', UniformDist(), 7, seed=3),
    ]
    rows, references = [], {}
    for inst in instances:
        start_ref = time.perf_counter()
        continuous = inst.continuous_reference(grid_step=grid_step)
        uc, vc = inst.first_stage_continuous(continuous[1])
        continuous[2]['value'], continuous[2]['initial'] = vc, uc
        continuous[2]['seconds'] = time.perf_counter() - start_ref
        references[inst.name] = continuous
        budgets = [4, 8, 16] if inst.T == 3 else ([4] if inst.T >= 7 else [4, 8])
        for b in budgets:
            for method in METHODS:
                reps = replications if method in ('Monte Carlo', 'Sobol') else 1
                for rep in range(reps):
                    start = time.perf_counter()
                    atoms, probs, kind, profile = make_tree(inst.dist, method, b,
                                                             seed=9100 + 101 * rep + b)
                    construction_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    result = inst.evaluate_tree(atoms, probs, continuous)
                    solve_seconds = time.perf_counter() - start
                    certificate = application_certificate(inst, atoms, probs, kind, profile, inst.dist)
                    total_seconds = construction_seconds + solve_seconds
                    scale = max(1.0, abs(vc))
                    assert result['residual'] < 2e-6 * scale
                    assert certificate >= abs(result['gap']) - 3e-6 * scale
                    rows.append(dict(instance=inst.name, horizon=inst.T, B=b,
                                     method=method, rep=rep, value=result['value'],
                                     reference=vc, signed_error=result['gap'],
                                     absolute_error=abs(result['gap']),
                                     relative_error=abs(result['gap']) / scale,
                                     certificate=certificate,
                                     relative_certificate=certificate / scale,
                                     residual=result['residual'],
                                     seconds=total_seconds,
                                     construction_seconds=construction_seconds,
                                     solve_seconds=solve_seconds,
                                     kind=kind))
        print('Completed application', inst.name, 'reference', vc,
              'reference seconds', continuous[2]['seconds'], flush=True)
    return rows, references


def aggregate_application(rows):
    summary = []
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        summary.append(dict(
            method=method,
            mean_relative_error=float(np.mean([r['relative_error'] for r in selected])),
            median_relative_error=float(np.median([r['relative_error'] for r in selected])),
            max_relative_error=float(np.max([r['relative_error'] for r in selected])),
            mean_relative_certificate=float(np.mean([r['relative_certificate'] for r in selected])),
            mean_seconds=float(np.mean([r['seconds'] for r in selected])),
            max_residual=float(np.max([r['residual'] for r in selected]))))
    return summary
# ===== Original deterministic verification =====
ROOT = Path(__file__).resolve().parent
A, B = 0.0, 10.0
OMEGA = 6.0 * math.pi / 10.0

def f(x):
    return (1.0 + 0.3 * math.sin(OMEGA * x)) / 10.0

def fp(x):
    return 0.3 * OMEGA * math.cos(OMEGA * x) / 10.0

def F(x):
    return (x + 0.3 * (1.0 - math.cos(OMEGA * x)) / OMEGA) / 10.0

def integrate(func, a, b, eps=2e-12):
    return quad(func, a, b, epsabs=eps, epsrel=eps, limit=200)[0]

def weighted_mass(power, a, b):
    return integrate(lambda x: f(x) ** power, a, b)

def equal_mass_interval(Bcells, power, left, right):
    """Consecutive cells with equal integral of f**power on an interval."""
    total = weighted_mass(power, left, right)
    target = total / Bcells
    edges = [left]
    for _ in range(Bcells - 1):
        a = edges[-1]
        edges.append(brentq(lambda x: weighted_mass(power, a, x) - target,
                            a, right, xtol=2e-13, rtol=8.9e-16))
    edges.append(right)
    return np.array(edges)

def equal_mass_tree(Bcells, power):
    return equal_mass_interval(Bcells, power, A, B)

def arbitrary_tree(edges):
    return np.asarray(edges, dtype=float)

def cell_data(edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        p = integrate(f, a, b)
        mean_integrand = integrate(lambda x: x * f(x), a, b)
        c = mean_integrand / p
        # Split at the centroid so quadrature never crosses a hinge kink.
        g = 0.5 * (integrate(lambda x: (c - x) * f(x), a, c)
                   + integrate(lambda x: (x - c) * f(x), c, b))
        second = integrate(lambda x: (x - c) ** 2 * f(x), a, b)
        mad_half = 0.5 * (integrate(lambda x: (c - x) * f(x), a, c)
                          + integrate(lambda x: (x - c) * f(x), c, b))
        assert abs(g - mad_half) < 2e-10
        out.append(dict(a=a, b=b, p=p, c=c, g=g, second=second))
    return out

def profile(edges):
    return max(cell['g'] for cell in cell_data(edges))

def direct_hinge_gap(edges, eta, jump=1.5, slope=-0.75):
    """Gap of slope + jump*(x-eta)_+ for one fixed tree."""
    continuous = integrate(lambda x: (slope * x + jump * max(x - eta, 0.0)) * f(x), A, B)
    discrete = 0.0
    for cell in cell_data(edges):
        c = cell['c']
        discrete += cell['p'] * (slope * c + jump * max(c - eta, 0.0))
    return continuous - discrete

def fmt(x, digits=6):
    return f'{x:.{digits}f}'

def sci(x, digits=3):
    mantissa, exponent = f'{x:.{digits}e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'

results = {}

# 1. Exact fixed-tree PL minimax identities.
Lbar = 0.75
fixed_edges = {
    'Equal width': np.linspace(A, B, 9),
    r'Equal $\sqrt f$ mass': equal_mass_tree(8, 0.5),
    'Irregular': arbitrary_tree([0, 0.8, 2.1, 4.6, 5.2, 7.7, 8.0, 9.1, 10]),
}
fixed_rows = []
for name, edges in fixed_edges.items():
    cells = cell_data(edges)
    imax = max(range(len(cells)), key=lambda i: cells[i]['g'])
    G = cells[imax]['g']
    direct = direct_hinge_gap(edges, cells[imax]['c'], jump=2 * Lbar, slope=-Lbar)
    identity = 2 * Lbar * G
    assert abs(direct - identity) < 2e-9
    fixed_rows.append((name, len(cells), imax + 1, G, direct, abs(direct - identity)))
results['fixed'] = fixed_rows

# 2. Equal-sqrt(f)-mass convergence.
Gamma = weighted_mass(0.5, A, B) ** 2
sharp_target = Gamma / 8.0
sharp_rows = []
for b in [8, 16, 32, 64, 128, 256]:
    edges = equal_mass_tree(b, 0.5)
    G = profile(edges)
    scaled = b * b * G
    sharp_rows.append((b, G, scaled, scaled / sharp_target))
assert sharp_rows[-1][3] > 0.98 and sharp_rows[-1][3] < 1.03
results['sharp'] = sharp_rows

# 3. Smooth-cost squared quantization and the f^(1/3) law.
I3 = weighted_mass(1.0 / 3.0, A, B)
zador_target = I3 ** 3 / 12.0
smooth_rows = []
for b in [8, 16, 32, 64, 128, 256]:
    edges = equal_mass_tree(b, 1.0 / 3.0)
    distortion = sum(cell['second'] for cell in cell_data(edges))
    scaled = b * b * distortion
    smooth_rows.append((b, distortion, scaled, scaled / zador_target))
assert 0.99 < smooth_rows[-1][3] < 1.03
results['smooth'] = smooth_rows

# 4. Fixed-tree multistage minimax values on additive LP instances.
def additive_stage_gap(edges, L):
    cells = cell_data(edges)
    cell = max(cells, key=lambda z: z['g'])
    a = cell['c']
    cont = integrate(lambda x: (-L * x + 2 * L * max(x - a, 0.0)) * f(x), A, B)
    disc = sum(z['p'] * (-L * z['c'] + 2 * L * max(z['c'] - a, 0.0)) for z in cells)
    return cont - disc, 2 * L * cell['g']

multi_rows = []
for T in [2, 3, 5, 10]:
    for b in [8, 16]:
        direct_sum = 0.0
        formula_sum = 0.0
        for t in range(2, T + 1):
            Lt = 0.4 + 0.05 * t
            direct, formula = additive_stage_gap(equal_mass_tree(b, 0.5), Lt)
            direct_sum += direct
            formula_sum += formula
        assert abs(direct_sum - formula_sum) < 5e-9
        multi_rows.append((T, b, direct_sum, formula_sum, abs(direct_sum - formula_sum)))
results['multi'] = multi_rows

# 5. Exact Bellman-slack decomposition on a three-stage production LP.
# All continuous expectations below are exact for independent U[0,1] variables.
def U_hinge(a):
    """E[(U-a)_+] for U uniform on [0,1], for real a."""
    if a <= 0.0:
        return 0.5 - a
    if a >= 1.0:
        return 0.0
    return 0.5 * (1.0 - a) ** 2

Pshort, Kcap = 2.0, 0.8
Rpen, A1cost, A2cost = 1.2, 0.35, 0.5
lp_edges = np.linspace(0.0, 1.0, 5)
lp_q = np.full(4, 0.25)
lp_atoms = (lp_edges[:-1] + lp_edges[1:]) / 2.0

def J2_cont_objective(x):
    return A2cost * x + Pshort * U_hinge(Kcap * x)

def J2_tree_objective(x):
    return A2cost * x + Pshort * sum(
        q * max(z - Kcap * x, 0.0) for q, z in zip(lp_q, lp_atoms))

# The continuous stage-2 objective is quadratic on [0,1].
x2_cont = min(1.0, max(0.0, (1.0 - A2cost / (Pshort * Kcap)) / Kcap))
# The tree objective is convex PL, so endpoints and kinks suffice.
x2_candidates = np.r_[0.0, 1.0, np.clip(lp_atoms / Kcap, 0.0, 1.0)]
x2_tree = min(x2_candidates, key=J2_tree_objective)
J2_cont = J2_cont_objective(x2_cont)
J2_tree = J2_tree_objective(x2_tree)

def cont1(x):
    return A1cost * x + Rpen * U_hinge(x) + J2_cont

def disc1(x):
    return (A1cost * x + Rpen * sum(
        q * max(z - x, 0.0) for q, z in zip(lp_q, lp_atoms)) + J2_tree)

x1_cont = 1.0 - A1cost / Rpen
x1_candidates = np.r_[0.0, 1.0, lp_atoms]
x1_tree = min(x1_candidates, key=disc1)
V1_cont = cont1(x1_cont)
V1_tree = disc1(x1_tree)

def V2_cont(s, z):
    return Rpen * max(z - s, 0.0) + J2_cont

def V2_tree(s, z):
    return Rpen * max(z - s, 0.0) + J2_tree

eps2 = Rpen * (U_hinge(x1_tree) - sum(
    q * max(z - x1_tree, 0.0) for q, z in zip(lp_q, lp_atoms)))
eps3 = Pshort * (U_hinge(Kcap * x2_tree) - sum(
    q * max(z - Kcap * x2_tree, 0.0) for q, z in zip(lp_q, lp_atoms)))
sigma1 = cont1(x1_tree) - cont1(x1_cont)
sigma2 = J2_cont_objective(x2_tree) - J2_cont_objective(x2_cont)
G2_values = [V2_cont(x1_tree, z) - V2_tree(x1_tree, z) for z in lp_atoms]
local_G1 = eps2 + sum(q * g for q, g in zip(lp_q, G2_values)) - sigma1
lp_gap = V1_cont - V1_tree
lp_rhs = eps2 + eps3 - sigma1 - sigma2
assert x1_tree != x1_cont and x2_tree != x2_cont
assert sigma1 > 1e-8 and sigma2 > 1e-8
assert lp_gap >= -1e-12 and min(G2_values) >= -1e-12
assert abs(local_G1 - lp_gap) < 2e-12
assert abs(lp_rhs - lp_gap) < 2e-12
results['slack'] = dict(x1c=x1_cont, x1b=x1_tree, x2c=x2_cont,
                        x2b=x2_tree, sigma1=sigma1, sigma2=sigma2,
                        V1c=V1_cont, V1b=V1_tree, gap=lp_gap,
                        eps=eps2 + eps3, slacks=sigma1 + sigma2,
                        rhs=lp_rhs, residual=abs(lp_rhs - lp_gap))

# 6. Moreau fixed-mu asymptotics and separate small-mu limit.
def moreau_psi(x, mu):
    if x < 5.0 - mu:
        return 5.0 - x - 0.5 * mu
    if x > 5.0 + mu:
        return x - 5.0 - 0.5 * mu
    return (x - 5.0) ** 2 / (2.0 * mu)

def moreau_tree(mu, total_cells):
    core = equal_mass_interval(total_cells - 2, 1.0 / 3.0,
                               5.0 - mu, 5.0 + mu)
    return np.r_[A, core, B]

def moreau_measurements(mu, total_cells):
    edges = moreau_tree(mu, total_cells)
    cells = cell_data(edges)
    continuous = (integrate(lambda x: moreau_psi(x, mu) * f(x), A, 5.0 - mu)
                  + integrate(lambda x: moreau_psi(x, mu) * f(x),
                              5.0 - mu, 5.0 + mu)
                  + integrate(lambda x: moreau_psi(x, mu) * f(x), 5.0 + mu, B))
    discrete = sum(cell['p'] * moreau_psi(cell['c'], mu) for cell in cells)
    direct = continuous - discrete
    core_cells = cell_data(edges[1:-1])
    second_moment = sum(cell['second'] for cell in core_cells)
    formula = second_moment / (2.0 * mu)
    assert abs(direct - formula) < 2e-9
    return direct

mu_fixed = 0.5
Imu_fixed = (mu_fixed ** (-1.0 / 3.0)
             * weighted_mass(1.0 / 3.0, 5.0 - mu_fixed, 5.0 + mu_fixed))
moreau_fixed_rows = []
for b in [12, 22, 42, 82, 162, 322, 642, 1282]:
    gap_mu = moreau_measurements(mu_fixed, b)
    target = Imu_fixed ** 3 / (24.0 * b * b)
    moreau_fixed_rows.append((b, gap_mu, gap_mu / target))
assert 0.99 < moreau_fixed_rows[-1][2] < 1.01

C_K = 2.0 * f(5.0) ** (1.0 / 3.0)
moreau_small_rows = []
small_B = 1002
for mu in [0.5, 0.25, 0.125, 0.0625]:
    gap_mu = moreau_measurements(mu, small_B)
    target = mu * mu * C_K ** 3 / (24.0 * small_B * small_B)
    moreau_small_rows.append((mu, gap_mu, gap_mu / target))
assert 0.97 < moreau_small_rows[-1][2] < 1.04
results['moreau_fixed'] = moreau_fixed_rows
results['moreau_small'] = moreau_small_rows

# 7. Exact finite-B optimal trees by threshold feasibility.
def single_cell_g(a, b):
    if b - a < 1e-13:
        return 0.0
    return cell_data([a, b])[0]['g']

def maximal_right(a, level):
    if single_cell_g(a, B) <= level:
        return B
    return brentq(lambda b: single_cell_g(a, b) - level, a, B,
                  xtol=2e-12, rtol=8.9e-16)

def threshold_reachable(level, Bcells):
    a = A
    for _ in range(Bcells):
        a = maximal_right(a, level)
        if a >= B - 1e-9:
            return True
    return False

def optimal_pl_tree(Bcells):
    if Bcells == 1:
        return np.array([A, B])
    high = profile(np.linspace(A, B, Bcells + 1))
    low = 0.0
    for _ in range(52):
        mid = (low + high) / 2.0
        if threshold_reachable(mid, Bcells):
            high = mid
        else:
            low = mid
    edges = [A]
    a = A
    for _ in range(Bcells - 1):
        a = maximal_right(a, high)
        edges.append(a)
    edges.append(B)
    return np.array(edges)

optimal_edges = {}
optimal_rows = []
for b in [2, 3, 4, 8, 16, 32, 64]:
    edges = optimal_pl_tree(b)
    optimal_edges[b] = edges
    Gopt = profile(edges)
    Gsqrt = profile(equal_mass_tree(b, 0.5))
    assert Gopt <= Gsqrt + 2e-9
    optimal_rows.append((b, Gopt, Gsqrt, Gsqrt / Gopt, b * b * Gopt))
results['optimal'] = optimal_rows

# Independent direct finite-B minimization for B=2 and B=3.
def direct_partition_objective(y):
    edges = np.r_[A, np.sort(np.atleast_1d(y)), B]
    return profile(edges)

direct2 = minimize_scalar(lambda x: direct_partition_objective(float(x)),
                          bounds=(A + 1e-7, B - 1e-7), method='bounded',
                          options={'xatol': 1e-11})
direct3 = differential_evolution(direct_partition_objective, [(A, B), (A, B)],
                                 seed=123, tol=1e-9, polish=True)
assert abs(direct2.fun - optimal_rows[0][1]) < 2e-6
assert abs(direct3.fun - optimal_rows[1][1]) < 2e-6

# Check the explicit first-order estimate for this test density.
fmin_test = 0.07
lip_f = 0.3 * OMEGA / 10.0
lip_sqrt_f = lip_f / (2.0 * math.sqrt(fmin_test))
A_sqrt = weighted_mass(0.5, A, B)
C_f = (A_sqrt ** 2 * (B - A) * lip_sqrt_f / (4.0 * math.sqrt(fmin_test))
       + 3.0 * A_sqrt ** 3 * lip_f / (8.0 * fmin_test ** 1.5))
for b, Gopt, _, _, scaled in optimal_rows:
    assert Gamma / 8.0 - C_f / b <= scaled + 2e-8
    assert scaled <= Gamma / 8.0 + C_f / b + 2e-8
results['explicit_constant'] = C_f

# 8. Comparative nonuniform three-stage production LP.
mean_xi = integrate(lambda x: x * f(x), A, B)
def general_hinge(a):
    if a <= A:
        return mean_xi - a
    if a >= B:
        return 0.0
    return integrate(lambda x: (x - a) * f(x), a, B)

def continuous_minimizer(obj):
    res = minimize_scalar(obj, bounds=(A, B), method='bounded',
                          options={'xatol': 1e-10})
    return res.x, res.fun

def tree_minimizer(obj, atoms, scale=1.0):
    candidates = np.r_[A, B, np.clip(atoms / scale, A, B)]
    values = [obj(float(x)) for x in candidates]
    i = int(np.argmin(values))
    return float(candidates[i]), values[i]

x2_cont_general, J2_cont_general = continuous_minimizer(
    lambda x: A2cost * x + Pshort * general_hinge(Kcap * x))
x1_cont_general, V1_cont_general = continuous_minimizer(
    lambda x: A1cost * x + Rpen * general_hinge(x) + J2_cont_general)

lp_methods = {
    'Equal width': lambda b: np.linspace(A, B, b + 1),
    'Equal probability': lambda b: equal_mass_tree(b, 1.0),
    r'Equal $f^{1/3}$ mass': lambda b: equal_mass_tree(b, 1.0 / 3.0),
    r'Equal $\sqrt f$ mass': lambda b: equal_mass_tree(b, 0.5),
    r'Finite-$B$ optimal': lambda b: optimal_edges[b],
}
lp_rows = []
for b in [4, 8, 16, 32]:
    certificates = {}
    for method, constructor in lp_methods.items():
        edges = constructor(b)
        cells = cell_data(edges)
        q = np.array([cell['p'] for cell in cells])
        atoms = np.array([cell['c'] for cell in cells])
        G = max(cell['g'] for cell in cells)
        def J2_tree_general(x):
            return A2cost * x + Pshort * sum(q * np.maximum(atoms - Kcap * x, 0.0))
        x2_tree_general, J2_tree_value = tree_minimizer(J2_tree_general, atoms, Kcap)
        def disc1_general(x):
            return (A1cost * x + Rpen * sum(q * np.maximum(atoms - x, 0.0))
                    + J2_tree_value)
        x1_tree_general, V1_tree_general = tree_minimizer(disc1_general, atoms)
        gap_general = V1_cont_general - V1_tree_general
        eps2_general = Rpen * (general_hinge(x1_tree_general)
                               - sum(q * np.maximum(atoms - x1_tree_general, 0.0)))
        eps3_general = Pshort * (general_hinge(Kcap * x2_tree_general)
                                 - sum(q * np.maximum(atoms - Kcap * x2_tree_general, 0.0)))
        sigma1_general = ((A1cost * x1_tree_general + Rpen * general_hinge(x1_tree_general)
                           + J2_cont_general) - V1_cont_general)
        sigma2_general = (A2cost * x2_tree_general
                          + Pshort * general_hinge(Kcap * x2_tree_general)
                          - J2_cont_general)
        decomposition = eps2_general + eps3_general - sigma1_general - sigma2_general
        certificate = 2.0 * (Rpen + Pshort) * G
        assert gap_general >= -2e-8
        assert certificate >= gap_general - 2e-8
        assert abs(gap_general - decomposition) < 3e-7
        certificates[method] = certificate
        lp_rows.append(dict(B=b, method=method, gap=gap_general,
                            certificate=certificate, x1=x1_tree_general,
                            x2=x2_tree_general, V1B=V1_tree_general,
                            eps=eps2_general + eps3_general,
                            slacks=sigma1_general + sigma2_general,
                            residual=abs(gap_general - decomposition)))
    optimal_name = r'Finite-$B$ optimal'
    assert certificates[optimal_name] <= min(certificates.values()) + 2e-8
results['comparative_lp'] = lp_rows
results['continuous_lp'] = dict(x1=x1_cont_general, x2=x2_cont_general,
                                V1=V1_cont_general)

# 9. COR-focused multi-family benchmarks and larger inventory applications.
# Set COR_USE_CACHE=1 only to reproduce the document from previously computed
# deterministic outputs. With no cache, every number is recomputed.
import os
import pickle

if os.environ.get('COR_USE_CACHE') == '1':
    with open(ROOT / 'density_results.pkl', 'rb') as handle:
        density_rows = pickle.load(handle)
    application_rows = []
    for filename in ['app_sine3.pkl', 'app_norm5.pkl', 'app_mix5.pkl', 'app_unif7.pkl']:
        with open(ROOT / filename, 'rb') as handle:
            application_rows.extend(pickle.load(handle))
    with open(ROOT / 'reference_coarse.pkl', 'rb') as handle:
        coarse_reference_values = pickle.load(handle)
else:
    density_rows = run_density_benchmark(replications=6)
    application_rows, _ = run_application_benchmark(replications=4, grid_step=0.04)
    coarse_reference_values = {}
    reference_specs = [
        ('Sine-3', SineDist(), 3, 0),
        ('TruncNorm-5', TruncNormal(5.4, 1.45, 'TruncNorm'), 5, 1),
        ('Bimodal-5', MixtureDist(), 5, 2),
        ('Uniform-7', UniformDist(), 7, 3),
    ]
    for instance_name, distribution, horizon, instance_seed in reference_specs:
        reference_instance = InventoryInstance(instance_name, distribution, horizon, instance_seed)
        coarse_continuous = reference_instance.continuous_reference(grid_step=0.08)
        _, coarse_value = reference_instance.first_stage_continuous(coarse_continuous[1])
        coarse_reference_values[instance_name] = coarse_value

application_summary = aggregate_application(application_rows)
reference_values = {row['instance']: row['reference'] for row in application_rows}
reference_grid_rows = [(name, reference_values[name], coarse_reference_values[name],
                        abs(reference_values[name] - coarse_reference_values[name]))
                       for name in sorted(reference_values)]
assert max(row[3] for row in reference_grid_rows) < 0.01
assert min(row['mean_relative_certificate'] for row in application_summary) == next(
    row['mean_relative_certificate'] for row in application_summary
    if row['method'] == 'Finite-B optimal')
results['density_benchmark'] = density_rows
results['application_benchmark'] = application_rows
results['application_summary'] = application_summary
results['reference_grid'] = reference_grid_rows

# Generate external vector figures and the main numerical section.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGDIR = ROOT / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 8.5,
    'axes.titlesize': 9.5,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 1.0,
    'grid.color': '#d4d4d4',
    'grid.linewidth': 0.55,
    'lines.linewidth': 1.25,
    'lines.markersize': 4,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
COLORS = ['#173f8a', '#b3272d', '#2d6a4f', '#6a3d9a', '#e07a00',
          '#287d8e', '#7a7a7a', '#c43c96']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'x', '>']

def budget_ticks(ax, values):
    values = list(values)
    ax.set_xscale('log', base=2)
    ax.set_xticks(values)
    ax.set_xticklabels([rf'$2^{{{int(round(math.log2(v)))}}}$' for v in values])
    ax.minorticks_off()

def save_vector_figure(fig, stem):
    for suffix in ('pdf', 'eps'):
        fig.savefig(FIGDIR / f'{stem}.{suffix}', format=suffix,
                    bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)

# Figure 1: scaled PL and smooth asymptotic laws.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
opt_curve = [(r[0], r[4]) for r in optimal_rows if r[0] != 3]
sqrt_curve = [(r[0], r[0] ** 2 * r[2]) for r in optimal_rows if r[0] != 3]
ax = axes[0]
ax.plot(*zip(*opt_curve), color=COLORS[0], marker='o', label='finite-$B$ optimal')
ax.plot(*zip(*sqrt_curve), color=COLORS[1], marker='s', label=r'equal $\sqrt{f}$ mass')
ax.axhline(sharp_target, color='black', linestyle='--', linewidth=1.0,
           label=r'$\Gamma(f)/8$')
budget_ticks(ax, [r[0] for r in opt_curve])
ax.set_title('(a) PL profiles')
ax.set_xlabel('$B$')
ax.set_ylabel(r'$B^2\mathfrak{G}$')
ax.legend(loc='upper right', framealpha=1.0)
ax = axes[1]
ax.plot([r[0] for r in smooth_rows], [r[2] for r in smooth_rows],
        color=COLORS[0], marker='o', label=r'equal $f^{1/3}$ mass')
ax.axhline(zador_target, color='black', linestyle='--', linewidth=1.0,
           label=r'$(\int f^{1/3})^3/12$')
budget_ticks(ax, [r[0] for r in smooth_rows])
ax.set_title('(b) Smooth distortion')
ax.set_xlabel('$B$')
ax.set_ylabel('$B^2D_B$')
ax.legend(loc='lower right', framealpha=1.0)
fig.tight_layout(w_pad=2.0)
save_vector_figure(fig, 'scaled_asymptotic_laws')

# Figure 2: density, exact optimal boundaries, and cell widths.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
ax = axes[0]
xx = np.linspace(A, B, 401)
ax.plot(xx, [f(x) for x in xx], color=COLORS[0])
for edge in optimal_edges[8][1:-1]:
    ax.axvline(edge, color='#d98585', linestyle='--', linewidth=0.85)
ax.set_ylim(-0.004, 0.14)
ax.set_title('(a) Density and optimal cells')
ax.set_xlabel('$x$')
ax.set_ylabel('$f(x)$')
ax = axes[1]
for j, (label, edges) in enumerate([
        ('finite-$B$ optimal', optimal_edges[8]),
        (r'equal $\sqrt{f}$', equal_mass_tree(8, 0.5)),
        (r'equal $f^{1/3}$', equal_mass_tree(8, 1.0 / 3.0))]):
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    ax.plot(midpoints, np.diff(edges), color=COLORS[j], marker=MARKERS[j], label=label)
ax.set_title('(b) Cell widths for $B=8$')
ax.set_xlabel('cell midpoint')
ax.set_ylabel('width')
ax.set_ylim(1.08, 1.48)
ax.legend(loc='upper left', framealpha=1.0)
fig.tight_layout(w_pad=2.0)
save_vector_figure(fig, 'optimal_tree_geometry')

# Figure 3: actual LP gaps and robust minimax certificates.
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
handles = []
for j, method in enumerate(lp_methods):
    method_label = method.replace(r'\sqrt f', r'\sqrt{f}')
    rows = [row for row in lp_rows if row['method'] == method]
    x = [row['B'] for row in rows]
    for ax, key in zip(axes, ('gap', 'certificate')):
        line, = ax.plot(x, [row[key] for row in rows], color=COLORS[j],
                        marker=MARKERS[j], label=method_label)
        if ax is axes[0] and j == 0:
            handles.append(line)
for ax, title, ylabel in [
        (axes[0], '(a) Actual LP gap', '$V_1^*-V_1^B$'),
        (axes[1], '(b) Robust minimax certificate',
         r'$2(r\mathfrak{G}_2+p\mathfrak{G}_3)$')]:
    ax.set_yscale('log')
    budget_ticks(ax, [4, 8, 16, 32])
    ax.set_title(title)
    ax.set_xlabel('$B$')
    ax.set_ylabel(ylabel)
legend_handles = [plt.Line2D([0], [0], color=COLORS[j], marker=MARKERS[j],
                            label=method.replace(r'\sqrt f', r'\sqrt{f}'))
                  for j, method in enumerate(lp_methods)]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=2.0)
save_vector_figure(fig, 'multistage_lp_comparison')

# Figure 4: common multi-density benchmark, normalized within each density/budget.
METHOD_LABELS = {
    'Finite-B optimal': 'Finite-$B$ optimal',
    'Equal width': 'Equal width',
    'Equal probability': 'Equal probability',
    'Equal sqrt-f': r'Equal $\sqrt{f}$',
    'Equal f^(1/3)': r'Equal $f^{1/3}$',
    'Forward W1': 'Forward $W_1$',
    'Monte Carlo': 'Monte Carlo',
    'Sobol': 'Sobol',
}
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
for j, method in enumerate(METHODS):
    budgets = [4, 8, 16, 32]
    if method in METHODS[:5]:
        y = [np.mean([r['profile_ratio'] for r in density_rows
                      if r['method'] == method and r['B'] == b]) for b in budgets]
        axes[0].plot(budgets, y, color=COLORS[j], marker=MARKERS[j], label=METHOD_LABELS[method])
    y = [np.mean([r['W1_ratio'] for r in density_rows
                  if r['method'] == method and r['B'] == b]) for b in budgets]
    axes[1].plot(budgets, y, color=COLORS[j], marker=MARKERS[j], label=METHOD_LABELS[method])
for ax, title, ylabel in [
        (axes[0], '(a) Minimax profile', r'mean $\mathfrak{G}/\mathfrak{G}_B^*$'),
        (axes[1], '(b) Transport distance', 'mean $W_1/W_1$(optimal tree)')]:
    budget_ticks(ax, [4, 8, 16, 32])
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('$B$')
    ax.set_ylabel(ylabel)
axes[0].set_ylim(0.92, 4.15)
axes[1].set_ylim(0.92, 2.48)
legend_handles = [plt.Line2D([0], [0], color=COLORS[j], marker=MARKERS[j],
                            label=METHOD_LABELS[m]) for j, m in enumerate(METHODS)]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=2.0)
save_vector_figure(fig, 'density_method_benchmark')

# Figure 5: application errors across densities, budgets, and horizons.
application_instances = ['Sine-3', 'TruncNorm-5', 'Bimodal-5', 'Uniform-7']
fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.45))
for ax, instance_name in zip(axes.ravel(), application_instances):
    instance_rows = [r for r in application_rows if r['instance'] == instance_name]
    budgets = sorted(set(r['B'] for r in instance_rows))
    for j, method in enumerate(METHODS):
        y = [np.mean([r['relative_error'] for r in instance_rows
                      if r['method'] == method and r['B'] == b]) for b in budgets]
        ax.plot(budgets, y, color=COLORS[j], marker=MARKERS[j], label=METHOD_LABELS[method])
    ax.set_yscale('log')
    budget_ticks(ax, budgets)
    ax.set_title(instance_name.replace('-', ', $T=') + '$')
    ax.set_xlabel('$B$')
    ax.set_ylabel(r'$|V_1^*-V_1^B|/|V_1^*|$')
fig.legend(handles=legend_handles, loc='lower center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=2.0, w_pad=2.0)
save_vector_figure(fig, 'inventory_application_comparison')

lines = []
add = lines.append
add(r'\section{Numerical experiments}\label{sec:numerics}')
add('The accompanying deterministic script computes every table entry and plot coordinate, '
    'regenerates this section and the appendix checks, and stops on a failed assertion. '
    r'Quadrature tolerances are $2\times10^{-12}$. Equal-mass boundaries and maximal admissible '
    'endpoints are computed by root finding; convex PL decisions are checked at all breakpoints '
    'and endpoints. The experiments illustrate the theorems but do not replace their proofs. '
    'All runs use one Python process on an Intel Xeon Platinum CPU. Continuous-reference '
    'construction is excluded from method timings; construction-only and total tree-solution '
    'times are identified separately below.')
add(r'The test density on $\Xi=[0,10]$ is')
add(r'\[')
add(r'f(x)=\frac{1+0.3\sin(6\pi x/10)}{10},\qquad 0\le x\le10.')
add(r'\]')

add(r'\subsection{Exact optimal trees and asymptotic laws}')
add(r'Figure~\ref{fig:scaledlaws} compares the exact finite-$B$ optimizer from '
    r'Theorem~\ref{thm:finiteopt} with equal $\sqrt f$-mass cells. The smooth panel uses '
    r'equal $f^{1/3}$-mass cells. Both approach their distinct theoretically predicted constants.')
add(r'\begin{figure}[htbp]')
add(r'\centering')
add(r'\includegraphics[width=\textwidth]{figures/scaled_asymptotic_laws.pdf}')
add(r'\caption{Scaled PL and smooth errors. The two theories have different optimal mass laws and different limiting constants.}')
add(r'\label{fig:scaledlaws}')
add(r'\end{figure}')

add(r'Table~\ref{tab:optimalnumeric} records the exact optimal values. Independent direct '
    'partition optimization for $B=2,3$ agrees with the threshold algorithm to numerical tolerance.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Exact finite-$B$ optimal profiles and comparison with equal $\sqrt f$-mass cells.}')
add(r'\label{tab:optimalnumeric}')
add(r'\begin{tabular}{rrrr}')
add(r'\toprule')
add(r'$B$ & $\mathfrak G_B^*$ & $\mathfrak G_{\sqrt f}/\mathfrak G_B^*$ & $B^2\mathfrak G_B^*$\\')
add(r'\midrule')
for b, Gopt, Gsqrt, ratio, scaled in optimal_rows:
    add(f'{b} & ${sci(Gopt)}$ & {fmt(ratio)} & {fmt(scaled)}\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add(r'For this density, the explicit constant in \eqref{eq:Cf} is '
    rf'$C_f={fmt(C_f,4)}$, and every computed optimal value satisfies \eqref{{eq:explicitrate}}. '
    r'Figure~\ref{fig:treegeometry} shows that the finite-$B$ optimal, PL-asymptotic, and '
    'smooth-asymptotic constructions have different cell geometries.')
add(r'\begin{figure}[htbp]')
add(r'\centering')
add(r'\includegraphics[width=\textwidth]{figures/optimal_tree_geometry.pdf}')
add(r'\caption{Geometry of eight-cell trees. Vertical lines in panel (a) are the exact optimal internal boundaries.}')
add(r'\label{fig:treegeometry}')
add(r'\end{figure}')

add(r'\FloatBarrier')
add(r'\subsection{Comparative three-stage production LP}')
add(r'We next compare the trees in a nonuniform version of the production example. '
    r'Let $\xi_2,\xi_3$ be independent with density $f$, and use $p=2$, $K=0.8$, '
    r'$r=1.2$, $a_1=0.35$, and $a_2=0.5$. The model is')
add(r'\begin{align*}')
add(r'V_3(s,\xi_3)&=p(\xi_3-Ks)_+,\\')
add(r'V_2(s,\xi_2)&=r(\xi_2-s)_++\min_{0\le x\le10}\{a_2x+\E V_3(x,\xi_3)\},\\')
add(r'V_1^*&=\min_{0\le x\le10}\{a_1x+\E V_2(x,\xi_2)\}.')
add(r'\end{align*}')
add('Each method supplies the stage trees for both random variables. '
    r'We report both the actual value gap and the robust certificate $2(r\mathfrak G_2+p\mathfrak G_3)$. '
    'Continuous expectations are integrated directly, and every tree Bellman problem is solved exactly '
    'at its breakpoints. The path decomposition is checked for every method and budget.')
add(r'\begin{figure}[htbp]')
add(r'\centering')
add(r'\includegraphics[width=\textwidth]{figures/multistage_lp_comparison.pdf}')
add(r'\caption{Comparison on the three-stage LP. The exact optimal tree minimizes the robust certificate, but a particular LP need not have the same ranking.}')
add(r'\label{fig:lpcomparison}')
add(r'\end{figure}')

row8 = next(row for row in lp_rows if row['B'] == 8 and row['method'] == r'Finite-$B$ optimal')
add(r'For the exact optimal tree at $B=8$, Table~\ref{tab:lpcertificate} gives the '
    'value and decomposition. The certificate dominates the actual gap, while the Jensen sum '
    'without slacks would again overstate the gap.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Nonuniform production LP with the exact optimal tree, $B=8$.}')
add(r'\label{tab:lpcertificate}')
add(r'\begin{tabular}{lr}')
add(r'\toprule Quantity & Value\\ \midrule')
for label, value in [(r'$V_1^*$', V1_cont_general), (r'$V_1^B$', row8['V1B']),
                     (r'$V_1^*-V_1^B$', row8['gap']),
                     (r'Jensen sum', row8['eps']), (r'slack sum', row8['slacks']),
                     (r'robust certificate', row8['certificate'])]:
    add(f'{label} & {fmt(value,9)}\\\\')
add(f'Absolute decomposition residual & ${sci(row8["residual"],1)}$\\\\')
add(r'\bottomrule\end{tabular}')
add(r'\end{table}')
row32opt = next(row for row in lp_rows if row['B'] == 32 and row['method'] == r'Finite-$B$ optimal')
row32prob = next(row for row in lp_rows if row['B'] == 32 and row['method'] == 'Equal probability')
add(f"At $B=32$, the optimal-tree certificate is {fmt(row32opt['certificate'],8)}, "
    f"the smallest of all methods, whereas its actual gap is {fmt(row32opt['gap'],8)}. "
    f"Equal-probability cells have a smaller actual gap, {fmt(row32prob['gap'],8)}, "
    f"but a larger certificate, {fmt(row32prob['certificate'],8)}. "
    'This is consistent with the theory: the optimizer minimizes the worst case over the '
    'sensitivity class, not the error of one selected instance.')

add(r'\subsection{Common multi-family benchmark and stronger baselines}')
add('We next separate construction quality from performance on a particular LP. '
    'Four bounded densities are used: uniform, the sinusoidal density above, a normal '
    'distribution with mean $5.4$ and standard deviation $1.45$ truncated to $[0,10]$, '
    'and a $0.42/0.58$ mixture of truncated normals centered at $2.2$ and $7.3$. '
    'The budgets are $B=4,8,16,32$. In addition to the barycentric constructions, '
    'we test equal-weight Monte Carlo and scrambled Sobol inverse-CDF samples, and '
    'forward scenario selection under the $L^1$ probability metric '
    r'\cite{dupacova03}. Forward selection starts from 768 midpoint-quantile scenarios, '
    'adds the representative giving the largest decrease in empirical $W_1$, and '
    'reassigns probability by nearest representative. Monte Carlo and Sobol results '
    'are averaged over six independent scrambled/seeded replications.')
add(r'For any finite-support approximation $Q$, the one-dimensional transport distance '
    r'$W_1(P,Q)$ is computed exactly by monotone quantile transport: consecutive '
    r'source-quantile intervals are assigned to the ordered atoms according to their actual '
    r'probabilities. Thus unequal reduction weights and equal sampling weights are both '
    r'handled correctly; Voronoi reassignment is used only where the construction itself '
    r'specifies it. The distance gives the common transport certificate. Barycentric trees additionally have '
    'the sharper, generally different, maximum-cell minimax certificate proved above. '
    'Sampling and reduced-support trees need not be barycentric: their value errors '
    'can have either sign, and Jensen one-sided bounds are not applied to them.')
add(r'\begin{figure}[htbp]')
add(r'\centering')
add(r'\includegraphics[width=\textwidth]{figures/density_method_benchmark.pdf}')
add(r'\caption{Multi-density construction benchmark. Each curve averages ratios within every density and budget; sampling curves average six replications. The denominator in panel (b) is the transport distance of the exact minimax-profile tree, not a claim that this tree minimizes $W_1$.}')
add(r'\label{fig:densitybenchmark}')
add(r'\end{figure}')
add(r'\begin{table}[htbp]')
add(r'\centering\small\setlength{\tabcolsep}{4.5pt}')
add(r'\caption{Construction results aggregated over four densities and four budgets.}')
add(r'\label{tab:densitysummary}')
add(r'\begin{tabular}{lrrrr}')
add(r'\toprule Method & mean $\mathfrak G/\mathfrak G_B^*$ & mean $W_1$ ratio & max $W_1$ ratio & time (ms)\\')
add(r'\midrule')
for method in METHODS:
    selected = [r for r in density_rows if r['method'] == method]
    g_values = [r['profile_ratio'] for r in selected if r['kind'] == 'CM']
    g_text = fmt(float(np.mean(g_values))) if g_values else '--'
    mean_w1 = float(np.mean([r['W1_ratio'] for r in selected]))
    max_w1 = float(np.max([r['W1_ratio'] for r in selected]))
    mean_ms = 1000.0 * float(np.mean([r['seconds'] for r in selected]))
    add(f"{METHOD_LABELS[method]} & {g_text} & {fmt(mean_w1)} & {fmt(max_w1)} & {fmt(mean_ms,2)}\\\\")
add(r'\bottomrule\end{tabular}')
add(r'\end{table}')
add(r'The exact algorithm has profile ratio one by construction. Equal $\sqrt f$-mass '
    'cells are close asymptotically, while equal $f^{1/3}$ cells optimize a different '
    'smooth-cost objective and do not minimize the PL profile. Forward reduction is '
    'competitive in transport distance, and Sobol improves substantially on Monte Carlo. '
    'Neither observation implies the same ranking for the maximum-cell certificate. '
    'Timings cover tree construction only; they are indicative single-process measurements, '
    'not a total-work complexity claim.')

add(r'\FloatBarrier')
add(r'\subsection{A six-item multiperiod inventory application}')
add('The larger test is a six-item inventory system with a common scalar demand factor, '
    'stagewise-independent demands, shared initial capacity, holding and backlog costs, '
    'and capacitated replenishment. For item $j$, let $d_j$ be its demand loading and')
add(r'\[')
add(r'\ell_{tj}(s,\xi)=r_{tj}(d_j\xi-s)_++h_{tj}(s-d_j\xi)_+,\qquad t=2,\ldots,T.')
add(r'\]')
add(r'If $y=s-d_j\xi_t$ is inventory after current demand and $u$ is replenishment, '
    'the next state is $y+u$. With $H_{Tj}=0$, the Bellman functions are')
add(r'\begin{align*}')
add(r'H_{tj}(y)&=\min_{0\le u\le U_j}\{a_{tj}u+K_{t+1,j}(y+u)\},\qquad t=2,\ldots,T-1,\\')
add(r'K_{tj}(z)&=\E[\ell_{tj}(z,\xi_t)+H_{tj}(z-d_j\xi_t)],\qquad t=2,\ldots,T.')
add(r'\end{align*}')
add('The initial decisions share a capacity constraint:')
add(r'\[')
add(r'V_1^*=\min_{\substack{0\le u_j\le U_j\\\sum_j u_j\le C}}\sum_{j=1}^6\{c_ju_j+K_{2j}(u_j)\}.')
add(r'\]')
add(r'The loadings are $(0.55,0.75,0.90,1.05,1.25,1.45)$, item capacities are '
    r'$(7,8,9,10,11,12)$, and the shared capacities are $34,31,37,37$ in the four instances. '
    'The deterministic holding, shortage, and ordering costs vary by item and stage and '
    'are printed explicitly in the accompanying script. The instances use $T=3$ with the '
    'sine density, $T=5$ with truncated-normal and bimodal densities, and $T=7$ with the '
    'uniform density. Budgets are $4,8,16$ for $T=3$, $4,8$ for $T=5$, and $4$ for $T=7$. '
    'This deliberately varied design avoids extrapolating from one horizon or distribution.')
add('Continuous-reference Bellman functions use a $0.04$ state grid and 48-node '
    'Gauss--Legendre expectation quadrature. Halving the grid from $0.08$ changes every '
    'reference value by less than $0.008$. Tree Bellman functions are constructed as exact '
    'convex PL functions, and the shared-capacity first stage is solved as an LP. '
    'For the application we use the valid sensitivity constants '
    r'$\bar L_{tj}=\sum_{k=t}^T(r_{kj}+h_{kj})$ and weights $w_t=\sum_jd_j\bar L_{tj}$. '
    r'The barycentric certificate is $2\mathfrak G\sum_tw_t$; for general supports it is '
    r'$W_1\sum_tw_t$. These are robust sensitivity bounds for this application, not claims '
    'that its structured value functions attain the general minimax hard instance. '
    r'Indeed, $\ell_{tj}$ has state-Lipschitz constant $r_{tj}+h_{tj}$; expectation '
    'and the replenishment inf-convolution do not increase this constant, which gives '
    r'the recursion for $\bar L_{tj}$. Recursive stage-by-stage transport coupling then '
    'yields the general-support certificate, while the CM case uses the sharper maximum-cell bound.')
add(r'\begin{figure}[htbp]')
add(r'\centering')
add(r'\includegraphics[width=\textwidth]{figures/inventory_application_comparison.pdf}')
add(r'\caption{Absolute relative value errors in the six-item inventory study. Monte Carlo and Sobol points average four replications; deterministic methods use one run. Non-barycentric trees can have signed errors of either sign, so absolute errors are plotted.}')
add(r'\label{fig:inventorycomparison}')
add(r'\end{figure}')
add(r'\begin{table}[htbp]')
add(r'\centering\small\setlength{\tabcolsep}{4pt}')
add(r'\caption{Inventory results aggregated over all instance--budget combinations. Errors and certificates are relative to $|V_1^*|$ and reported as percentages; runtime includes construction and tree solution.}')
add(r'\label{tab:inventorysummary}')
add(r'\begin{tabular}{lrrrrr}')
add(r'\toprule Method & mean error & median error & mean certificate & time (s) & max residual\\')
add(r'\midrule')
for summary_row in application_summary:
    add(f"{METHOD_LABELS[summary_row['method']]} & {fmt(100*summary_row['mean_relative_error'],3)}\\% & "
        f"{fmt(100*summary_row['median_relative_error'],3)}\\% & "
        f"{fmt(100*summary_row['mean_relative_certificate'],3)}\\% & "
        f"{fmt(summary_row['mean_seconds'],3)} & ${sci(summary_row['max_residual'],1)}$\\\\")
add(r'\bottomrule\end{tabular}')
add(r'\end{table}')
opt_summary = next(r for r in application_summary if r['method'] == 'Finite-B optimal')
f3_summary = next(r for r in application_summary if r['method'] == 'Equal f^(1/3)')
add(f"The exact tree has the smallest mean relative robust certificate, "
    f"{fmt(100*opt_summary['mean_relative_certificate'],3)}\\%. Its mean absolute "
    f"relative value error is {fmt(100*opt_summary['mean_relative_error'],3)}\\%, "
    f"compared with {fmt(100*f3_summary['mean_relative_error'],3)}\\% for equal "
    f"$f^{{1/3}}$-mass cells. Thus certificate optimality does not imply victory on "
    'every realized LP error. Forward reduction occasionally gives a smaller actual '
    'error but a much larger robust certificate. Monte Carlo shows substantial '
    'instance-to-instance variation, while Sobol is more stable at these small budgets. '
    r'The largest absolute Bellman-decomposition residual is '
    f"${sci(opt_summary['max_residual'],1)}$ for the exact method and "
    f"${sci(max(r['max_residual'] for r in application_summary),1)}$ overall. "
    'Detailed per-instance values, including signed errors, appear in Appendix~\\ref{app:extendedbenchmark}.')

add(r'\FloatBarrier')

main_numerics_path = ROOT / 'sections' / 'numerics.tex'
main_numerics_path.parent.mkdir(parents=True, exist_ok=True)
main_numerics_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

# Generate additional checks for the appendix.
lines = []
add = lines.append
add(r'\section{Additional deterministic numerical checks}\label{app:numerics}')
add('This section checks the retained identities and asymptotics on deterministic test instances. '
    'It is computational corroboration, not a substitute for the proofs. '
    'The accompanying script computes every entry, regenerates this section, and stops on a failed assertion. '
    r'It uses adaptive quadrature with absolute and relative tolerances $2\times10^{-12}$, '
    'root finding for equal-mass boundaries, and exact breakpoint enumeration for the convex PL optimizers.')
add('')
add(r'The common test density on $\Xi=[0,10]$ is')
add(r'\[')
add(r'f(x)=\frac{1+0.3\sin(6\pi x/10)}{10},\qquad 0\le x\le10,')
add(r'\]')
add('which is positive, continuously differentiable, and normalized. '
    'All trees below use conditional-mean atoms. Residual means the absolute difference between '
    'the directly evaluated expression and the stated identity.')

add(r'\subsection{Exact fixed-tree values and scalar asymptotics}')
add(r'For $\bar L=0.75$, Table~\ref{tab:fixednumeric} evaluates the maximizing hinge at the centroid '
    r'of a profile-maximizing cell. The irregular tree is included to check that no equal-width hypothesis is used.')
add(r'\begin{table}[htbp]')
add(r'\centering\small\setlength{\tabcolsep}{4.5pt}')
add(r'\caption{Direct hinge gaps and the exact fixed-tree identity for eight-cell trees.}')
add(r'\label{tab:fixednumeric}')
add(r'\begin{tabular}{lrrrrr}')
add(r'\toprule')
add(r'Tree & $i_{\max}$ & $\mathfrak G$ & direct gap & $2\bar L\mathfrak G$ & residual\\')
add(r'\midrule')
for name, b, imax, G, direct, residual in fixed_rows:
    add(f'{name} & {imax} & {fmt(G)} & {fmt(direct)} & {fmt(2*Lbar*G)} & ${sci(residual,1)}$\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add(r'Table~\ref{tab:sharpnumeric} uses equal $\sqrt f$-mass cells and compares with '
    r'$\Gamma(f)/8$, the limiting value of $B^2\mathfrak G$ in Theorem~\ref{thm:sharp}.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Convergence of equal-$\sqrt f$-mass trees.}')
add(r'\label{tab:sharpnumeric}')
add(r'\begin{tabular}{rrrr}')
add(r'\toprule')
add(r'$B$ & $\mathfrak G$ & $B^2\mathfrak G$ & $B^2\mathfrak G/(\Gamma/8)$\\')
add(r'\midrule')
for b, G, scaled, ratio in sharp_rows:
    add(f'{b} & ${sci(G)}$ & {fmt(scaled)} & {fmt(ratio)}\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add(r'For the smooth law, Table~\ref{tab:smoothnumeric} uses equal $f^{1/3}$-mass cells. '
    r'Its distortion $D_B=\sum_i\int_{R_i}(x-\xi_i)^2f(x)\,dx$ is compared with '
    r'$(\int f^{1/3})^3/12$. This direct construction avoids attributing a local Lloyd fixed point to a global optimum.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Squared quantization by equal-$f^{1/3}$-mass cells.}')
add(r'\label{tab:smoothnumeric}')
add(r'\begin{tabular}{rrrr}')
add(r'\toprule')
add(r'$B$ & $D_B$ & $B^2D_B$ & $B^2D_B/[(\int f^{1/3})^3/12]$\\')
add(r'\midrule')
for b, distortion, scaled, ratio in smooth_rows:
    add(f'{b} & ${sci(distortion)}$ & {fmt(scaled)} & {fmt(ratio)}\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add(r'\FloatBarrier')
add(r'\subsection{Multistage values and nonzero Bellman slacks}')
add(r'Table~\ref{tab:multinumeric} sums the directly integrated stage gaps for the additive hard instances '
    r'in \eqref{eq:hardcost}, using equal $\sqrt f$-mass trees and $\bar L_t=0.4+0.05t$. '
    'The two columns are independent evaluations of the same exact multistage value.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Finite-budget multistage minimax values on additive LP instances.}')
add(r'\label{tab:multinumeric}')
add(r'\begin{tabular}{rrrrr}')
add(r'\toprule')
add(r'$T$ & $B$ & direct sum & $2\sum_t\bar L_t\mathfrak G_t$ & residual\\')
add(r'\midrule')
for T, b, direct_sum, formula_sum, residual in multi_rows:
    add(f'{T} & {b} & {fmt(direct_sum)} & {fmt(formula_sum)} & ${sci(residual,1)}$\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add('A separate three-stage production example tests the path decomposition when decisions and slacks are nontrivial. '
    r'Let $\xi_2,\xi_3$ be independent uniform variables on $[0,1]$, each discretized by four equal-width cells. '
    r'With $p=2$, $K=0.8$, $r=1.2$, $a_1=0.35$, and $a_2=0.5$, define')
add(r'\begin{align*}')
add(r'V_3(s,\xi_3)&=p(\xi_3-Ks)_+,\\')
add(r'V_2(s,\xi_2)&=r(\xi_2-s)_++\min_{0\le x\le1}\{a_2x+\E V_3(x,\xi_3)\},\\')
add(r'V_1^*&=\min_{0\le x\le1}\{a_1x+\E V_2(x,\xi_2)\}.')
add(r'\end{align*}')
add('The discretized model replaces each expectation by its CM-tree expectation. '
    'These convex problems are LP-representable using nonnegative shortfall variables. '
    'Continuous uniform hinge expectations and tree expectations are evaluated exactly; '
    'all one-dimensional convex minimizers are checked at their breakpoints and endpoints.')
s = results['slack']
add(f"The resulting decisions are $x_1^*={fmt(s['x1c'])}$, $\\hat x_1={fmt(s['x1b'])}$, "
    f"$x_2^*={fmt(s['x2c'])}$, and $\\hat x_2={fmt(s['x2b'])}$. "
    f"Thus $\\sigma_1={fmt(s['sigma1'])}>0$ and $\\sigma_2={fmt(s['sigma2'])}>0$. "
    r'Table~\ref{tab:slacknumeric} verifies both the local recursion and the full path identity; '
    'omitting the slack sum would not reproduce the gap.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Exact three-stage Bellman-slack decomposition.}')
add(r'\label{tab:slacknumeric}')
add(r'\begin{tabular}{lr}')
add(r'\toprule')
add(r'Quantity & Value\\')
add(r'\midrule')
for label, key in [(r'$V_1^*$', 'V1c'), (r'$V_1^B$', 'V1b'),
                   (r'$V_1^*-V_1^B$', 'gap'),
                   (r'$\sum_{t=2}^3\E^{\mathrm{tree}}\varepsilon_t$', 'eps'),
                   (r'$\sum_{t=1}^2\E^{\mathrm{tree}}\sigma_t$', 'slacks'),
                   (r'Jensen sum $-$ slack sum', 'rhs')]:
    add(f'{label} & {fmt(s[key],9)}\\\\')
add(f'Absolute residual & ${sci(s["residual"],1)}$\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')

add(r'\FloatBarrier')
add(r'\subsection{Moreau-envelope limits}')
add(r'Finally, take $\psi(x)=|x-5|$ and use the abutment quantizer: two affine-region cells and '
    r'$B-2$ equal $f^{1/3}$-mass cells inside $S_\mu=[5-\mu,5+\mu]$. '
    r'The directly integrated gap agrees with the core second moment divided by $2\mu$ in every run. '
    r'In the fixed-$\mu$ rows, the ratio denominator is $\Ical_\mu^3/(24B^2)$; '
    r'in the small-$\mu$ rows it is $\mu^2C_K^3/(24B^2)$, with $C_K=2f(5)^{1/3}$.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Fixed-$\mu$ Moreau asymptotics and the separate small-$\mu$ limit.}')
add(r'\label{tab:moreaunumeric}')
add(r'\begin{tabular}{lrrrr}')
add(r'\toprule')
add(r'Regime & $\mu$ & $B$ & gap & ratio\\')
add(r'\midrule')
for b, gap_mu, ratio in moreau_fixed_rows:
    add(f'Fixed & $0.5$ & {b} & ${sci(gap_mu)}$ & {fmt(ratio)}\\\\')
add(r'\midrule')
for mu, gap_mu, ratio in moreau_small_rows:
    add(f'Small & ${fmt(mu,4)}$ & {small_B} & ${sci(gap_mu)}$ & {fmt(ratio)}\\\\')
add(r'\bottomrule')
add(r'\end{tabular}')
add(r'\end{table}')
add(r'The two Moreau experiments are deliberately separate: the first lets $B$ grow at fixed $\mu$, '
    r'whereas the second changes $\mu$ only after using a large fixed budget. '
    r'No joint $\mu,B$ limit is inferred.')
add(r'\FloatBarrier')

add(r'\subsection{Extended COR benchmark tables}\label{app:extendedbenchmark}')
add(r'Table~\ref{tab:referencegrid} records the continuous-reference grid check for '
    'the inventory applications. The fine grid is used for every reported value error '
    'and decomposition residual; the coarse calculation is an independent reference check.')
add(r'\begin{table}[htbp]')
add(r'\centering\small')
add(r'\caption{Inventory reference values on successive state grids.}')
add(r'\label{tab:referencegrid}')
add(r'\begin{tabular}{lrrr}')
add(r'\toprule Instance & grid $0.04$ & grid $0.08$ & absolute change\\ \midrule')
for name, fine_value, coarse_value, change in reference_grid_rows:
    add(f'{name} & {fmt(fine_value,6)} & {fmt(coarse_value,6)} & {fmt(change,6)}\\\\')
add(r'\bottomrule\end{tabular}\end{table}')

add(r'\begin{table}[htbp]')
add(r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}')
add(r'\caption{Detailed inventory results at $B=8$ for the first three instances and $B=4$ for Uniform-7. Sampling rows average four replications; their residual column is the maximum.}')
add(r'\label{tab:inventorydetailed}')
add(r'\begin{tabular}{llrrrrr}')
add(r'\toprule Instance & Method & signed error & absolute error & certificate & time (s) & residual\\ \midrule')
for instance_name in application_instances:
    selected_budget = 4 if instance_name == 'Uniform-7' else 8
    for method in METHODS:
        selected = [r for r in application_rows if r['instance'] == instance_name
                    and r['B'] == selected_budget and r['method'] == method]
        signed = float(np.mean([r['signed_error'] for r in selected]))
        absolute = float(np.mean([r['absolute_error'] for r in selected]))
        certificate = float(np.mean([r['certificate'] for r in selected]))
        seconds = float(np.mean([r['seconds'] for r in selected]))
        residual = float(np.max([r['residual'] for r in selected]))
        add(f'{instance_name} & {METHOD_LABELS[method]} & {fmt(signed,5)} & {fmt(absolute,5)} & '
            f'{fmt(certificate,4)} & {fmt(seconds,3)} & ${sci(residual,1)}$\\\\')
    if instance_name != application_instances[-1]:
        add(r'\midrule')
add(r'\bottomrule\end{tabular}\end{table}')

add(r'\FloatBarrier')
add(r'\begin{table}[htbp]')
add(r'\centering\footnotesize\setlength{\tabcolsep}{4pt}')
add(r'\caption{Detailed construction results at $B=16$. Sampling rows average six replications.}')
add(r'\label{tab:densitydetailed}')
add(r'\begin{tabular}{llrrrr}')
add(r'\toprule Density & Method & $\mathfrak G/\mathfrak G_B^*$ & $W_1$ & $W_1$ ratio & time (ms)\\ \midrule')
for density_name in ['Uniform', 'Sine', 'TruncNorm', 'Bimodal']:
    for method in METHODS:
        selected = [r for r in density_rows if r['density'] == density_name
                    and r['B'] == 16 and r['method'] == method]
        g_values = [r['profile_ratio'] for r in selected if r['kind'] == 'CM']
        g_text = fmt(float(np.mean(g_values))) if g_values else '--'
        w1 = float(np.mean([r['W1'] for r in selected]))
        ratio = float(np.mean([r['W1_ratio'] for r in selected]))
        ms = 1000.0 * float(np.mean([r['seconds'] for r in selected]))
        add(f'{density_name} & {METHOD_LABELS[method]} & {g_text} & {fmt(w1,6)} & '
            f'{fmt(ratio,4)} & {fmt(ms,2)}\\\\')
    if density_name != 'Bimodal':
        add(r'\midrule')
add(r'\bottomrule\end{tabular}\end{table}')
add(r'\FloatBarrier')

supplement_path = ROOT / 'sections' / 'numeric_supplement.tex'
supplement_path.parent.mkdir(parents=True, exist_ok=True)
supplement_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print('Completed all original and COR-focused experiments')
for key in ['fixed', 'sharp', 'smooth', 'multi', 'slack', 'moreau_fixed', 'moreau_small']:
    print(key, results[key])
print(f'Wrote {main_numerics_path}')
print(f'Wrote {supplement_path}')
