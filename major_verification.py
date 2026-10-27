"""Independent numerical verification for the strengthened COR manuscript.

Requires NumPy, SciPy, and Matplotlib. This script recomputes the scalar Markov certificate
and allocation checks, forward/backward W1 comparisons, exact-construction scaling through
B=128, and the real-consumption-calibrated chronological policy study. It also audits the
weighted transport calculation and solves independent deterministic-equivalent LPs. Running
it writes sections/major_numerics.tex and sections/major_supplement.tex and regenerates the
two major-revision figures in PDF and EPS formats. MAJOR_USE_CACHE=1 rebuilds document
outputs from existing local results; ordinary execution recomputes everything without caches.
Run verification.py separately for the original one-stage, multistage, and inventory results.
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


def application_certificate(inst, atoms, kind, profile, dist):
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
                    w1 = transport_w1(dist, atoms)
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
                    certificate = application_certificate(inst, atoms, kind, profile, inst.dist)
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


# ===== Major COR revision: Markov trees, stronger reduction, scaling, real data =====
import csv
import os
import pickle
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / 'figures'
FIGDIR.mkdir(exist_ok=True)
(ROOT / 'sections').mkdir(exist_ok=True)
(ROOT / 'data').mkdir(exist_ok=True)


def backward_reduction(dist, cells, grid_size=384):
    """Dupačová--Heitsch--Römisch backward reduction under the L1 metric."""
    grid = dist.ppf((np.arange(grid_size) + 0.5) / grid_size)
    weights = np.full(grid_size, 1.0 / grid_size)
    distances = np.abs(grid[:, None] - grid[None, :])
    selected = np.ones(grid_size, dtype=bool)
    for _ in range(grid_size - cells):
        ids = np.flatnonzero(selected)
        deleted = ~selected
        dsel = distances[:, ids]
        nearest_pos = np.argmin(dsel, axis=1)
        nearest = dsel[np.arange(grid_size), nearest_pos]
        nearest_ids = ids[nearest_pos]
        if len(ids) > 1:
            second = np.partition(dsel, 1, axis=1)[:, 1]
        else:
            second = np.full(grid_size, np.inf)
        base = float(np.sum(weights[deleted] * nearest[deleted]))
        objectives = np.full(grid_size, np.inf)
        for ell in ids:
            affected = deleted & (nearest_ids == ell)
            value = base + weights[ell] * nearest[ell]
            value += float(np.sum(weights[affected] * (second[affected] - nearest[affected])))
            objectives[ell] = value
        selected[int(np.argmin(objectives))] = False
    atoms = grid[selected]
    assignment = np.argmin(np.abs(grid[:, None] - atoms[None, :]), axis=1)
    probs = np.array([weights[assignment == j].sum() for j in range(cells)])
    order = np.argsort(atoms)
    return atoms[order], probs[order]


def make_tree_with_backward(dist, method, cells, seed=0, grid_size=384):
    if method == 'Backward W1':
        atoms, probs = backward_reduction(dist, cells, grid_size)
        return atoms, probs, 'transport', float('nan')
    if method == 'Forward W1':
        atoms, probs = forward_reduction(dist, cells, grid_size)
        return atoms, probs, 'transport', float('nan')
    return make_tree(dist, method, cells, seed)


@dataclass(eq=False)
class MarkovNode:
    stage: int
    atom: float
    prob: float
    children: list
    dist: object = None
    atoms: object = None
    probs: object = None
    local_profile: float = float('nan')
    local_w1: float = float('nan')
    budget: int = 0

    def __post_init__(self):
        self.K = {}
        self.H = {}


class AR1MarkovModel:
    """Scalar Markov process with truncated-normal conditional distributions."""
    def __init__(self, alpha, rho, sigma, initial=5.0, name='AR1', sigma_slope=0.0):
        self.alpha, self.rho, self.sigma = alpha, rho, sigma
        self.initial, self.name = initial, name
        self.sigma_slope = sigma_slope

    def conditional(self, parent):
        mu = float(np.clip(self.alpha + self.rho * parent, 0.05, 9.95))
        sigma = self.sigma * math.exp(self.sigma_slope * math.tanh((parent - 5.0) / 2.0))
        return TruncNormal(mu, sigma, self.name)


def exact_budget_allocation(dists, probabilities, total_budget, iterations=38):
    """Exact integer DP for one stage with a fixed parent set."""
    n = len(dists)
    if total_budget < n:
        raise ValueError('Every parent needs at least one child')
    maximum = total_budget - n + 1
    profiles, trees = [], []
    for dist in dists:
        local_profiles, local_trees = [np.nan], [None]
        for b in range(1, maximum + 1):
            edges = optimal_tree_dist(dist, b, iterations=iterations)
            local_trees.append(edges)
            local_profiles.append(max(z['g'] for z in cell_data_dist(dist, edges)))
        profiles.append(np.array(local_profiles))
        trees.append(local_trees)
    dp = np.full((n + 1, total_budget + 1), np.inf)
    choice = np.zeros((n + 1, total_budget + 1), dtype=int)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for total in range(i, total_budget + 1):
            for b in range(1, min(maximum, total - i + 1) + 1):
                if dp[i - 1, total - b] < np.inf:
                    value = dp[i - 1, total - b] + probabilities[i - 1] * profiles[i - 1][b]
                    if value < dp[i, total]:
                        dp[i, total] = value
                        choice[i, total] = b
    budgets = [0] * n
    remaining = total_budget
    for i in range(n, 0, -1):
        budgets[i - 1] = int(choice[i, remaining])
        remaining -= budgets[i - 1]
    assert sum(budgets) == total_budget and min(budgets) >= 1
    return budgets, [trees[i][budgets[i]] for i in range(n)], float(dp[n, total_budget])


def build_markov_tree(model, method, average_budget, horizon=4, seed=0,
                      allocate=False, grid_size=192, iterations=38):
    root = MarkovNode(1, float(model.initial), 1.0, [])
    parents = [root]
    for stage in range(2, horizon + 1):
        dists = [model.conditional(node.atom) for node in parents]
        probabilities = np.array([node.prob for node in parents])
        probabilities = probabilities / probabilities.sum()
        budgets = [average_budget] * len(parents)
        edges_list = [None] * len(parents)
        if allocate:
            if method != 'Finite-B optimal':
                raise ValueError('Exact allocation applies to the exact local optimizer')
            budgets, edges_list, _ = exact_budget_allocation(
                dists, probabilities, len(parents) * average_budget, iterations)
        children = []
        for i, (parent, dist, b) in enumerate(zip(parents, dists, budgets)):
            atoms, probs, kind, profile = make_tree_with_backward(
                dist, method, b, seed=seed + 1009 * stage + 37 * i, grid_size=grid_size)
            if edges_list[i] is not None:
                atoms, probs = cm_atoms(dist, edges_list[i])
                profile = max(z['g'] for z in cell_data_dist(dist, edges_list[i]))
            parent.dist, parent.atoms, parent.probs = dist, atoms, probs
            parent.local_profile = profile
            parent.local_w1 = transport_w1(dist, atoms, probs)
            parent.budget = b
            parent.children = [MarkovNode(stage, float(a), parent.prob * float(q), [])
                               for a, q in zip(atoms, probs)]
            children.extend(parent.children)
        parents = children
    return root


def markov_stage_parents(root):
    stages = {1: [root]}
    for stage in range(2, 20):
        parents = [node for parent in stages[stage - 1] for node in parent.children]
        if not parents:
            break
        stages[stage] = parents
    return stages


def markov_certificate(root, stage_weights, kind):
    value = 0.0
    for stage, parents in markov_stage_parents(root).items():
        if stage >= max(markov_stage_parents(root)) or stage == 1 and not root.children:
            continue
        for parent in parents:
            local = (2.0 * parent.local_profile if kind == 'CM' else parent.local_w1)
            value += stage_weights[stage + 1] * parent.prob * local
    return float(value)


def markov_additive_tree(root, stage_functions):
    stages = markov_stage_parents(root)
    value = 0.0
    for stage in range(2, max(stages) + 1):
        for parent in stages[stage - 1]:
            value += parent.prob * float(np.sum(parent.probs * stage_functions[stage](parent.atoms)))
    return value


def markov_additive_continuous(model, stage_functions, horizon):
    """Direct nested adaptive quadrature for additive Markov costs."""
    def continuation(stage, parent):
        if stage > horizon:
            return 0.0
        dist = model.conditional(parent)
        def integrand(x):
            return (stage_functions[stage](x) + continuation(stage + 1, x)) * dist.pdf(x)
        return quad(integrand, dist.A, dist.B, epsabs=3e-9, epsrel=3e-9, limit=120)[0]
    return continuation(2, model.initial)


def next_K_markov(Hfunctions, atoms, probs, r, h):
    if all(H is None for H in Hfunctions):
        # Preserve both asymptotic hinge slopes, not merely the segment between
        # demand atoms; otherwise extrapolation can create spurious negative costs.
        knots = [np.r_[atoms[0] - 1.0, atoms, atoms[-1] + 1.0]]
    else:
        knots = [atoms]
    for H in Hfunctions:
        if H is not None:
            knots.extend([atoms + k for k in H.x])
    knots = np.array(sorted(set(np.round(np.concatenate(knots), 11))))
    values = []
    for z in knots:
        value = expected_hinge_tree(float(z), atoms, probs, r, h)
        for H, probability, atom in zip(Hfunctions, probs, atoms):
            if H is not None:
                value += float(probability * H(z - atom))
        values.append(value)
    return PLFunction(knots, values)


def markov_tree_dp(inst, root):
    """Exact convex PL Bellman recursion on a conditional scenario tree."""
    for j in range(inst.n):
        def fill(parent):
            stage = parent.stage + 1
            if j in parent.K:
                return parent.K[j]
            Hchildren = []
            for child in parent.children:
                if stage < inst.T:
                    Kchild = fill(child)
                    Hchild = procurement_transform(Kchild, inst.order[stage - 2, j], inst.capacity[j])
                    child.H[j] = Hchild
                    Hchildren.append(Hchild)
                else:
                    Hchildren.append(None)
            atoms = inst.loads[j] * parent.atoms
            parent.K[j] = next_K_markov(Hchildren, atoms, parent.probs,
                                         inst.r[stage - 2, j], inst.h[stage - 2, j])
            return parent.K[j]
        fill(root)
    Ks = [[root.K[j]] for j in range(inst.n)]
    decision, value = inst.first_stage_tree(Ks)
    return decision, value


def simulate_markov_policy(inst, root, initial, observations):
    """Nearest-node lifting of the tree policy to an observed scalar path."""
    total = float(np.dot(inst.initial_cost, initial))
    states = initial.copy()
    parent = root
    for stage, xi in enumerate(observations, start=2):
        child = min(parent.children, key=lambda node: abs(node.atom - xi))
        for j in range(inst.n):
            demand = inst.loads[j] * xi
            total += inst.r[stage - 2, j] * max(demand - states[j], 0.0)
            total += inst.h[stage - 2, j] * max(states[j] - demand, 0.0)
            y = states[j] - demand
            if stage < inst.T:
                order = procurement_policy(child.H[j], y)
                total += inst.order[stage - 2, j] * order
                states[j] = y + order
        parent = child
    return total


def save_major_figure(fig, stem):
    for suffix in ('pdf', 'eps'):
        fig.savefig(FIGDIR / f'{stem}.{suffix}', format=suffix,
                    bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def pct(x):
    return rf'{100*x:.3f}\%'


def sciformat(x):
    if x == 0:
        return '0'
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10 ** exponent
    return f'{mantissa:.1f}\\times10^{{{exponent}}}'


def run_markov_certificate_study():
    model = AR1MarkovModel(1.75, 0.65, 1.05, initial=4.4, name='Synthetic AR1',
                           sigma_slope=0.65)
    methods = ['Finite-B optimal', 'Equal sqrt-f', 'Forward W1', 'Backward W1',
               'Monte Carlo', 'Sobol']
    rows = []
    for horizon in [3, 4]:
        stages = list(range(2, horizon + 1))
        weights = {t: 0.32 + 0.07 * (horizon - t) for t in stages}
        hinges = {t: 3.4 + 0.8 * (t - 2) for t in stages}
        stage_functions = {
            t: (lambda x, L=weights[t], a=hinges[t]: -L * x + 2 * L * np.maximum(x - a, 0.0))
            for t in stages}
        continuous = markov_additive_continuous(model, stage_functions, horizon)
        for method in methods:
            reps = 4 if method in ('Monte Carlo', 'Sobol') else 1
            for rep in range(reps):
                for allocated in ([False, True] if method == 'Finite-B optimal' and horizon == 3 else [False]):
                    start = time.perf_counter()
                    tree = build_markov_tree(model, method, 2, horizon,
                                             seed=12000 + 101 * rep, allocate=allocated)
                    elapsed = time.perf_counter() - start
                    kind = 'CM' if method in ('Finite-B optimal', 'Equal sqrt-f') else 'transport'
                    certificate = markov_certificate(tree, weights, kind)
                    tree_value = markov_additive_tree(tree, stage_functions)
                    gap = continuous - tree_value
                    if kind == 'CM':
                        assert gap >= -2e-7
                    assert abs(gap) <= certificate + 2e-6
                    # Independently evaluate the conditional local certificate terms.
                    for stage, parents in markov_stage_parents(tree).items():
                        for parent in parents:
                            if parent.children:
                                assert abs(parent.probs.sum() - 1.0) < 2e-12
                                assert parent.local_w1 >= -1e-12
                    rows.append(dict(horizon=horizon, method=method, allocated=allocated,
                                     rep=rep, continuous=continuous, tree_value=tree_value,
                                     gap=gap, absolute_gap=abs(gap), certificate=certificate,
                                     seconds=elapsed, nodes=sum(len(v) for v in markov_stage_parents(tree).values())))
    # Exhaustive integer-allocation check at a two-parent stage.
    dists = [model.conditional(3.7), model.conditional(6.2)]
    probs = np.array([0.35, 0.65])
    budgets, edges, objective = exact_budget_allocation(dists, probs, 6)
    brute = np.inf
    for b1 in range(1, 6):
        b2 = 6 - b1
        value = 0.0
        for dist, prob, b in zip(dists, probs, [b1, b2]):
            e = optimal_tree_dist(dist, b, iterations=38)
            value += prob * max(z['g'] for z in cell_data_dist(dist, e))
        brute = min(brute, value)
    assert abs(objective - brute) < 2e-9, (objective, brute)
    equal_objective = 0.0
    for dist, prob in zip(dists, probs):
        e = optimal_tree_dist(dist, 3, iterations=38)
        equal_objective += prob * max(z['g'] for z in cell_data_dist(dist, e))
    assert objective <= equal_objective + 2e-9
    return rows, dict(budgets=budgets, objective=objective, brute=brute,
                      equal_objective=equal_objective)


def run_reduction_and_scaling():
    distributions = [UniformDist(), SineDist(), TruncNormal(5.4, 1.45, 'TruncNorm'), MixtureDist()]
    reduction_rows = []
    for dist in distributions:
        for b in [4, 8, 16, 32]:
            for method in ['Forward W1', 'Backward W1']:
                start = time.perf_counter()
                atoms, probs, _, _ = make_tree_with_backward(dist, method, b, grid_size=384)
                elapsed = time.perf_counter() - start
                assert len(np.unique(atoms)) == b and abs(probs.sum() - 1) < 2e-12
                reduction_rows.append(dict(density=dist.name, B=b, method=method,
                                           W1=transport_w1(dist, atoms, probs), seconds=elapsed))
    scaling_rows = []
    for dist in [SineDist(), MixtureDist()]:
        for b in [4, 8, 16, 32, 64, 128]:
            start = time.perf_counter()
            edges = optimal_tree_dist(dist, b, iterations=42)
            elapsed = time.perf_counter() - start
            profile = max(z['g'] for z in cell_data_dist(dist, edges))
            sqrt_edges = equal_mass_dist(dist, b, 0.5)
            sqrt_profile = max(z['g'] for z in cell_data_dist(dist, sqrt_edges))
            assert profile <= sqrt_profile + 2e-8
            scaling_rows.append(dict(density=dist.name, B=b, profile=profile,
                                     sqrt_profile=sqrt_profile, seconds=elapsed))
    return reduction_rows, scaling_rows


def load_real_consumption():
    path = ROOT / 'data' / 'macrodata_realcons.csv'
    if not path.exists():
        raise FileNotFoundError('The documented data/macrodata_realcons.csv file is required')
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    years = np.array([int(r['year']) for r in rows])
    quarters = np.array([int(r['quarter']) for r in rows])
    realcons = np.array([float(r['realcons']) for r in rows])
    growth = 100.0 * np.diff(np.log(realcons))
    labels = np.array([f'{y}Q{q}' for y, q in zip(years[1:], quarters[1:])])
    assert len(growth) == 202 and labels[0] == '1959Q2' and labels[-1] == '2009Q3'
    assert labels[139] == '1994Q1' and labels[140] == '1994Q2'
    return realcons, growth, labels


def fit_real_markov(growth, training=140):
    train = growth[:training]
    center, scale = train.mean(), train.std(ddof=0)
    xi = np.clip(5.0 + 1.5 * (growth - center) / scale, 0.05, 9.95)
    x, y = xi[:training - 1], xi[1:training]
    design = np.column_stack([np.ones(len(x)), x])
    alpha, rho = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([alpha, rho])
    sigma = residual.std(ddof=2)
    return xi, dict(center=center, scale=scale, alpha=alpha, rho=rho, sigma=sigma,
                    training=training, marginal_mean=xi[:training].mean(),
                    marginal_sigma=xi[:training].std(ddof=1))


def run_real_data_study(max_origins=20, replications=2):
    realcons, growth, labels = load_real_consumption()
    xi, fit = fit_real_markov(growth)
    training, horizon = fit['training'], 4
    origins = np.linspace(training, len(xi) - horizon + 1, max_origins, dtype=int)
    origins = np.unique(origins)
    methods = ['Markov optimal', 'Independent optimal', 'Equal sqrt-f',
               'Forward W1', 'Backward W1', 'Monte Carlo', 'Sobol']
    inst = InventoryInstance('RealCons-4', UniformDist(), horizon, seed=4)
    weights = {t: float(sum(inst.loads[j] * inst.L[t - 2, j] for j in range(inst.n)))
               for t in range(2, horizon + 1)}
    rows = []
    for origin in origins:
        observations = xi[origin:origin + horizon - 1]
        for method in methods:
            reps = replications if method in ('Monte Carlo', 'Sobol') else 1
            for rep in range(reps):
                if method == 'Independent optimal':
                    model = AR1MarkovModel(fit['marginal_mean'], 0.0, fit['marginal_sigma'],
                                           initial=xi[origin - 1], name='Independent real-data fit')
                    local_method = 'Finite-B optimal'
                else:
                    model = AR1MarkovModel(fit['alpha'], fit['rho'], fit['sigma'],
                                           initial=xi[origin - 1], name='Real-data AR1')
                    local_method = 'Finite-B optimal' if method == 'Markov optimal' else method
                for b in [2, 3]:
                    start = time.perf_counter()
                    tree = build_markov_tree(model, local_method, b, horizon,
                                             seed=15000 + 13 * origin + 97 * rep,
                                             allocate=False, grid_size=192)
                    construction = time.perf_counter() - start
                    start = time.perf_counter()
                    initial, model_value = markov_tree_dp(inst, tree)
                    solve = time.perf_counter() - start
                    realized = simulate_markov_policy(inst, tree, initial, observations)
                    kind = 'CM' if method in ('Markov optimal', 'Independent optimal', 'Equal sqrt-f') else 'transport'
                    certificate = markov_certificate(tree, weights, kind)
                    rows.append(dict(origin=int(origin), label=labels[origin], method=method,
                                     rep=rep, B=b, realized=realized, model_value=model_value,
                                     certificate=certificate, seconds=construction + solve,
                                     construction_seconds=construction, solve_seconds=solve,
                                     initial=initial, observations=observations.copy()))
        print('Completed real-data origin', labels[origin], flush=True)
    # Paired standard errors use one averaged value per origin and method/budget.
    summaries = []
    for method in methods:
        for b in [2, 3]:
            selected = [r for r in rows if r['method'] == method and r['B'] == b]
            per_origin = []
            for origin in origins:
                values = [r['realized'] for r in selected if r['origin'] == origin]
                per_origin.append(float(np.mean(values)))
            exact = []
            for origin in origins:
                values = [r['realized'] for r in rows if r['method'] == 'Markov optimal'
                          and r['B'] == b and r['origin'] == origin]
                exact.append(float(np.mean(values)))
            differences = np.array(per_origin) - np.array(exact)
            summaries.append(dict(method=method, B=b, mean_cost=float(np.mean(per_origin)),
                                  se_cost=float(np.std(per_origin, ddof=1) / math.sqrt(len(per_origin))),
                                  mean_certificate=float(np.mean([r['certificate'] for r in selected])),
                                  mean_seconds=float(np.mean([r['seconds'] for r in selected])),
                                  paired_difference=float(np.mean(differences)),
                                  paired_se=float(np.std(differences, ddof=1) / math.sqrt(len(differences))),
                                  wins_vs_exact=int(np.sum(differences < -1e-9)),
                                  origins=len(per_origin)))
    return rows, summaries, dict(realcons=realcons, growth=growth, labels=labels,
                                 xi=xi, fit=fit, origins=origins)


def deterministic_equivalent_markov(inst, root):
    """Independent deterministic-equivalent LP audit of the PL tree recursion."""
    variables = []
    bounds = []
    objective = []

    def new(cost=0.0, lb=0.0, ub=None):
        idx = len(variables)
        variables.append(f'x{idx}')
        bounds.append((lb, ub))
        objective.append(cost)
        return idx

    initial = [new(inst.initial_cost[j], 0.0, inst.capacity[j]) for j in range(inst.n)]
    A_ub, b_ub = [], []
    row = {initial[j]: 1.0 for j in range(inst.n)}
    A_ub.append(row)
    b_ub.append(inst.shared_capacity)
    A_eq, b_eq = [], []
    node_variables = {}
    stages = markov_stage_parents(root)
    for stage in range(2, inst.T + 1):
        for parent in stages[stage - 1]:
            for child_index, node in enumerate(parent.children):
                nv = {}
                demand_base = node.atom
                for j in range(inst.n):
                    shortage = new(inst.r[stage - 2, j] * node.prob, 0.0, None)
                    holding = new(inst.h[stage - 2, j] * node.prob, 0.0, None)
                    order = None if stage == inst.T else new(
                        inst.order[stage - 2, j] * node.prob, 0.0, inst.capacity[j])
                    state = None if stage == inst.T else new(0.0, None, None)
                    # z = previous z - demand + shortage - holding + order.
                    if stage == 2:
                        previous = initial[j]
                    else:
                        previous = node_variables[(parent.stage - 1, parent.atom, id(parent))][j]['state']
                    demand = inst.loads[j] * demand_base
                    # Positive/negative inventory parts at demand realization.
                    # This terminal balance is essential: omitting it would allow
                    # uncosted backlog to disappear at the terminal nodes.
                    A_eq.append({shortage: 1.0, holding: -1.0, previous: 1.0})
                    b_eq.append(demand)
                    if state is not None:
                        row = {state: 1.0, previous: -1.0}
                        if order is not None:
                            row[order] = -1.0
                        A_eq.append(row)
                        b_eq.append(-demand)
                    nv[j] = dict(shortage=shortage, holding=holding, order=order, state=state)
                node_variables[(node.stage - 1, node.atom, id(node))] = nv
    c = np.array(objective)

    def expand(rows, n):
        from scipy.sparse import lil_matrix
        matrix = lil_matrix((len(rows), n))
        for i, row in enumerate(rows):
            for j, value in row.items():
                matrix[i, j] = value
        return matrix.tocsr()

    res = linprog(c, A_ub=expand(A_ub, len(c)), b_ub=np.array(b_ub),
                  A_eq=expand(A_eq, len(c)) if A_eq else None,
                  b_eq=np.array(b_eq) if b_eq else None,
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    return res.x[initial], float(res.fun)


def independent_numerical_audits(real_data):
    # Weighted continuous-to-discrete W1 against a dense empirical optimal-transport value.
    from scipy.stats import wasserstein_distance
    dist = MixtureDist()
    atoms, probs, _, _ = make_tree(dist, 'Monte Carlo', 5, seed=777)
    exact = transport_w1(dist, atoms, probs)
    grid = dist.ppf((np.arange(12000) + 0.5) / 12000)
    empirical = wasserstein_distance(grid, atoms, np.full(len(grid), 1 / len(grid)), probs)
    assert abs(exact - empirical) < 2e-4, (exact, empirical)
    # Independent LP verification on a small conditional tree.
    model = AR1MarkovModel(1.75, 0.65, 1.05, initial=4.4, sigma_slope=0.65)
    tree = build_markov_tree(model, 'Finite-B optimal', 2, horizon=3)
    inst = InventoryInstance('Audit', UniformDist(), 3, seed=0)
    u_dp, v_dp = markov_tree_dp(inst, tree)
    u_lp, v_lp = deterministic_equivalent_markov(inst, tree)
    assert abs(v_dp - v_lp) < 2e-7 * max(1.0, abs(v_lp)), (v_dp, v_lp)
    assert np.max(np.abs(u_dp - u_lp)) < 2e-6
    synthetic_dp_value, synthetic_lp_value = v_dp, v_lp
    # Repeat on one real-data-calibrated origin.
    origin = int(real_data['origins'][0])
    fit, xi = real_data['fit'], real_data['xi']
    real_model = AR1MarkovModel(fit['alpha'], fit['rho'], fit['sigma'], xi[origin - 1])
    real_tree = build_markov_tree(real_model, 'Finite-B optimal', 2, horizon=4, grid_size=192)
    real_inst = InventoryInstance('Real audit', UniformDist(), 4, seed=4)
    u_dp, v_dp = markov_tree_dp(real_inst, real_tree)
    u_lp, v_lp = deterministic_equivalent_markov(real_inst, real_tree)
    assert abs(v_dp - v_lp) < 3e-7 * max(1.0, abs(v_lp)), (v_dp, v_lp)
    return dict(W1_exact=exact, W1_empirical=empirical,
                synthetic_DP=synthetic_dp_value, synthetic_LP=synthetic_lp_value,
                real_DP=v_dp, real_LP=v_lp,
                max_initial_difference=float(np.max(np.abs(u_dp - u_lp))))


def generate_major_outputs(markov_rows, allocation_audit, reduction_rows, scaling_rows,
                           real_rows, real_summaries, real_data, audits):
    colors = ['#24547a', '#a04b4b', '#34846f', '#86629b', '#c18b2d', '#555555', '#b05c91']
    marks = ['o', 's', '^', 'D', 'v', 'x', '>']
    method_names = ['Finite-B optimal', 'Equal sqrt-f', 'Forward W1', 'Backward W1',
                    r'Monte Carlo', 'Sobol']
    # Figure 6: conditional robust certificates and construction scaling.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    for j, method in enumerate(method_names):
        x, y = [], []
        for horizon in [3, 4]:
            selected = [r for r in markov_rows if r['method'] == method
                        and r['horizon'] == horizon and not r['allocated']]
            x.append(horizon)
            y.append(np.mean([r['certificate'] for r in selected]))
        axes[0].plot(x, y, color=colors[j], marker=marks[j], label=method)
    axes[0].set_xticks([3, 4])
    axes[0].set_xlabel('$T$')
    axes[0].set_ylabel('conditional robust certificate')
    axes[0].set_title('(a) Scalar Markov trees')
    for j, density in enumerate(['Sine', 'Bimodal']):
        selected = [r for r in scaling_rows if r['density'] == density]
        axes[1].plot([r['B'] for r in selected], [r['seconds'] for r in selected],
                     color=colors[j], marker=marks[j], label=density)
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].set_xticks([4, 8, 16, 32, 64, 128])
    axes[1].set_xticklabels(['4', '8', '16', '32', '64', '128'])
    axes[1].set_xlabel('$B$')
    axes[1].set_ylabel('construction time (s)')
    axes[1].set_title('(b) Exact construction scaling')
    handles = [plt.Line2D([0], [0], color=colors[j], marker=marks[j], label=m)
               for j, m in enumerate(method_names)]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.19, 1, 1), w_pad=2.0)
    save_major_figure(fig, 'markov_certificates_and_scaling')

    # Figure 7: actual series split and paired held-out policy costs.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), gridspec_kw={'width_ratios': [1.05, 1.35]})
    growth = real_data['growth']
    training = real_data['fit']['training']
    axes[0].plot(np.arange(training), growth[:training], color=colors[0], linewidth=0.8,
                 label='training')
    axes[0].plot(np.arange(training, len(growth)), growth[training:], color=colors[1],
                 linewidth=0.9, label='held out')
    axes[0].axvline(training - 0.5, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('quarterly observation index')
    axes[0].set_ylabel('real-consumption log growth (%)')
    axes[0].set_title('(a) Chronological data split')
    axes[0].legend(frameon=False, fontsize=8)
    real_methods = ['Markov optimal', 'Independent optimal', 'Equal sqrt-f',
                    r'Forward W1', 'Backward W1', 'Monte Carlo', 'Sobol']
    ypos = np.arange(len(real_methods))
    for k, b in enumerate([2, 3]):
        values = [next(s for s in real_summaries if s['method'] == m and s['B'] == b)['mean_cost']
                  for m in real_methods]
        errors = [next(s for s in real_summaries if s['method'] == m and s['B'] == b)['se_cost']
                  for m in real_methods]
        axes[1].barh(ypos + (k - 0.5) * 0.34, values, height=0.32,
                     color=colors[k], label=f'$B={b}$', xerr=errors, capsize=2,
                     error_kw={'linewidth': 0.6})
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(['Markov exact', 'Independent exact', r'Equal $\sqrt{f}$',
                             r'Forward $W_1$', 'Backward $W_1$', 'Monte Carlo', 'Sobol'], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('mean held-out realized cost')
    axes[1].set_title('(b) Four-stage inventory policies')
    axes[1].legend(frameon=False)
    fig.tight_layout(w_pad=2.0)
    save_major_figure(fig, 'real_data_policy_comparison')

    # Main numerical section.
    lines = []
    add = lines.append
    add(r'\FloatBarrier')
    add(r'\subsection{Scalar Markov trees and conditional budget allocation}')
    add(r'We now test Theorems~\ref{thm:markovcertificate} and~\ref{thm:budgetdp}. '
        r'The synthetic process has conditional law $\xi_{t+1}\mid\xi_t=x$ given by a '
        r'normal distribution truncated to $[0,10]$, with mean $1.75+0.65x$ and '
        r'standard deviation $1.05\exp\{0.65\tanh((x-5)/2)\}$. Thus both conditional '
        r'locations and dispersions vary across parents. The horizons are $T=3,4$, '
        r'with an average of two children per parent. Additive stochastic-linear cost '
        r'functions are integrated directly by nested quadrature and independently on '
        r'the tree. Every CM gap is nonnegative and every signed gap, including those '
        r'of the sampling and transport methods, satisfies its stated certificate.')
    add(r'\begin{figure}[htbp]')
    add(r'\centering')
    add(r'\includegraphics[width=\textwidth]{figures/markov_certificates_and_scaling.pdf}')
    add(r'\caption{Conditional Markov certificates and exact-construction timings. Panel (a) averages sampling replications. Panel (b) uses one Python process and is an empirical scaling study, not a complexity theorem.}')
    add(r'\label{fig:markovscaling}')
    add(r'\end{figure}')
    add(r'\begin{table}[htbp]')
    add(r'\centering\small\setlength{\tabcolsep}{4.5pt}')
    add(r'\caption{Markov-tree results averaged over horizons and replications.}')
    add(r'\label{tab:markovsummary}')
    add(r'\begin{tabular}{lrrrr}')
    add(r'\toprule Method & mean certificate & max $|$gap$|$ & mean time (s) & max nodes\\')
    add(r'\midrule')
    for method in method_names:
        selected = [r for r in markov_rows if r['method'] == method and not r['allocated']]
        add(fr"{method.replace('Finite-B optimal', 'Finite-$B$ optimal').replace('sqrt-f', 'sqrt-$f$').replace('W1', '$W_1$')} & "
            rf"{np.mean([r['certificate'] for r in selected]):.6f} & "
            rf"{max(r['absolute_gap'] for r in selected):.6f} & "
            rf"{np.mean([r['seconds'] for r in selected]):.3f} & {max(r['nodes'] for r in selected)}\\")
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')
    allocation_reduction = 100 * (allocation_audit['equal_objective'] - allocation_audit['objective']) / allocation_audit['equal_objective']
    add(fr"In a separate two-parent example with total child budget six, exhaustive "
        rf"allocation and dynamic programming both select {allocation_audit['budgets']}. "
        rf"The weighted optimal profile falls from {allocation_audit['equal_objective']:.6f} "
        rf"for equal three-child allocation to {allocation_audit['objective']:.6f}, a "
        rf"{allocation_reduction:.3f}\% reduction; the two methods agree to "
        rf"{abs(allocation_audit['objective']-allocation_audit['brute']):.2e}. "
        r'Allocation is conditional on the fixed parent tree; it is not represented as '
        r'joint global optimization over all preceding partitions.')

    add(r'\subsection{Backward reduction and larger construction budgets}')
    add(r'Backward reduction starts with all 384 midpoint-quantile scenarios and deletes '
        r'the scenario causing the smallest increase in empirical $W_1$, followed by '
        r'nearest-representative probability reassignment \cite{heitschromisch2009}. '
        r'Forward selection uses the same initial grid. Across the four previous densities '
        r'and $B=4,8,16,32$, their mean distances and construction times are:')
    add(r'\begin{table}[htbp]')
    add(r'\centering\small')
    add(r'\caption{Forward and backward distance-based scenario-reduction baselines.}')
    add(r'\label{tab:reductionsummary}')
    add(r'\begin{tabular}{lrrr}')
    add(r'\toprule Method & mean $W_1$ & best-of-four count & total time (s)\\')
    add(r'\midrule')
    for method in ['Forward W1', 'Backward W1']:
        selected = [r for r in reduction_rows if r['method'] == method]
        wins = 0
        for density in sorted(set(r['density'] for r in reduction_rows)):
            for b in [4, 8, 16, 32]:
                f = next(r for r in selected if r['density'] == density and r['B'] == b)
                other = next(r for r in reduction_rows if r['method'] != method
                             and r['density'] == density and r['B'] == b)
                wins += int(f['W1'] < other['W1'] - 1e-12)
        add(fr"{method.replace('W1', '$W_1$')} & {np.mean([r['W1'] for r in selected]):.6f} & "
            rf"{wins}/16 & {sum(r['seconds'] for r in selected):.3f}\\")
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')
    add(r'Neither reduction rule dominates the other on every density--budget pair. '
        r'The exact minimax-profile constructor was also run through $B=128$ on the '
        r'sinusoidal and bimodal densities. Its profile remains no larger than the '
        r'equal-$\sqrt f$ construction at every budget. Figure~\ref{fig:markovscaling}(b) '
        r'reports actual timings; these measurements do not establish an asymptotic '
        r'runtime bound and are distinct from tree-solution cost.')

    add(r'\FloatBarrier')
    add(r'\subsection{Real-data-calibrated chronological policy evaluation}')
    add(r'The final experiment uses the 203 quarterly U.S. macroeconomic observations in '
        r'the documented \texttt{statsmodels.datasets.macrodata} series '
        r'\cite{statsmodelsmacrodata}. We extract real personal consumption expenditures '
        r'(\texttt{realcons}), take $100\Delta\log(\mathrm{realcons})$, and retain the '
        r'chronological ordering. The first 140 growth observations, through 1994Q1, are '
        r'used for centering, scaling, and AR(1) estimation; all 62 later observations are '
        r'held out. The packaged CSV preserves the source values and the script checks '
        r'the endpoints and sample size. This is a real-data-calibrated common demand '
        r'factor, while item loadings, capacities, and costs remain controlled parameters; '
        r'it is not presented as an industrial deployment.')
    fit = real_data['fit']
    add(fr"The scaled demand factor is clipped to $[0.05,9.95]$. Its fitted conditional "
        rf"mean is ${fit['alpha']:.6f}+{fit['rho']:.6f}\xi_t$, with residual standard "
        rf"deviation ${fit['sigma']:.6f}$; conditional distributions are truncated normal. "
        r'We evaluate 20 four-stage origins from 1994Q2 through 2009Q1 without refitting. '
        r'At each origin, the previous observed factor determines the root conditional '
        r'distribution. Every method supplies a complete nonanticipative policy, not merely '
        r'an initial decision. For an observed path, the policy is lifted by selecting the '
        r'nearest current child node, charging actual holding, shortage, and ordering costs, '
        r'and continuing along that node. Twenty origins are a limited sample, so we report '
        r'paired differences and standard errors rather than claiming broad superiority.')
    add(r'\begin{figure}[htbp]')
    add(r'\centering')
    add(r'\includegraphics[width=\textwidth]{figures/real_data_policy_comparison.pdf}')
    add(r'\caption{Real-consumption data and held-out policy costs. Error bars are standard errors across 20 chronological origins; sampling methods use two averaged replications per origin.}')
    add(r'\label{fig:realpolicy}')
    add(r'\end{figure}')
    add(r'\begin{table}[htbp]')
    add(r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}')
    add(r'\caption{Held-out real-data-calibrated inventory results. Paired differences are method cost minus Markov exact cost at the same budget; positive values favor Markov exact.}')
    add(r'\label{tab:realsummary}')
    add(r'\begin{tabular}{lrrrrr}')
    add(r'\toprule Method & $B$ & mean cost & paired difference & paired SE & mean certificate\\')
    add(r'\midrule')
    for summary in real_summaries:
        label = summary['method'].replace('optimal', 'exact').replace('sqrt-f', 'sqrt-$f$').replace('W1', '$W_1$')
        add(fr"{label} & {summary['B']} & {summary['mean_cost']:.3f} & "
            rf"{summary['paired_difference']:.3f} & {summary['paired_se']:.3f} & "
            rf"{summary['mean_certificate']:.3f}\\")
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')
    best = min(real_summaries, key=lambda s: s['mean_cost'])
    exact3 = next(s for s in real_summaries if s['method'] == 'Markov optimal' and s['B'] == 3)
    sqrt3 = next(s for s in real_summaries if s['method'] == 'Equal sqrt-f' and s['B'] == 3)
    ind2 = next(s for s in real_summaries if s['method'] == 'Independent optimal' and s['B'] == 2)
    ind3 = next(s for s in real_summaries if s['method'] == 'Independent optimal' and s['B'] == 3)
    add(fr"The lowest mean cost is {best['mean_cost']:.3f}, attained by {best['method']} "
        rf"at $B={best['B']}$. At $B=3$, equal $\sqrt f$ cells cost {sqrt3['mean_cost']:.3f} "
        rf"versus {exact3['mean_cost']:.3f} for Markov exact, although Markov exact has the "
        rf"smaller mean barycentric certificate ({exact3['mean_certificate']:.3f} versus "
        rf"{sqrt3['mean_certificate']:.3f}). Markov exact has lower mean cost than "
        rf"independent exact by {ind2['paired_difference']:.3f} at $B=2$ and "
        rf"{ind3['paired_difference']:.3f} at $B=3$, but the paired standard errors are "
        rf"{ind2['paired_se']:.3f} and {ind3['paired_se']:.3f}, and other constructions "
        r'are better on the overall mean. These results do not support a blanket claim '
        r'that fitted dependence or certificate optimality improves every realized policy. '
        r'They instead separate distributional certificates, model fit, and decision performance. '
        r'The certificates bound fitted-model value approximation; they are not bounds on '
        r'paired held-out realized-cost differences.')
    add(r'As implementation audits, the exact weighted $W_1$ calculation agrees with a '
        rf"12,000-point empirical transport computation ({audits['W1_exact']:.8f} versus "
        rf"{audits['W1_empirical']:.8f}), and independent deterministic-equivalent LPs "
        r'reproduce the PL dynamic-programming values on synthetic and real-data trees. '
        r'Detailed origin-level costs and all numerical checks appear in Appendix~\ref{app:majorbenchmark}.')
    add(r'\FloatBarrier')
    (ROOT / 'sections' / 'major_numerics.tex').write_text('\n'.join(lines) + '\n')

    # Supplementary tables and audit values.
    lines = []
    add = lines.append
    add(r'\section{Extended Markov, scaling, and real-data results}\label{app:majorbenchmark}')
    add(r'\subsection{Conditional Markov certificate details}')
    add(r'\begin{table}[H]')
    add(r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}')
    add(r'\caption{Individual Markov-tree runs; MC/Sobol entries average their replications.}')
    add(r'\label{tab:markovdetails}')
    add(r'\begin{tabular}{rlrrrr}')
    add(r'\toprule $T$ & Method & allocated & gap & certificate & nodes\\')
    add(r'\midrule')
    for horizon in [3, 4]:
        for method in method_names:
            for allocated in ([False, True] if method == 'Finite-B optimal' and horizon == 3 else [False]):
                selected = [r for r in markov_rows if r['horizon'] == horizon
                            and r['method'] == method and r['allocated'] == allocated]
                add(fr"{horizon} & {method.replace('Finite-B optimal', 'Finite-$B$ optimal').replace('sqrt-f', 'sqrt-$f$').replace('W1', '$W_1$')} & "
                    rf"{'yes' if allocated else 'no'} & {np.mean([r['gap'] for r in selected]):.7f} & "
                    rf"{np.mean([r['certificate'] for r in selected]):.7f} & {max(r['nodes'] for r in selected)}\\")
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')

    add(r'\subsection{Exact-construction scaling details}')
    add(r'\begin{table}[H]')
    add(r'\centering\small')
    add(r'\caption{Exact profiles, equal-$\sqrt f$ ratios, and measured construction times.}')
    add(r'\label{tab:scalingdetails}')
    add(r'\begin{tabular}{lrrrr}')
    add(r'\toprule Density & $B$ & $\mathfrak G_B^*$ & $\mathfrak G_{\sqrt f}/\mathfrak G_B^*$ & time (s)\\')
    add(r'\midrule')
    for r in scaling_rows:
        add(fr"{r['density']} & {r['B']} & ${sciformat(r['profile'])}$ & "
            rf"{r['sqrt_profile']/r['profile']:.6f} & {r['seconds']:.3f}\\")
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')

    add(r'\subsection{Held-out origin-level results}')
    add(r'\begin{table}[H]')
    add(r'\centering\scriptsize\setlength{\tabcolsep}{3pt}')
    add(r'\caption{Realized costs by origin, averaged over sampling replications.}')
    add(r'\label{tab:realorigins}')
    real_methods = ['Markov optimal', 'Independent optimal', 'Equal sqrt-f',
                    r'Forward W1', 'Backward W1', 'Monte Carlo', 'Sobol']
    add(r'\begin{tabular}{ll' + 'r' * len(real_methods) + '}')
    add(r'\toprule Origin & $B$ & Markov & independent & equal $\sqrt f$ & forward & backward & MC & Sobol\\')
    add(r'\midrule')
    for origin in real_data['origins']:
        label = next(r['label'] for r in real_rows if r['origin'] == origin)
        for b in [2, 3]:
            values = []
            for method in real_methods:
                selected = [r['realized'] for r in real_rows if r['origin'] == origin
                            and r['method'] == method and r['B'] == b]
                values.append(np.mean(selected))
            add(fr"{label} & {b} & " + ' & '.join(f'{v:.2f}' for v in values) + r'\\')
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')

    add(r'\subsection{Independent implementation audits}')
    add(r'\begin{table}[H]')
    add(r'\centering\small')
    add(r'\caption{Independent transport and deterministic-equivalent checks.}')
    add(r'\label{tab:majoraudits}')
    add(r'\begin{tabular}{lr}')
    add(r'\toprule Quantity & Value\\')
    add(r'\midrule')
    for label, key in [('Exact weighted $W_1$', 'W1_exact'),
                       ('Dense empirical $W_1$', 'W1_empirical'),
                       ('Synthetic tree DP value', 'synthetic_DP'),
                       ('Synthetic deterministic-equivalent LP', 'synthetic_LP'),
                       ('Real-data tree DP value', 'real_DP'),
                       ('Real-data deterministic-equivalent LP', 'real_LP')]:
        add(fr'{label} & {audits[key]:.10f}\\')
    add(r'\bottomrule\end{tabular}')
    add(r'\end{table}')
    add(r'The transport comparison uses 12,000 midpoint-quantile source scenarios and '
        r'monotone quantile transport with the actual atom probabilities. The LP audit constructs '
        r'all inventory balances, holding and backlog variables, capacitated orders, and '
        r'the shared initial constraint directly, without using the PL recursion.')
    (ROOT / 'sections' / 'major_supplement.tex').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    use_cache = os.environ.get('MAJOR_USE_CACHE', '0') == '1'
    if use_cache and (ROOT / 'major_markov.pkl').exists():
        with (ROOT / 'major_markov.pkl').open('rb') as handle:
            markov_rows, allocation_audit = pickle.load(handle)
    else:
        markov_rows, allocation_audit = run_markov_certificate_study()
        with (ROOT / 'major_markov.pkl').open('wb') as handle:
            pickle.dump((markov_rows, allocation_audit), handle)
    if use_cache and (ROOT / 'major_scaling.pkl').exists():
        with (ROOT / 'major_scaling.pkl').open('rb') as handle:
            reduction_rows, scaling_rows = pickle.load(handle)
    else:
        reduction_rows, scaling_rows = run_reduction_and_scaling()
        with (ROOT / 'major_scaling.pkl').open('wb') as handle:
            pickle.dump((reduction_rows, scaling_rows), handle)
    if use_cache and (ROOT / 'major_real.pkl').exists():
        with (ROOT / 'major_real.pkl').open('rb') as handle:
            real_rows, real_summaries, real_data = pickle.load(handle)
    else:
        real_rows, real_summaries, real_data = run_real_data_study()
        with (ROOT / 'major_real.pkl').open('wb') as handle:
            pickle.dump((real_rows, real_summaries, real_data), handle)
    audits = independent_numerical_audits(real_data)
    generate_major_outputs(markov_rows, allocation_audit, reduction_rows, scaling_rows,
                           real_rows, real_summaries, real_data, audits)
    print('Completed all Markov, reduction, scaling, real-data, and independent LP audits',
          flush=True)
