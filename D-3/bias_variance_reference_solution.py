"""Reference solution for the Day 3 challenge: bias, variance and no-free-lunch on a toy dataset.
Run:  python bv_reference.py       (prints every number used in the faculty key; saves plots)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

SIGMA = 0.3
f = lambda x: np.sin(2 * np.pi * x)

def make_data(n, rng, fun=f, sigma=SIGMA):
    x = rng.uniform(0, 1, n)
    return x.reshape(-1, 1), fun(x) + rng.normal(0, sigma, n)

def poly(d):
    return make_pipeline(PolynomialFeatures(d), LinearRegression())

mse = lambda a, b: float(np.mean((a - b) ** 2))

# ---------- Part A: train vs test error against degree (averaged over 50 repeats for a stable curve)
rng = np.random.default_rng(42)
degs = list(range(1, 13))
Xte, yte = make_data(1000, np.random.default_rng(7))
tr, te = np.zeros(len(degs)), np.zeros(len(degs))
R = 50
for r in range(R):
    X, y = make_data(30, rng)
    for i, d in enumerate(degs):
        m = poly(d).fit(X, y)
        tr[i] += mse(y, m.predict(X)) / R
        te[i] += mse(yte, m.predict(Xte)) / R
print("PART A  degree: train MSE / test MSE  (noise floor = %.3f)" % SIGMA**2)
for d, a, b in zip(degs, tr, te):
    print(f"  deg {d:2d}: train {a:.3f}  test {b:.3f}")
print("  best test degree:", degs[int(np.argmin(te))])
# single-fit illustration (seed 42)
X, y = make_data(30, np.random.default_rng(42))
print("  single fit seed 42, deg 1/3/9 train:",
      [round(mse(y, poly(d).fit(X, y).predict(X)), 3) for d in (1, 3, 9)],
      "test:", [round(mse(yte, poly(d).fit(X, y).predict(Xte)), 3) for d in (1, 3, 9)])
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(degs, tr, "o-", label="train"); ax.plot(degs, te, "s-", label="test")
ax.axhline(SIGMA**2, ls=":", c="gray", label="noise floor"); ax.set_yscale("log")
ax.set_xlabel("polynomial degree"); ax.set_ylabel("MSE"); ax.legend(); fig.tight_layout(); fig.savefig("partA.png", dpi=150)

# ---------- Part B/C: bias^2 and variance by repeated sampling
grid = np.linspace(0.02, 0.98, 100).reshape(-1, 1)
truth = f(grid.ravel())
def bias_var(d, n, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    P = np.empty((reps, len(grid)))
    for k in range(reps):
        X, y = make_data(n, rng)
        P[k] = poly(d).fit(X, y).predict(grid)
    mean = P.mean(0)
    return float(np.mean((mean - truth) ** 2)), float(P.var(0).mean())
print("\nPART B/C  bias^2 and variance (200 training sets)")
res = {}
for n in (30, 300):
    for d in (1, 3, 9):
        b, v = bias_var(d, n)
        res[(n, d)] = (b, v)
        print(f"  n={n:3d} deg {d}: bias^2 {b:.4f}  variance {v:.4f}  bias^2+var {b+v:.4f}  (+noise {SIGMA**2:.2f} = expected test MSE {b+v+SIGMA**2:.3f})")

# ---------- Part D: no free lunch mini-demo
fun = {"D1 linear": lambda x: 2 * x, "D2 curved": f, "D3 random": lambda x: np.zeros_like(x)}
models = {"Linear reg": lambda: poly(1), "Degree-9 poly": lambda: poly(9),
          "kNN (k=5)": lambda: KNeighborsRegressor(5), "Mean": lambda: DummyRegressor()}
print("\nPART D  mean test MSE over 100 repeats (30 train points each)")
rng = np.random.default_rng(11)
table = {}
for dn, fn in fun.items():
    acc = {m: 0.0 for m in models}
    for r in range(100):
        X, y = make_data(30, rng, fun=fn)
        Xt, yt = make_data(500, rng, fun=fn)
        if dn == "D3 random":
            y = rng.normal(0, 1, 30); yt = rng.normal(0, 1, 500)   # pure noise unrelated to x
        for mn, mk in models.items():
            acc[mn] += mse(yt, mk().fit(X, y).predict(Xt)) / 100
    table[dn] = acc
    order = sorted(acc, key=acc.get)
    print(" ", dn, {k: round(v, 3) for k, v in acc.items()}, "-> ranks", order)