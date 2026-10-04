"""
market_making_egt.py

Author: Minh Nguyen

This script re-implements the numerical analysis in "Modeling Market Making Behavior
Using Evolutionary Game Theory," a Dartmouth Math 30.04 paper I co-authored with Parker
Tomlinson, Adeoluwawumi Adedoyin and Tuan Ha (also my writing sample). Market makers
choose tight spreads, wide spreads or exit against informed, momentum and passive
traders. The code builds both players' payoff matrices, evaluates the replicator
dynamics, draws slope fields across trader mixes, competition and volatility, and
computes Moran fixation probabilities for 500 firms, checking all 18 values the paper
reports. Reproducing the figures showed that the competition values stated in the text
do not produce the paper's Figure 3.3; the code flags this. The model comes from our
paper. I wrote this script with AI assistance: Claude drafted the outline and
docstrings, and Codex helped test it and run the numerical checks.

Usage:
    python market_making_egt.py
    Writes figures next to this script in figures/, prints tables, and runs checks.

Requires Python 3.10+, numpy, matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

__author__ = "Minh Nguyen"
FIG_DIR = Path(__file__).resolve().parent / "figures"

MM_STRATEGIES = ("tight", "wide", "exit")              # x_t, x_w, x_c


# ----------------------------------------------------------------------
# 1. Parameters (paper Section 2.1; default values from Section 3.1.1)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    """Model parameters from Section 3.1.1 of the paper."""
    s_T: float = 2.0     # spread revenue per interaction, tight
    s_W: float = 2.5     # spread revenue per interaction, wide
    lam_T: float = 1.0   # volume multiplier, tight spread
    lam_W: float = 0.4   # volume multiplier, wide spread
    a: float = 5.0       # adverse-selection edge of informed traders
    v: float = 2.0       # market volatility
    beta: float = 0.45   # momentum edge per unit of volatility
    c_f: float = 3.5     # information cost, informed traders
    c_m: float = 0.5     # active-management cost, momentum traders


DEFAULT = Params()

# Fixed trader populations (y_i, y_m, y_p) used in Sections 3.1, 3.2, 3.3, 3.4, 3.6.
TRADER_MIXES = {
    "equal": (1 / 3, 1 / 3, 1 / 3),
    "informed": (2 / 3, 1 / 6, 1 / 6),
    "momentum": (1 / 6, 2 / 3, 1 / 6),
    "passive": (1 / 6, 1 / 6, 2 / 3),
}


# ----------------------------------------------------------------------
# 2. Payoffs and fitness (paper Sections 2.1-2.2, equations 1-7)
# ----------------------------------------------------------------------
def payoff_matrices(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_mm, A_tr), each 3x3.

    Rows = market-maker strategies (tight, wide, exit);
    columns = trader types (informed, momentum, passive).
    A_mm[r, c] is the market maker's payoff, A_tr[r, c] the trader's.

    Trader information/management costs are paid even when market makers exit.
    """
    A_mm = np.array([
        [p.lam_T * (p.s_T - p.a), p.lam_T * (p.s_T - p.beta * p.v), p.lam_T * p.s_T],
        [p.lam_W * (p.s_W - p.a), p.lam_W * (p.s_W - p.beta * p.v), p.lam_W * p.s_W],
        [0.0, 0.0, 0.0],
    ])
    A_tr = -A_mm - np.array([p.c_f, p.c_m, 0.0])
    return A_mm, A_tr


def mm_fitness(p: Params, y: tuple[float, float, float]) -> np.ndarray:
    """Expected payoff of each market-maker strategy against trader mix y.

    Returns np.array([f_t, f_w, f_c]) (equations 1-5).
    The trader mix enters fitness only through the edge e = y_i*a + y_m*beta*v;
    with y fixed, fitness is independent of x, so a unique best strategy present
    initially attracts the dynamics to its vertex and is also the finite-population
    ESS for positive selection intensity.
    """
    return payoff_matrices(p)[0] @ np.asarray(y, dtype=float)


# ----------------------------------------------------------------------
# 3. Replicator dynamics (paper Section 2.2, equations 8-10)
# ----------------------------------------------------------------------
def replicator_rhs(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Time derivative of x = (x_t, x_w, x_c) on the simplex.

    x_dot_k = x_k * (f_k - phi), where phi = x . f (average fitness, eq. 6-7).
    The result sums to zero for any x on the simplex, up to rounding.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    return x * (f - x @ f)


def replicator_rhs_closed_form(
    x: np.ndarray, y: tuple[float, float, float]
) -> np.ndarray:
    """Equations 11-13 typed in directly (default parameters only).

    Used only as an independent check of replicator_rhs() in Section 6.
    """
    x_t, x_w, x_c = x
    y_i, y_m, _ = y
    return np.array([
        x_t * (y_i * (5 * x_t + 2 * x_w - 5)
               + y_m * (0.9 * x_t + 0.36 * x_w - 0.9) - 2 * x_t - x_w + 2),
        x_w * (y_i * (5 * x_t + 2 * x_w - 2)
               + y_m * (0.9 * x_t + 0.36 * x_w - 0.36) - 2 * x_t - x_w + 1),
        x_c * (y_i * (5 * x_t + 2 * x_w)
               + y_m * (0.9 * x_t + 0.36 * x_w) - 2 * x_t - x_w),
    ])


# ----------------------------------------------------------------------
# 4. Slope fields (paper Figures 3.1-3.4)
# ----------------------------------------------------------------------
def simplex_grid(n: int) -> np.ndarray:
    """Points (x_t, x_w, x_c) with x_t + x_w + x_c = 1, all >= 0, on an n-step grid.

    Returns (n + 1)(n + 2)/2 points, including edges and all three vertices.
    """
    return np.array([(i / n, j / n, (n - i - j) / n)
                     for i in range(n + 1) for j in range(n + 1 - i)])


def plot_slope_field(
    ax, p: Params, y: tuple[float, float, float], title: str, n: int = 20
) -> None:
    """Draw the replicator slope field on the (x_t, x_w) plane (x_c = 1 - x_t - x_w).

    Arrows show direction with equal lengths; stationary points have no arrow.
    A unique highest-fitness strategy is the ESS for this fixed trader mix.
    """
    points = simplex_grid(n)
    f = mm_fitness(p, y)
    velocity = np.array([replicator_rhs(x, f)[:2] for x in points])
    speed = np.linalg.norm(velocity, axis=1)
    direction = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity),
                          where=speed[:, None] > 1e-12)
    ax.quiver(points[:, 0], points[:, 1], direction[:, 0], direction[:, 1],
              angles="xy", scale_units="xy", scale=30, pivot="mid",
              color="#3569a8", width=0.004, minlength=0)
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black", linewidth=1)
    vertices = np.array([[1, 0], [0, 1], [0, 0]])
    # Roundoff at a payoff tie must not create a spurious unique ESS.
    best = np.flatnonzero(np.isclose(f, f.max(), rtol=1e-12, atol=1e-14))
    ax.scatter(vertices[best, 0], vertices[best, 1], marker="*", s=180,
               color="#b52332", zorder=3)
    names = ", ".join(MM_STRATEGIES[k] for k in best)
    label = f"ESS: {names}" if len(best) == 1 else f"Tied best payoffs: {names}"
    ax.set(title=f"{title}\n{label}", xlabel=r"Tight share $x_t$",
           ylabel=r"Wide share $x_w$", xlim=(-0.06, 1.06), ylim=(-0.06, 1.06))
    ax.set_aspect("equal")


def figure_default(p: Params = DEFAULT) -> plt.Figure:
    """Figure 3.1: default parameters, equal trader mix."""
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    plot_slope_field(ax, p, TRADER_MIXES["equal"], "Equal trader mix")
    fig.suptitle("Figure 3.1: Default example")
    return fig


def figure_trader_mix(p: Params = DEFAULT) -> plt.Figure:
    """Figure 3.2: informed-, momentum-, and passive-biased mixes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, name in zip(axes, ("informed", "momentum", "passive")):
        plot_slope_field(ax, p, TRADER_MIXES[name], f"{name.capitalize()}-biased mix")
    fig.suptitle("Figure 3.2: Effect of trading population composition")
    return fig


def figure_competition(
    p: Params = DEFAULT, lam_W_values=(0.05, 0.1, 0.4, 0.8)
) -> plt.Figure:
    """Figure 3.3: vary lam_W (competition), equal mix.

    The text states lam_W = 0.1, 0.4, 0.8 (competition ratios 0.2, 0.8, 1.6).
    These values all favor wide spreads, whereas the paper's "More Competitive"
    panel converges to tight; the text's values and the figure differ.
    With DEFAULT and equal traders, tight wins only when lam_W < 0.0625.
    The added lam_W = 0.05 panel illustrates that result, with ratio 0.1.
    """
    if len(lam_W_values) == 0 or p.s_W == p.s_T:
        raise ValueError("Provide competition values and distinct tight/wide spreads")
    fig, axes = plt.subplots(1, len(lam_W_values), figsize=(5 * len(lam_W_values), 5),
                             squeeze=False, constrained_layout=True)
    for ax, value in zip(axes.flat, lam_W_values):
        ratio = value / (p.s_W - p.s_T)
        label = (
            "Stated text value" if value in (0.1, 0.4, 0.8) else "Illustrative value"
        )
        if value == 0.05:
            label = "Illustrative: below default threshold"
        title = (
            label + "\n"
            + rf"$\lambda_W = {value:g},\ \lambda_W/(s_W-s_T) = {ratio:g}$"
        )
        plot_slope_field(ax, replace(p, lam_W=value), TRADER_MIXES["equal"], title)
    fig.suptitle("Figure 3.3: Effect of competition (stated and illustrative values)")
    return fig


def figure_volatility(p: Params = DEFAULT, v_values=(0.5, 2.0, 5.0, 8.0)) -> plt.Figure:
    """Figure 3.4: vary v, momentum-biased mix.

    The paper does not list its plotted v values; here they are 0.5, 2, 5, 8.
    Under DEFAULT and this mix, tight wins below v = 25/9 (about 2.78),
    wide wins between 25/9 and 50/9 (about 5.56), and exit wins above 50/9.
    Adjacent strategies tie at the thresholds; the chosen values span the regimes.
    """
    if len(v_values) == 0:
        raise ValueError("Provide at least one volatility value")
    columns = min(2, len(v_values))
    rows = (len(v_values) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 5 * rows),
                             squeeze=False, constrained_layout=True)
    panel_names = {0.5: "Less Volatile", 2.0: "Default Example",
                   5.0: "More Volatile", 8.0: "Extremely Volatile"}
    for ax, value in zip(axes.flat, v_values):
        title = f"{panel_names.get(value, 'Volatility scenario')}\nv = {value:g}"
        plot_slope_field(ax, replace(p, v=value), TRADER_MIXES["momentum"], title)
    for ax in list(axes.flat)[len(v_values):]:
        ax.set_axis_off()
    fig.suptitle("Figure 3.4: Effect of volatility (chosen v values shown)")
    return fig


# ----------------------------------------------------------------------
# 5. Finite population: Moran process (paper Sections 3.5-3.6)
# ----------------------------------------------------------------------
N_DEFAULT = 500
W_DEFAULT = 0.06


def reproductive_fitness(f: np.ndarray, w: float = W_DEFAULT) -> np.ndarray:
    """F_j = 1 + w * f_j; Moran reproduction requires positive fitness."""
    if not np.isfinite(w) or w < 0:
        raise ValueError("Selection intensity must be finite and nonnegative")
    F = 1 + w * np.asarray(f, dtype=float)
    if not np.all(np.isfinite(F) & (F > 0)):
        raise ValueError("Reproductive fitness must be finite and strictly positive")
    return F


def fixation_probability(F_A: float, F_B: float, N: int = N_DEFAULT) -> float:
    """Probability that one A mutant takes over a resident B population.

    rho_{A/B} = (1 - 1/r) / (1 - r^(-N)),  r = F_A / F_B.

    Neutral fitness gives 1/N. expm1 avoids cancellation near r = 1.
    For r < 1, rewrite as r^(N-1) * (1-r)/(1-r^N) to avoid overflow.
    Probabilities below floating-point range may underflow to zero.
    """
    if N < 1:
        raise ValueError("N must be positive")
    if not (np.isfinite(F_A) and np.isfinite(F_B) and F_A > 0 and F_B > 0):
        raise ValueError("Reproductive fitness must be finite and strictly positive")
    if F_A == F_B:
        return 1 / N
    # log1p preserves tiny differences; separate logs avoid ratio overflow.
    delta = F_A - F_B
    log_r = (np.log1p(delta / F_B) if abs(delta) <= 0.5 * F_B
             else np.log(F_A) - np.log(F_B))
    if log_r > 0:
        return float(np.expm1(-log_r) / np.expm1(-N * log_r))
    return float(np.exp((N - 1) * log_r) * np.expm1(log_r) / np.expm1(N * log_r))


def fixation_table(
    p: Params, y: tuple[float, float, float], N: int = N_DEFAULT, w: float = W_DEFAULT
) -> dict[tuple[str, str], float]:
    """rho_{A/B} for all six ordered pairs of distinct strategies.

    Keys are (mutant A, resident B) using the short names "t", "w", "c".
    """
    F = reproductive_fitness(mm_fitness(p, y), w)
    names = ("t", "w", "c")
    return {(A, B): fixation_probability(F[i], F[j], N)
            for i, A in enumerate(names) for j, B in enumerate(names) if i != j}


def finite_population_ess(
    table: dict[tuple[str, str], float], N: int = N_DEFAULT
) -> list[str]:
    """Strategies that no single mutant can invade (rho_{B/A} < 1/N for every B != A).

    Returns short names; neutral ties do not satisfy this strict condition.
    """
    names = ("t", "w", "c")
    return [A for A in names if all(table[B, A] < 1 / N for B in names if B != A)]


def simulate_fixation(
    F_A: float, F_B: float, N: int, trials: int, rng: np.random.Generator
) -> float:
    """Monte Carlo estimate of rho_{A/B} for the Moran birth-death process.

    Start with 1 mutant; each step pick a reproducer proportional to fitness and a
    uniformly random individual to die; stop at 0 or N mutants.
    """
    # Common rescaling leaves reproduction probabilities unchanged.
    a, b = F_A / max(F_A, F_B), F_B / max(F_A, F_B)
    fixations = 0
    for _ in range(trials):
        mutants = 1
        while 0 < mutants < N:
            birth_probability = mutants * a / (mutants * a + (N - mutants) * b)
            birth_is_mutant = rng.random() < birth_probability
            death_is_mutant = rng.random() < mutants / N
            mutants += int(birth_is_mutant) - int(death_is_mutant)
        fixations += mutants == N
    return fixations / trials


# ----------------------------------------------------------------------
# 6. Checks against the paper (all 18 values in Section 3.6)
# ----------------------------------------------------------------------
PAPER_FIXATION = {
    "informed": {("t", "w"): 6.33e-17, ("w", "c"): 1.57e-7, ("t", "c"): 5.61e-22,
                 ("c", "t"): 0.0890, ("w", "t"): 0.0670, ("c", "w"): 0.0236},
    "momentum": {("t", "w"): 0.00826, ("t", "c"): 0.0329, ("w", "t"): 1.41e-4,
                 ("w", "c"): 0.0250, ("c", "t"): 1.87e-9, ("c", "w"): 8.30e-8},
    "passive": {("t", "w"): 0.0232, ("t", "c"): 0.0575, ("w", "t"): 1.91e-7,
                ("w", "c"): 0.0351, ("c", "t"): 8.47e-15, ("c", "w"): 6.27e-10},
}


def run_checks() -> None:
    """Check paper results, numerical stability, and seeded Moran simulations."""
    A_mm, A_tr = payoff_matrices(DEFAULT)
    np.testing.assert_allclose(A_mm, [[-3, 1.1, 2], [-1, 0.64, 1], [0, 0, 0]])
    np.testing.assert_allclose(A_tr, [[-0.5, -1.6, -2], [-2.5, -1.14, -1],
                                    [-3.5, -0.5, 0]])
    print("PASS: both payoff matrices match the paper")

    rng = np.random.default_rng(30)
    points = np.vstack([simplex_grid(4), rng.dirichlet(np.ones(3), size=20)])
    for y in TRADER_MIXES.values():
        for x in points:
            rhs = replicator_rhs(x, mm_fitness(DEFAULT, y))
            np.testing.assert_allclose(
                rhs, replicator_rhs_closed_form(x, y), atol=1e-14
            )
            np.testing.assert_allclose(rhs.sum(), 0, atol=1e-14)
    print("PASS: replicator dynamics agree with equations 11-13")
    print("PASS: replicator dynamics conserve total population share")

    np.testing.assert_equal(fixation_probability(1.0, 1.0), 1 / N_DEFAULT)
    print("PASS: neutral fixation is 1/500 = 0.002")

    for name, reported in PAPER_FIXATION.items():
        table = fixation_table(DEFAULT, TRADER_MIXES[name])
        for pair, expected in reported.items():
            # atol=0 is essential: an absolute tolerance would accept zero for tiny rho.
            np.testing.assert_allclose(table[pair], expected, rtol=0.01, atol=0,
                                       err_msg=f"{name}: {pair}")
    print("PASS: all 18 paper fixation probabilities agree within 1%")

    expected_ess = (("informed", ["c"]), ("momentum", ["t"]), ("passive", ["t"]))
    for name, expected in expected_ess:
        actual = finite_population_ess(fixation_table(DEFAULT, TRADER_MIXES[name]))
        np.testing.assert_equal(actual, expected)
    print("PASS: finite-population ESS matches all three paper cases")

    np.testing.assert_allclose(fixation_probability(0.5, 1.0), 2.0 ** -500,
                               rtol=1e-12, atol=0)
    np.testing.assert_allclose(fixation_probability(np.nextafter(1.0, 2.0), 1.0),
                               1 / N_DEFAULT, rtol=1e-12, atol=0)
    print("PASS: tiny probabilities and near-neutral fitness are stable")

    # Compare tight mutants with wide residents in the momentum-biased market.
    N, w, trials = 20, 0.5, 5000
    F = reproductive_fitness(mm_fitness(DEFAULT, TRADER_MIXES["momentum"]), w)
    expected = fixation_probability(F[0], F[1], N)  # tight mutant, wide resident
    estimate = simulate_fixation(F[0], F[1], N, trials, np.random.default_rng(30))
    standard_error = np.sqrt(expected * (1 - expected) / trials)
    assert abs(estimate - expected) <= 4 * standard_error, "Moran simulation disagrees"
    print(
        f"PASS: seeded Moran simulation (N={N}, w={w}) agrees within 4 standard errors"
    )


# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
def main() -> None:
    """Run checks, save four figures, and print fixation tables and finite ESS."""
    run_checks()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, make_figure in (("figure_3_1_default", figure_default),
                              ("figure_3_2_trader_mix", figure_trader_mix),
                              ("figure_3_3_competition", figure_competition),
                              ("figure_3_4_volatility", figure_volatility)):
        fig = make_figure()
        path = FIG_DIR / f"{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path.name} to {FIG_DIR.name}/")
    print("Note: Figure 3.3's text values favor wide spreads, while the paper's "
          '"More Competitive" panel converges to tight. The added lambda_W=0.05 '
          "panel illustrates tight convergence below the default threshold 0.0625.")

    print(
        f"\nMoran process: N={N_DEFAULT}, w={W_DEFAULT}, neutral rho={1 / N_DEFAULT:g}"
    )
    for name, y in TRADER_MIXES.items():
        table = fixation_table(DEFAULT, y)
        mix = ", ".join(f"{share:.3f}" for share in y)
        print(f"\n{name.capitalize()} trader mix: ({mix})")
        print("Mutant/resident        rho       Above neutral?")
        for (A, B), rho in table.items():
            print(f"{A}/{B:1}             {rho:12.6e}    {rho > 1 / N_DEFAULT}")
        print(f"Finite-population ESS: {finite_population_ess(table)}")


if __name__ == "__main__":
    main()
