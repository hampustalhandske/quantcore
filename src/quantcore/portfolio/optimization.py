"""Portfolio weight optimization: mean-variance, risk parity, and Kelly sizing.

Mean-variance (Markowitz 1952):
    min_w  w^T Sigma w  s.t.  w^T mu = mu_target, sum(w) = 1, w >= 0
    or for unconstrained tangency:
    w_tangency = Sigma^{-1}(mu - r_f) / 1^T Sigma^{-1}(mu - r_f)

Fully-invested quadratic programs (sum(w) = 1, w >= 0 unless shorting is
allowed):
    minimum variance   min_w  w^T Sigma w
    mean-variance      max_w  w^T mu - 0.5*lambda*w^T Sigma w
    with turnover cost max_w  w^T mu - 0.5*lambda*w^T Sigma w
                              - c*||w - w_prev||_1
    All three are convex quadratic programs; the third becomes one once the
    trade w - w_prev is written as purchases minus sales, both non-negative,
    which makes the L1 cost linear (Lobo, Fazel & Boyd 2007). They are
    solved exactly by one primal active-set method (Nocedal & Wright 2006,
    Algorithm 16.3): constraints believed to bind are held as equalities,
    the resulting equality-constrained problem is solved in closed form
    (from a Cholesky factor of the free assets' block, or in the null space
    of those constraints where that block is singular), and the set is
    revised one constraint at a time until the Karush-Kuhn-Tucker
    conditions hold. The covariance must be positive semi-definite. The
    solution does not depend on the units of mu and Sigma: multiplying mu,
    Sigma and c by a common positive factor leaves every decision of the
    iteration unchanged.

Risk parity / Equal Risk Contribution (Maillard et al. 2010):
    RC_i = w_i * (Sigma*w)_i / (w^T Sigma w)  (risk contribution of asset i)
    Target: all RC_i equal to 1/k. Solved via Spinu's (2013) strictly convex
    reformulation
        min_{w > 0}  0.5 * w^T Sigma w - sum_i b_i * ln(w_i),   b_i = 1/k
    whose first-order condition w_i * (Sigma*w)_i = b_i is exactly the
    equal-risk-contribution condition once w is rescaled to sum to 1. The
    Hessian Sigma + diag(b_i / w_i^2) is positive definite for any PSD
    Sigma, so the minimiser is unique and a damped Newton iteration
    converges from any interior starting point.

Kelly fraction (Kelly 1956, single-asset):
    f* = (mu - r_f) / sigma^2

References:
    Markowitz, H. (1952), "Portfolio Selection." Kelly, J.L. (1956), "A New
    Interpretation of Information Rate." Maillard, Roncalli & Teïletche
    (2010), "The Properties of Equally Weighted Risk Contribution
    Portfolios." Spinu, F. (2013), "An Algorithm for Computing Risk Parity
    Weights." Lobo, Fazel & Boyd (2007), "Portfolio Optimization with
    Linear and Fixed Transaction Costs." Nocedal, J. & Wright, S.J. (2006),
    "Numerical Optimization," 2nd ed., ch. 16. Bland, R.G. (1977), "New
    Finite Pivoting Rules for the Simplex Method." See docs/REFERENCES.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def _validate_cov_matrix(cov_matrix: npt.NDArray[np.float64]) -> None:
    if cov_matrix.ndim != 2 or cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("cov_matrix must be a square 2D array")
    if not np.all(np.isfinite(cov_matrix)):
        raise ValueError("cov_matrix must contain only finite values")


def _validate_expected_returns(
    expected_returns: npt.NDArray[np.float64], cov_matrix: npt.NDArray[np.float64]
) -> None:
    if expected_returns.ndim != 1 or expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError("expected_returns must be a 1D array matching cov_matrix's dimension")
    if not np.all(np.isfinite(expected_returns)):
        raise ValueError("expected_returns must contain only finite values")


def _validate_risk_aversion_is_finite(risk_aversion: float) -> None:
    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")


def _validate_cov_matrix_is_positive_semidefinite(cov_matrix: npt.NDArray[np.float64]) -> None:
    """Raise `ValueError` unless the symmetric part of `cov_matrix` is positive semi-definite.

    `w'Aw` depends only on the symmetric part of `A`, so that is the part
    tested. An eigenvalue counts as negative when it is below `-_QP_RTOL`
    times the largest absolute row sum of the matrix, a bound on its largest
    eigenvalue. That margin is some thousands of times the round-off that
    gives the zero eigenvalues of a rank-deficient covariance either sign,
    so such a covariance is accepted.
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    symmetric = 0.5 * cov + 0.5 * cov.T
    floor = _QP_RTOL * _largest_row_sum(symmetric)
    # A matrix whose eigenvalues all exceed -floor has a Cholesky factor once
    # floor is added to its diagonal. The factorisation costs a fraction of
    # an eigenvalue computation, which is left for the matrices it rejects:
    # those may be singular rather than indefinite.
    shifted = symmetric.copy()
    np.fill_diagonal(shifted, np.diag(symmetric) + floor)
    try:
        np.linalg.cholesky(shifted)
    except np.linalg.LinAlgError:
        smallest = float(np.linalg.eigvalsh(symmetric)[0])
        if smallest < -floor:
            raise ValueError(
                f"cov_matrix must be positive semi-definite (its smallest eigenvalue is "
                f"{smallest:.6g})"
            ) from None


_QP_MAX_ITER_PER_CONSTRAINT = 100
_QP_RTOL = 1e-12
_QP_PIVOT_RTOL = 64.0 * float(np.finfo(np.float64).eps)
_QP_CHOLESKY_RTOL = 1e-6


def _normalised_rows(
    matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Divide each constraint row and its right-hand side by the row's largest coefficient.

    The constraint set is unchanged, and the multiplier of every row is then
    in the units of the objective's gradient, so one tolerance serves rows
    and bounds alike. An all-zero row is left as it is.
    """
    scale = np.max(np.abs(matrix), axis=1, initial=0.0)
    scale[scale == 0.0] = 1.0
    return matrix / scale[:, None], rhs / scale


def _solve_convex_qp(
    hessian: npt.NDArray[np.float64],
    linear: npt.NDArray[np.float64],
    eq_matrix: npt.NDArray[np.float64],
    eq_rhs: npt.NDArray[np.float64],
    ineq_matrix: npt.NDArray[np.float64],
    ineq_rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    func_name: str,
) -> npt.NDArray[np.float64]:
    """Exact minimiser of a convex quadratic program by a primal active-set method.

    Minimises `0.5 * x'Hx + linear'x` subject to `eq_matrix @ x = eq_rhs`,
    `ineq_matrix @ x >= ineq_rhs` and `lower <= x <= upper`, for a positive
    semi-definite `H` (Nocedal & Wright 2006, Algorithm 16.3).

    If `start`, moved inside the bounds, satisfies every constraint to
    round-off, the iteration begins there. Otherwise a feasible point is
    found first by minimising the total violation of the equality and
    inequality constraints, a linear program in `x` and one non-negative
    slack per violated constraint that the same iteration solves; a positive
    minimum means the constraints cannot all hold.

    Args:
        hessian: Symmetric positive semi-definite matrix `H`, shape (n, n).
        linear: Linear coefficients of the objective, shape (n,).
        eq_matrix: Equality-constraint rows, shape (m_eq, n); may have no rows.
        eq_rhs: Equality right-hand sides, shape (m_eq,).
        ineq_matrix: Inequality-constraint rows, shape (m_in, n); may have no
            rows.
        ineq_rhs: Inequality right-hand sides, shape (m_in,).
        lower: Lower bounds, shape (n,); `-inf` for none.
        upper: Upper bounds, shape (n,); `inf` for none.
        start: Point the iteration starts from, shape (n,). It need not be
            feasible. It decides which minimiser is returned when there are
            several, and a variable that starts on a bound and is never
            moved stays exactly there.
        func_name: Public function name used in error messages.

    Returns:
        A minimiser, shape (n,). Every bound holds exactly, and a variable
        stopped by a bound equals that bound exactly.

    Raises:
        RuntimeError: If the constraints are infeasible; if the objective is
            unbounded below on the feasible set; if `H` has negative
            curvature along a feasible direction (the problem is not
            convex); if `H`, `linear` or the solution is not finite; or if
            the iteration cap is reached. The message says which.
    """
    if not (np.all(np.isfinite(hessian)) and np.all(np.isfinite(linear))):
        raise RuntimeError(
            f"{func_name}: the objective is not finite (its coefficients overflow double precision)"
        )
    if np.any(lower > upper):
        raise RuntimeError(
            f"{func_name}: the constraints are infeasible (a lower bound exceeds its upper bound)"
        )
    eq_matrix, eq_rhs = _normalised_rows(eq_matrix, eq_rhs)
    ineq_matrix, ineq_rhs = _normalised_rows(ineq_matrix, ineq_rhs)
    x = np.clip(np.asarray(start, dtype=np.float64), lower, upper)

    eq_violated, ineq_violated = _violated_rows(eq_matrix, eq_rhs, ineq_matrix, ineq_rhs, x)
    if np.any(eq_violated) or np.any(ineq_violated):
        n = x.shape[0]
        eq_gap = (eq_rhs - eq_matrix @ x)[eq_violated]
        ineq_gap = (ineq_rhs - ineq_matrix @ x)[ineq_violated]
        n_slack = eq_gap.shape[0] + ineq_gap.shape[0]
        # One slack per violated row, signed so that it starts at the size of
        # the violation and the row holds exactly once the slack is zero.
        eq_slack = np.zeros((eq_matrix.shape[0], n_slack))
        eq_slack[np.flatnonzero(eq_violated), np.arange(eq_gap.shape[0])] = np.sign(eq_gap)
        ineq_slack = np.zeros((ineq_matrix.shape[0], n_slack))
        ineq_slack[
            np.flatnonzero(ineq_violated), eq_gap.shape[0] + np.arange(ineq_gap.shape[0])
        ] = 1.0
        relaxed = _active_set_iterations(
            hessian=np.zeros((n + n_slack, n + n_slack)),
            linear=np.concatenate([np.zeros(n), np.ones(n_slack)]),
            eq_matrix=np.hstack([eq_matrix, eq_slack]),
            ineq_matrix=np.hstack([ineq_matrix, ineq_slack]),
            ineq_rhs=ineq_rhs,
            lower=np.concatenate([lower, np.zeros(n_slack)]),
            upper=np.concatenate([upper, np.full(n_slack, np.inf)]),
            x=np.concatenate([x, np.abs(eq_gap), ineq_gap]),
            func_name=func_name,
        )
        x = relaxed[:n]
        eq_violated, ineq_violated = _violated_rows(eq_matrix, eq_rhs, ineq_matrix, ineq_rhs, x)
        if np.any(eq_violated) or np.any(ineq_violated):
            raise RuntimeError(
                f"{func_name}: the constraints are infeasible (no point satisfies all of them)"
            )

    return _active_set_iterations(
        hessian, linear, eq_matrix, ineq_matrix, ineq_rhs, lower, upper, x, func_name
    )


def _violated_rows(
    eq_matrix: npt.NDArray[np.float64],
    eq_rhs: npt.NDArray[np.float64],
    ineq_matrix: npt.NDArray[np.float64],
    ineq_rhs: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Which equality and inequality rows `x` violates by more than round-off.

    A row counts as satisfied when its residual is within `_QP_RTOL` of the
    size of the terms it is summed from.
    """
    abs_x = np.abs(x)
    eq_violated = np.abs(eq_matrix @ x - eq_rhs) > _QP_RTOL * (
        np.abs(eq_matrix) @ abs_x + np.abs(eq_rhs)
    )
    ineq_violated = ineq_rhs - ineq_matrix @ x > _QP_RTOL * (
        np.abs(ineq_matrix) @ abs_x + np.abs(ineq_rhs)
    )
    return eq_violated, ineq_violated


def _largest_row_sum(matrix: npt.NDArray[np.float64]) -> float:
    """Largest absolute row sum of `matrix`, an upper bound on the size of its eigenvalues."""
    return float(np.max(np.sum(np.abs(matrix), axis=1), initial=0.0))


def _min_norm_least_squares(
    matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Least-squares solution of `matrix @ y = rhs`, the shortest one if there are several."""
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return np.zeros(matrix.shape[1])
    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        # One equation or one unknown, the case of a single budget row: the
        # solution is a projection onto one vector and needs no factorisation.
        # The vector is scaled to a largest entry of 1 first, so that its
        # squared length cannot overflow or underflow.
        vector = matrix.ravel()
        size = float(np.max(np.abs(vector)))
        if size == 0.0:
            return np.zeros(matrix.shape[1])
        direction = vector / size
        squared_length = float(direction @ direction)
        if matrix.shape[1] == 1:
            return np.array([float(direction @ rhs) / squared_length / size])
        return direction * (float(rhs[0]) / size / squared_length)
    solution: npt.NDArray[np.float64] = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    return solution


def _cholesky_if_definite(
    matrix: npt.NDArray[np.float64], pivot_floor: float
) -> npt.NDArray[np.float64] | None:
    """Lower Cholesky factor of `matrix`, or None if it is not safely positive definite.

    "Safely" means every squared pivot exceeds `pivot_floor`; a matrix that
    fails is singular, indefinite or too ill-conditioned for a factor to
    separate its flat directions from its curved ones.
    """
    try:
        factor: npt.NDArray[np.float64] = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None
    if float(np.min(np.diag(factor), initial=np.inf)) ** 2 <= pivot_floor:
        return None
    return factor


def _cholesky_with_appended_variable(
    factor: npt.NDArray[np.float64],
    column: npt.NDArray[np.float64],
    diagonal: float,
    pivot_floor: float,
) -> npt.NDArray[np.float64] | None:
    """Cholesky factor of a matrix bordered by one more row and column.

    Args:
        factor: Lower Cholesky factor of the matrix `M`, shape (p, p).
        column: Entries of the new last column against the rows of `M`,
            shape (p,).
        diagonal: New last diagonal entry.
        pivot_floor: Smallest squared pivot accepted, as in
            `_cholesky_if_definite`.

    Returns:
        The factor of `[[M, column], [column', diagonal]]`, shape
        (p + 1, p + 1), or None if its last squared pivot is at or below
        `pivot_floor`.
    """
    size = factor.shape[0]
    border = np.zeros(0)
    if size > 0:
        border = solve_triangular(factor, column, lower=True, check_finite=False)
    pivot = diagonal - float(border @ border)
    if not pivot > pivot_floor:
        return None
    grown = np.zeros((size + 1, size + 1))
    grown[:size, :size] = factor
    grown[size, :size] = border
    grown[size, size] = np.sqrt(pivot)
    return grown


def _null_space_step(
    hessian_free: npt.NDArray[np.float64],
    rows_free: npt.NDArray[np.float64],
    gradient_free: npt.NDArray[np.float64],
    curvature_floor: float,
    tolerance: float,
    func_name: str,
) -> tuple[npt.NDArray[np.float64], bool]:
    """Step on the working set when the free block of the Hessian may be singular.

    In an orthonormal basis `Z` of the null space of `rows_free` the
    objective is a quadratic with Hessian `Z'HZ`, whose eigenvalues separate
    directions of positive curvature from flat ones (those at or below
    `curvature_floor`).

    Args:
        hessian_free: Hessian block of the free variables, shape (p, p).
        rows_free: Working constraint rows over the free variables, shape
            (m, p).
        gradient_free: Objective gradient at the free variables, shape (p,).
        curvature_floor: Curvature at or below which a direction is flat.
        tolerance: Gradient size below which a flat direction has no slope.
        func_name: Public function name used in error messages.

    Returns:
        `(step, bounded)`. With `bounded` True, `step` is the full step to
        the minimiser along the curved directions and no flat direction has
        a slope. With `bounded` False the objective slopes along a flat
        direction, `step` is the steepest descent within the flat directions
        and the objective falls without limit along it.

    Raises:
        RuntimeError: If a direction has curvature below `-curvature_floor`.
    """
    _, singular_values, right = np.linalg.svd(rows_free, full_matrices=True)
    rank = 0
    if singular_values.shape[0] > 0:
        rank_floor = max(rows_free.shape) * float(np.finfo(np.float64).eps * singular_values[0])
        rank = int(np.count_nonzero(singular_values > rank_floor))
    null_basis = right[rank:].T
    if null_basis.shape[1] == 0:
        return np.zeros(gradient_free.shape[0]), True
    reduced_gradient = null_basis.T @ gradient_free
    curvatures, directions = np.linalg.eigh(null_basis.T @ hessian_free @ null_basis)
    if curvatures[0] < -curvature_floor:
        raise RuntimeError(
            f"{func_name}: the problem is not convex (the quadratic term has negative "
            f"curvature along a feasible direction, so it is not positive semi-definite)"
        )
    coefficients = directions.T @ reduced_gradient
    flat = curvatures <= curvature_floor
    flat_slope: npt.NDArray[np.float64] = null_basis @ (directions[:, flat] @ coefficients[flat])
    if float(np.max(np.abs(flat_slope), initial=0.0)) > tolerance:
        return -flat_slope, False
    curved = ~flat
    newton: npt.NDArray[np.float64] = null_basis @ (
        directions[:, curved] @ (coefficients[curved] / curvatures[curved])
    )
    return -newton, True


def _active_set_iterations(
    hessian: npt.NDArray[np.float64],
    linear: npt.NDArray[np.float64],
    eq_matrix: npt.NDArray[np.float64],
    ineq_matrix: npt.NDArray[np.float64],
    ineq_rhs: npt.NDArray[np.float64],
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    func_name: str,
) -> npt.NDArray[np.float64]:
    """Run the active-set iteration of `_solve_convex_qp` from a feasible point `x`.

    The working set is the set of constraints currently held as equalities:
    every equality row, the variables fixed at a bound, and the inequality
    rows that have stopped a step. Each iteration does one of two things.

    - If the objective can still be reduced without leaving the working set,
      it moves toward the minimiser on the working set. A step is cut at the
      first bound or inequality row it reaches, which joins the working set.
    - Otherwise the point minimises the objective on the working set and its
      Lagrange multipliers are examined. If none has the wrong sign the
      Karush-Kuhn-Tucker conditions hold, which for a convex problem
      certifies a global minimiser, and the point is returned. If one does,
      that constraint leaves the working set.

    The step is computed in one of two ways. While the Hessian block of the
    free variables is safely positive definite, its Cholesky factor gives
    the minimiser directly (the Schur-complement solution of the
    equality-constrained problem, Nocedal & Wright 2006, sec. 16.2); the
    factor is extended in O(p^2) when a variable is freed and recomputed
    when one is fixed. Otherwise the step comes from `_null_space_step`,
    which handles zero curvature: a flat direction is followed only if the
    objective slopes along it, alone and as far as the constraints allow,
    and a flat step that nothing stops means the problem is unbounded.

    Multipliers of the working rows are least-squares solutions, so linearly
    dependent working constraints (repeated rows, or more constraints active
    at a point than it has dimensions) need no special handling: dependent
    rows get the minimum-norm multipliers.

    The constraint that leaves is the one with the most wrongly-signed
    multiplier, except directly after a step of zero length, where the
    lowest-indexed candidate leaves and the lowest-indexed blocking
    constraint enters (Bland 1977), which keeps a degenerate point from
    being cycled through.

    Every tolerance is relative. Gradients and multipliers are compared with
    `_QP_RTOL` times the size of the terms the gradient is summed from.
    Those terms can all vanish at an optimum (no linear term, and all weight
    on a variable whose row of `H` is zero), so the comparison is never made
    with less than the change one unit of round-off in the largest entry of
    `x` makes to the largest such term, `eps * max|H| * max|x|`: a gradient
    below that is zero to the resolution of `x`. Before a point that passes
    only by that floor is returned, its free entries within round-off of
    zero (`_QP_PIVOT_RTOL` times the largest entry) are set to exactly zero;
    if that point, or the one a single full step from it leads to, meets the
    conditions without the floor it is returned instead. This recovers exact
    zeros, and entries far below the round-off of the largest one, where the
    optimum has them. Curvature is compared with `_QP_RTOL` times the
    largest absolute row sum of `H`. Multiplying `H` and `linear` by a
    common positive factor therefore leaves every decision unchanged.

    The gradient is updated from the rows of the variables that moved while
    they are few, and recomputed in full before a point is returned, so the
    conditions that certify the result are evaluated without accumulated
    error.

    Args:
        hessian: Symmetric positive semi-definite matrix `H`, shape (n, n).
        linear: Linear coefficients of the objective, shape (n,).
        eq_matrix: Equality rows, shape (m_eq, n), each scaled to a largest
            coefficient of 1.
        ineq_matrix: Inequality rows `ineq_matrix @ x >= ineq_rhs`, shape
            (m_in, n), scaled likewise.
        ineq_rhs: Inequality right-hand sides, shape (m_in,).
        lower: Lower bounds, shape (n,).
        upper: Upper bounds, shape (n,).
        x: Feasible starting point, shape (n,), inside the bounds.
        func_name: Public function name used in error messages.

    Returns:
        A minimiser, shape (n,).

    Raises:
        RuntimeError: If the objective is unbounded below, if `H` has
            negative curvature along a feasible direction, if a step or an
            iterate overflows, or if
            `_QP_MAX_ITER_PER_CONSTRAINT * (n + m_in + 1)` iterations pass
            without the Karush-Kuhn-Tucker conditions being met.
    """
    n = x.shape[0]
    m_eq = eq_matrix.shape[0]
    m_in = ineq_matrix.shape[0]
    abs_hessian = np.abs(hessian)
    linear_scale = float(np.max(np.abs(linear), initial=0.0))
    hessian_scale = float(np.max(abs_hessian, initial=0.0))
    hessian_norm = float(np.max(np.sum(abs_hessian, axis=1), initial=0.0))
    eps = float(np.finfo(np.float64).eps)
    curvature_floor = _QP_RTOL * hessian_norm
    pivot_floor = _QP_CHOLESKY_RTOL * hessian_norm

    pinned = lower == upper
    fixed = (x == lower) | (x == upper)
    # Indices of the free variables, in the order of the rows of `factor`.
    free = np.flatnonzero(~fixed)
    row_active = np.zeros(m_in, dtype=np.bool_)
    stalled = False
    # Set while a copy of the point with its round-off entries zeroed is being
    # examined: the point to return if neither the copy nor one full step
    # from it passes. `stepped` records that the step has been taken.
    fallback: npt.NDArray[np.float64] | None = None
    stepped = False

    # `factor` is the Cholesky factor of hessian[free][:, free]. It is None
    # both before it has been computed (`definite` None) and when that block
    # is known not to be safely positive definite (`definite` False).
    factor: npt.NDArray[np.float64] | None = None
    definite: bool | None = None

    # Gradient, and the size of the terms each of its entries is summed from.
    # `exact` is False once either has been updated instead of recomputed.
    gradient = hessian @ x + linear
    term_scale = abs_hessian @ np.abs(x)
    exact = True

    max_iter = _QP_MAX_ITER_PER_CONSTRAINT * (n + m_in + 1)
    for _ in range(max_iter):
        largest = float(np.max(np.abs(x), initial=0.0))
        term_tolerance = _QP_RTOL * max(linear_scale, float(np.max(term_scale, initial=0.0)))
        tolerance = term_tolerance
        if fallback is None:
            tolerance = max(term_tolerance, eps * hessian_scale * largest)
        rows = np.vstack([eq_matrix, ineq_matrix[row_active]])
        rows_free = rows[:, free]
        gradient_free = gradient[free]
        row_multipliers = _min_norm_least_squares(rows_free.T, gradient_free)
        projected = gradient_free - rows_free.T @ row_multipliers

        slope = float(np.max(np.abs(projected), initial=0.0))
        if slope > tolerance:
            if definite is None:
                factor = _cholesky_if_definite(hessian[np.ix_(free, free)], pivot_floor)
                definite = factor is not None
            if factor is not None:
                # The projected gradient differs from the gradient by a
                # combination of the working rows, which changes the
                # multipliers but not the step, and unlike the gradient it
                # vanishes at the minimiser: the error of the solve then
                # shrinks with it instead of staying at the size of the
                # multipliers.
                solved = cho_solve(
                    (factor, True),
                    np.column_stack([projected, rows_free.T]),
                    check_finite=False,
                )
                bounded = True
                step_free = np.full(free.shape[0], np.nan)
                if np.all(np.isfinite(solved)):
                    descent, response = solved[:, 0], solved[:, 1:]
                    step_free = response @ _min_norm_least_squares(
                        rows_free @ response, rows_free @ descent
                    )
                    step_free -= descent
                    # How exactly this step keeps the working rows depends on
                    # the conditioning of the factor; removing its component
                    # along the rows restores that to round-off.
                    step_free -= _min_norm_least_squares(rows_free, rows_free @ step_free)
            else:
                step_free, bounded = _null_space_step(
                    hessian[np.ix_(free, free)],
                    rows_free,
                    gradient_free,
                    curvature_floor,
                    tolerance,
                    func_name,
                )
            if not np.all(np.isfinite(step_free)):
                raise RuntimeError(
                    f"{func_name}: the solution is not finite (the step toward the optimum "
                    f"overflows double precision)"
                )
            step = np.zeros(n)
            step[free] = step_free

            # Components of the step below round-off of its largest one cannot
            # tell which way a variable or a row is moving, so they block
            # nothing.
            abs_step = np.abs(step)
            negligible = _QP_PIVOT_RTOL * float(np.max(abs_step))
            falling = (step < -negligible) & np.isfinite(lower)
            rising = (step > negligible) & np.isfinite(upper)
            bound = np.where(falling, lower, upper)
            reach_bound = np.full(n, np.inf)
            moving = falling | rising
            reach_bound[moving] = (bound[moving] - x[moving]) / step[moving]
            row_rate = ineq_matrix @ step
            closing = ~row_active & (row_rate < -_QP_PIVOT_RTOL * (np.abs(ineq_matrix) @ abs_step))
            reach_row = np.full(m_in, np.inf)
            reach_row[closing] = (ineq_rhs[closing] - ineq_matrix[closing] @ x) / row_rate[closing]
            reach = np.maximum(np.concatenate([reach_bound, reach_row]), 0.0)
            blocker = int(np.argmin(reach))
            length = float(reach[blocker])

            full_step = bounded and length > 1.0
            if fallback is not None:
                if stepped or not full_step:
                    return fallback
                stepped = True
            if not full_step and not np.isfinite(length):
                raise RuntimeError(
                    f"{func_name}: the problem is unbounded (the objective keeps improving along "
                    f"a direction of zero curvature that no constraint stops)"
                )
            moved = np.clip(x + (step if full_step else length * step), lower, upper)
            if not full_step and blocker < n:
                moved[blocker] = bound[blocker]
            if not np.all(np.isfinite(moved)):
                raise RuntimeError(
                    f"{func_name}: the solution is not finite (it overflows double precision)"
                )
            unchanged = bool(np.array_equal(moved, x))
            if not unchanged:
                if 2 * free.shape[0] < n:
                    gradient = gradient + (moved[free] - x[free]) @ hessian[free]
                    term_scale = (
                        term_scale + (np.abs(moved[free]) - np.abs(x[free])) @ (abs_hessian[free])
                    )
                    exact = False
                else:
                    gradient = hessian @ moved + linear
                    term_scale = abs_hessian @ np.abs(moved)
                    exact = True
                x = moved
            if not full_step:
                if blocker < n:
                    fixed[blocker] = True
                    free = free[free != blocker]
                    factor, definite = None, None
                else:
                    row_active[blocker - n] = True
                stalled = unchanged
                continue
            if not unchanged:
                stalled = False
                continue

        # The point minimises the objective on the working set. gradient =
        # rows' * row_multipliers + reduced_cost, where reduced_cost is
        # non-zero only on fixed variables and is their bound multiplier.
        reduced_cost = gradient - rows.T @ row_multipliers
        releasable = fixed & ~pinned
        at_lower = releasable & (x == lower)
        at_upper = releasable & ~at_lower
        # Rate at which the objective falls when a constraint is released: a
        # variable at its lower bound gains from rising if its reduced cost is
        # negative, one at its upper bound from falling if it is positive, and
        # an inequality row from being left if its multiplier is negative.
        row_gain = np.full(m_in, -np.inf)
        row_gain[row_active] = -row_multipliers[m_eq:]
        gain = np.concatenate(
            [np.where(at_lower, -reduced_cost, np.where(at_upper, reduced_cost, -np.inf)), row_gain]
        )
        candidates = np.flatnonzero(gain > tolerance)
        if candidates.shape[0] == 0:
            if exact:
                if slope <= term_tolerance and not np.any(gain > term_tolerance):
                    return x
                if fallback is not None:
                    return fallback
                round_off = free[np.abs(x[free]) <= _QP_PIVOT_RTOL * largest]
                restored = x.copy()
                restored[round_off] = np.clip(0.0, lower[round_off], upper[round_off])
                if np.array_equal(restored, x):
                    return x
                fallback, x = x, restored
            gradient = hessian @ x + linear
            term_scale = abs_hessian @ np.abs(x)
            exact = True
            continue
        if fallback is not None:
            return fallback
        leaving = int(candidates[0]) if stalled else int(np.argmax(gain))
        if leaving < n:
            fixed[leaving] = False
            if factor is not None:
                factor = _cholesky_with_appended_variable(
                    factor, hessian[free, leaving], float(hessian[leaving, leaving]), pivot_floor
                )
                definite = factor is not None
            free = np.append(free, leaving)
        else:
            row_active[leaving - n] = False

    raise RuntimeError(
        f"{func_name}: the active-set iteration did not converge within {max_iter} iterations"
    )


def _fully_invested_minimiser(
    hessian: npt.NDArray[np.float64],
    linear: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    allow_short: bool,
    func_name: str,
) -> npt.NDArray[np.float64]:
    """Minimise `0.5 * w'Hw + linear'w` over weights that sum to 1.

    Args:
        hessian: Symmetric positive semi-definite matrix `H`, shape (k, k).
        linear: Linear coefficients, shape (k,).
        start: Feasible weights the iteration starts from, shape (k,).
        allow_short: If False, weights are also constrained to be
            non-negative.
        func_name: Public function name used in error messages.

    Returns:
        Optimal weights of shape (k,).
    """
    k = linear.shape[0]
    return _solve_convex_qp(
        hessian=hessian,
        linear=linear,
        eq_matrix=np.ones((1, k)),
        eq_rhs=np.ones(1),
        ineq_matrix=np.zeros((0, k)),
        ineq_rhs=np.zeros(0),
        lower=np.full(k, -np.inf if allow_short else 0.0),
        upper=np.full(k, np.inf),
        start=start,
        func_name=func_name,
    )


def _long_only_start(
    hessian: npt.NDArray[np.float64],
    linear: npt.NDArray[np.float64],
    fallback: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Feasible long-only weights close to the minimiser of `0.5 * w'Hw + linear'w`.

    Solves the budget-only problem on the assets still held, drops the ones
    that come out negative, and repeats until none does. The result is
    non-negative and sums to 1 but is not certified optimal: an asset dropped
    in an early round can belong to the optimum. `_solve_convex_qp` started
    from it restores such assets and certifies the result, in a number of
    iterations that depends on how far this guess is from the optimum rather
    than on how many assets the optimum holds.

    Args:
        hessian: Symmetric positive semi-definite matrix `H`, shape (k, k).
        linear: Linear coefficients, shape (k,).
        fallback: Feasible weights returned when a held block of `H` is not
            numerically positive definite, so the budget-only problem on it
            has no unique solution.

    Returns:
        Starting weights of shape (k,).
    """
    k = linear.shape[0]
    held = np.arange(k)
    while held.size > 0:
        block = hessian[np.ix_(held, held)]
        try:
            factor = cho_factor(block, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return fallback
        ones = np.ones(held.size)
        solved = cho_solve(factor, np.column_stack((ones, linear[held])), check_finite=False)
        inverse_ones, inverse_linear = solved[:, 0], solved[:, 1]
        denominator = float(ones @ inverse_ones)
        if not (np.isfinite(denominator) and denominator > 0.0):
            return fallback
        multiplier = (1.0 + float(ones @ inverse_linear)) / denominator
        weights = multiplier * inverse_ones - inverse_linear
        if not np.all(np.isfinite(weights)):
            return fallback
        negative = weights < 0.0
        if not negative.any():
            total = float(weights.sum())
            if not (np.isfinite(total) and total > 0.0):
                return fallback
            start = np.zeros(k)
            start[held] = weights / total
            return start
        held = held[~negative]
    return fallback


def _scaled_symmetric_part(
    cov_matrix: npt.NDArray[np.float64], scale: float
) -> npt.NDArray[np.float64]:
    """`scale * (cov_matrix + cov_matrix.T)` as float64, the Hessian of `0.5 * scale * w'(A + A')w`.

    Entries too large for the sum or the product come out infinite (or NaN
    for `scale = 0`) without a NumPy warning; `_solve_convex_qp` rejects a
    non-finite Hessian with its own error.
    """
    cov = np.asarray(cov_matrix, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        hessian: npt.NDArray[np.float64] = scale * (cov + cov.T)
    return hessian


def _validate_min_variance_inputs(cov_matrix: npt.NDArray[np.float64]) -> None:
    _validate_cov_matrix(cov_matrix)
    _validate_cov_matrix_is_positive_semidefinite(cov_matrix)


def min_variance_weights(
    cov_matrix: npt.NDArray[np.float64],
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute the minimum-variance portfolio: min w^T Sigma w s.t. sum(w) = 1.

    A convex quadratic program, solved exactly (to floating-point
    round-off) by the active-set method of `_solve_convex_qp`. The result
    does not depend on the units of `cov_matrix`. When the minimum is not
    unique (a singular `cov_matrix` can give several portfolios the same
    variance), one minimiser is returned.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k),
            positive semi-definite.
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1. With
        `allow_short=False` no weight is negative and an asset the
        constraint excludes has weight exactly 0.

    Raises:
        ValueError: If `cov_matrix` is not a square 2D array, contains a
            NaN or infinite value, or is not positive semi-definite (its
            symmetric part has an eigenvalue below -1e-12 times its largest
            absolute row sum).
        RuntimeError: If `cov_matrix` is too large for `cov_matrix +
            cov_matrix.T` to be finite, or if the active-set iteration
            exceeds its iteration cap. No silent fallback is returned in
            either case.

    References:
        Markowitz (1952); Nocedal & Wright (2006), ch. 16. See
        docs/REFERENCES.md.
    """
    _validate_min_variance_inputs(cov_matrix)
    return _min_variance_weights(cov_matrix, allow_short)


def _min_variance_weights(
    cov_matrix: npt.NDArray[np.float64],
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    # w'Sigma w = 0.5 * w'(Sigma + Sigma')w, so the Hessian is Sigma + Sigma'.
    hessian = _scaled_symmetric_part(cov_matrix, 1.0)
    if allow_short:
        start = np.full(k, 1.0 / k)
    else:
        vertex = np.zeros(k)
        vertex[int(np.argmin(np.diag(hessian)))] = 1.0
        start = _long_only_start(hessian, np.zeros(k), vertex)
    return _fully_invested_minimiser(
        hessian, np.zeros(k), start, allow_short, "min_variance_weights"
    )


def _validate_mean_variance_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    _validate_expected_returns(expected_returns, cov_matrix)
    _validate_risk_aversion_is_finite(risk_aversion)
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")
    _validate_cov_matrix_is_positive_semidefinite(cov_matrix)


def mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Maximize w^T mu - 0.5*risk_aversion*w^T Sigma w s.t. sum(w) = 1, w >= 0.

    A convex quadratic program, solved exactly (to floating-point
    round-off) by the active-set method of `_solve_convex_qp`. The result
    does not depend on the units of the inputs: multiplying
    `expected_returns` and `cov_matrix` by a common positive factor leaves
    the weights unchanged. When the maximum is not unique (a singular
    `risk_aversion * cov_matrix`), one maximiser is returned.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k),
            positive semi-definite.
        risk_aversion: Risk-aversion coefficient lambda >= 0.
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1. With
        `allow_short=False` no weight is negative and an asset the
        constraint excludes has weight exactly 0.

    Raises:
        ValueError: If the shapes are inconsistent, any input contains a
            NaN or infinite value, `risk_aversion` is negative, or
            `cov_matrix` is not positive semi-definite (its symmetric part
            has an eigenvalue below -1e-12 times its largest absolute row
            sum).
        RuntimeError: If the utility is unbounded above, which requires
            `allow_short=True` and a `risk_aversion * cov_matrix` with a
            direction of zero variance along which expected return can be
            added without limit (in particular `risk_aversion = 0`, and any
            covariance whose smallest eigenvalue is below about 1e-12 times
            its largest absolute row sum); if `risk_aversion * cov_matrix`
            or the optimal weights overflow double precision; or if the
            active-set iteration exceeds its iteration cap. No silent
            fallback is returned in any case.

    References:
        Markowitz (1952); Nocedal & Wright (2006), ch. 16. See
        docs/REFERENCES.md.
    """
    _validate_mean_variance_inputs(expected_returns, cov_matrix, risk_aversion)
    return _mean_variance_weights(expected_returns, cov_matrix, risk_aversion, allow_short)


def _mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    mu = np.asarray(expected_returns, dtype=np.float64)
    k = mu.shape[0]
    hessian = _scaled_symmetric_part(cov_matrix, 0.5 * risk_aversion)
    if allow_short:
        start = np.full(k, 1.0 / k)
    else:
        vertex = np.zeros(k)
        vertex[int(np.argmax(mu - 0.5 * np.diag(hessian)))] = 1.0
        start = _long_only_start(hessian, -mu, vertex)
    return _fully_invested_minimiser(hessian, -mu, start, allow_short, "mean_variance_weights")


def _validate_unconstrained_mean_variance_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    _validate_expected_returns(expected_returns, cov_matrix)
    _validate_risk_aversion_is_finite(risk_aversion)
    if risk_aversion <= 0.0:
        raise ValueError("risk_aversion must be strictly positive")


def unconstrained_mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    """Solve the unconstrained mean-variance first-order condition Sigma @ w = mu / lambda.

    This is the closed-form maximizer of w^T mu - 0.5*risk_aversion*w^T Sigma w
    with NO constraints imposed at all: no budget constraint (sum(w) == 1), no
    long-only constraint. The resulting weights need not sum to 1, may be
    negative (short positions), and may exceed 1x gross exposure (leverage).
    Use this for a raw tangency-style allocation (e.g. as a prior or building
    block in a downstream optimizer that layers its own constraints); use
    `mean_variance_weights` instead when a fully-invested, optionally
    long-only portfolio is required.

    Because this is a single linear solve with no constraints to iterate
    over, it is faster than `mean_variance_weights`.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k).
        risk_aversion: Risk-aversion coefficient lambda, must be strictly
            positive (weights are divided by it).

    Returns:
        Unconstrained portfolio weights of shape (k,). Not guaranteed to sum
        to 1.

    Raises:
        ValueError: If the shapes are inconsistent, any input contains a
            NaN or infinite value, or `risk_aversion` is not strictly
            positive.
    """
    _validate_unconstrained_mean_variance_inputs(expected_returns, cov_matrix, risk_aversion)
    return _unconstrained_mean_variance_weights(expected_returns, cov_matrix, risk_aversion)


def _unconstrained_mean_variance_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_aversion: float,
) -> npt.NDArray[np.float64]:
    # np.linalg.solve over an explicit np.linalg.inv: forming the inverse
    # explicitly squares the condition number's effect on rounding error,
    # whereas solve dispatches to a direct LU factorization of cov_matrix.
    return np.linalg.solve(cov_matrix, expected_returns) / risk_aversion


def _validate_l1_turnover_penalized_inputs(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
) -> None:
    _validate_cov_matrix(cov_matrix)
    _validate_expected_returns(expected_returns, cov_matrix)
    if previous_weights.ndim != 1 or previous_weights.shape[0] != cov_matrix.shape[0]:
        raise ValueError("previous_weights must be a 1D array matching cov_matrix's dimension")
    if not np.all(np.isfinite(previous_weights)):
        raise ValueError("previous_weights must contain only finite values")
    if abs(previous_weights.sum() - 1.0) > 1e-6:
        raise ValueError("previous_weights must sum to 1")
    if not np.isfinite(cost_bps):
        raise ValueError("cost_bps must be finite")
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative")
    _validate_risk_aversion_is_finite(risk_aversion)
    if risk_aversion < 0.0:
        raise ValueError("risk_aversion must be non-negative")
    _validate_cov_matrix_is_positive_semidefinite(cov_matrix)


def l1_turnover_penalized_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool = False,
) -> npt.NDArray[np.float64]:
    """Mean-variance weights penalized for turnover away from previous_weights.

    Maximizes w^T mu - 0.5*risk_aversion*w^T Sigma w - (cost_bps/10000)*||w -
    previous_weights||_1, s.t. sum(w) = 1, w >= 0 (unless allow_short). With
    cost_bps = 0 this is the program `mean_variance_weights` solves.

    The trade is written as purchases minus sales, both non-negative, which
    turns the L1 cost into a linear term and the problem into a convex
    quadratic program; it is solved exactly (to floating-point round-off)
    by the active-set method of `_solve_convex_qp`, started from no trade.
    The result does not depend on the units of the inputs: multiplying
    `expected_returns`, `cov_matrix` and `cost_bps` by a common positive
    factor leaves the weights unchanged.

    Args:
        expected_returns: Expected asset returns, shape (k,).
        cov_matrix: Covariance matrix of asset returns, shape (k, k),
            positive semi-definite.
        previous_weights: Current portfolio weights before rebalancing, shape
            (k,), must sum to 1 (within 1e-6). If they sum to 1 only within
            that tolerance and not to round-off, the difference is traded
            away, so the result sums to 1. With `allow_short=False` a
            negative entry is a short position that the result closes.
        cost_bps: Proportional transaction cost in basis points per unit of
            L1 turnover (must be non-negative).
        risk_aversion: Risk-aversion coefficient lambda >= 0.
        allow_short: If False (default), also constrains w >= 0.

    Returns:
        Portfolio weights of shape (k,), summing to 1. An asset that is
        optimally left untraded keeps its `previous_weights` entry bit for
        bit, so a no-trade optimum is returned as a copy of
        `previous_weights`. With `allow_short=False` no weight is negative
        and a position that is sold out entirely has weight exactly 0.

    Raises:
        ValueError: If the shapes are inconsistent, any input contains a
            NaN or infinite value, `previous_weights` does not sum to 1,
            `cost_bps` or `risk_aversion` is negative, or `cov_matrix` is
            not positive semi-definite (its symmetric part has an eigenvalue
            below -1e-12 times its largest absolute row sum).
        RuntimeError: If the utility is unbounded above, which requires
            `allow_short=True` and a `risk_aversion * cov_matrix` with a
            direction of zero variance along which expected return, net of
            the cost, can be added without limit; if `risk_aversion *
            cov_matrix` or the optimal weights overflow double precision; or
            if the active-set iteration exceeds its iteration cap. No silent
            fallback is returned in any case.

    References:
        Markowitz (1952); Lobo, Fazel & Boyd (2007); Nocedal & Wright
        (2006), ch. 16. See docs/REFERENCES.md.
    """
    _validate_l1_turnover_penalized_inputs(
        expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion
    )
    return _l1_turnover_penalized_weights(
        expected_returns, cov_matrix, previous_weights, cost_bps, risk_aversion, allow_short
    )


def _l1_turnover_penalized_weights(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    previous_weights: npt.NDArray[np.float64],
    cost_bps: float,
    risk_aversion: float,
    allow_short: bool,
) -> npt.NDArray[np.float64]:
    mu = np.asarray(expected_returns, dtype=np.float64)
    previous = np.asarray(previous_weights, dtype=np.float64)
    k = mu.shape[0]
    hessian = _scaled_symmetric_part(cov_matrix, 0.5 * risk_aversion)
    cost_rate = cost_bps / 10000.0

    # ||w - previous||_1 is not differentiable where w_i == previous_i. Writing
    # w = previous + buy - sell with buy, sell >= 0 and charging cost_rate on
    # buy + sell gives a smooth quadratic program in (buy, sell) with the same
    # optimum: buying and selling the same asset at once only adds cost, so at
    # most one of buy_i, sell_i is positive and buy_i + sell_i = |w_i -
    # previous_i|. Up to a constant its objective is
    #   0.5*(buy - sell)'H(buy - sell) + (marginal + cost_rate)'buy
    #       + (cost_rate - marginal)'sell,
    # with marginal the gradient of the smooth part at no trade.
    with np.errstate(over="ignore", invalid="ignore"):
        marginal = hessian @ previous - mu
        split_linear = np.concatenate([marginal + cost_rate, cost_rate - marginal])
    split_hessian = np.block([[hessian, -hessian], [-hessian, hessian]])

    # Net purchases must make the weights sum to 1. Previous weights that
    # already do so to round-off need no net trade at all, which keeps the
    # no-trade point feasible and lets it be returned unchanged.
    net_purchases = 1.0 - float(previous.sum())
    if abs(net_purchases) <= _QP_RTOL * (1.0 + float(np.abs(previous).sum())):
        net_purchases = 0.0

    lower = np.zeros(2 * k)
    upper = np.full(2 * k, np.inf)
    if not allow_short:
        # w_i >= 0 in terms of the trades: a short previous position has to be
        # bought back, and no more than a long previous position can be sold.
        lower[:k] = np.maximum(-previous, 0.0)
        upper[k:] = np.maximum(previous, 0.0)

    trades = _solve_convex_qp(
        hessian=split_hessian,
        linear=split_linear,
        eq_matrix=np.concatenate([np.ones(k), -np.ones(k)])[None, :],
        eq_rhs=np.array([net_purchases]),
        ineq_matrix=np.zeros((0, 2 * k)),
        ineq_rhs=np.zeros(0),
        lower=lower,
        upper=upper,
        start=np.zeros(2 * k),
        func_name="l1_turnover_penalized_weights",
    )
    return np.asarray(previous + trades[:k] - trades[k:], dtype=np.float64)


def _validate_risk_parity_inputs(cov_matrix: npt.NDArray[np.float64]) -> None:
    _validate_cov_matrix(cov_matrix)
    if np.any(np.diag(cov_matrix) <= 0.0):
        raise ValueError("cov_matrix must have strictly positive variances on its diagonal")


def risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the equal-risk-contribution (risk parity) portfolio.

    Solves Spinu's (2013) strictly convex reformulation
    `min_{w > 0} 0.5 * w'Sigma*w - (1/k) * sum(ln w)` by damped Newton
    iteration from the inverse-volatility portfolio, then rescales `w` to
    sum to 1. The result is the unique long-only portfolio whose normalised
    risk contributions `w_i * (Sigma*w)_i / (w'Sigma*w)` all equal `1/k`
    (Maillard, Roncalli & Teiletche 2010).

    Convergence target is a maximum risk-contribution deviation from `1/k`
    of 1e-10. On covariances so ill-conditioned that floating-point
    round-off in `Sigma*w` prevents the iteration from resolving the
    contributions that finely (a near-singular matrix, or eigenvalues
    spanning 1e8), the iteration stops once it can no longer improve and
    the best iterate is returned provided its deviation is within 1e-6;
    otherwise it raises.

    Args:
        cov_matrix: Covariance matrix of asset returns, shape (k, k), with
            strictly positive variances on the diagonal.

    Returns:
        Portfolio weights of shape (k,), strictly positive and summing to
        1, with equal risk contributions.

    Raises:
        ValueError: If `cov_matrix` is not square, contains a NaN or
            infinite value, or has a non-positive variance on its diagonal.
        RuntimeError: If the Newton iteration neither reaches the 1e-10
            target within `_RISK_PARITY_MAX_ITER` iterations nor stalls
            within the 1e-6 fallback tolerance. No silent fallback is
            returned in this case.
    """
    _validate_risk_parity_inputs(cov_matrix)
    return _risk_parity_weights(cov_matrix)


_RISK_PARITY_MAX_ITER = 200
_RISK_PARITY_TOL = 1e-10
_RISK_PARITY_STALL_TOL = 1e-6
_RISK_PARITY_ROUNDOFF_ITERS = 5


def _risk_parity_weights(cov_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    k = cov_matrix.shape[0]
    # Rescaling Sigma by a positive constant rescales the unconstrained
    # minimiser by its inverse square root and leaves the normalised weights
    # unchanged, so the problem is solved on a unit-mean-variance copy for
    # conditioning; the covariance's own scale never enters the tolerance.
    sigma = cov_matrix / float(np.mean(np.diag(cov_matrix)))
    budget = np.full(k, 1.0 / k)

    def objective(w: npt.NDArray[np.float64]) -> float:
        return float(0.5 * (w @ sigma @ w) - budget @ np.log(w))

    def gradient(w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        result: npt.NDArray[np.float64] = sigma @ w - budget / w
        return result

    def contribution_deviation(w: npt.NDArray[np.float64]) -> float:
        sigma_w = sigma @ w
        return float(np.max(np.abs(w * sigma_w / (w @ sigma_w) - budget)))

    # Inverse-volatility start: already the exact solution for a diagonal
    # Sigma and on the right scale for any other one.
    w = 1.0 / np.sqrt(np.diag(sigma))
    w /= w.sum()
    best_w, best_dev = w, contribution_deviation(w)
    # Once the objective can no longer resolve the predicted decrease, the
    # iterate is within round-off of the optimum; Newton's quadratic
    # convergence means a handful of further iterations is all that can
    # still help, after which the iteration is only wandering in round-off.
    roundoff_iters = 0
    for _ in range(_RISK_PARITY_MAX_ITER):
        if best_dev < _RISK_PARITY_TOL:
            return np.asarray(best_w / best_w.sum(), dtype=np.float64)
        if roundoff_iters > _RISK_PARITY_ROUNDOFF_ITERS:
            break
        grad = gradient(w)
        hessian = sigma + np.diag(budget / w**2)
        direction = np.linalg.solve(hessian, -grad)
        f_current = objective(w)
        slope = float(grad @ direction)
        grad_norm = float(np.linalg.norm(grad))
        # Backtracking line search: stay strictly inside w > 0 and satisfy
        # the Armijo sufficient-decrease condition. Close to the optimum the
        # predicted decrease drops below what `objective` can resolve in
        # floating point, so the step is then accepted on gradient-norm
        # descent instead (the Newton direction of a strictly convex
        # function is also the Newton direction of its gradient system).
        step = 1.0
        accepted = False
        while step >= 1e-12:
            candidate = w + step * direction
            if np.all(candidate > 0.0):
                predicted_decrease = -1e-4 * step * slope
                if objective(candidate) <= f_current - predicted_decrease:
                    accepted = True
                    break
                if predicted_decrease < 1e-12 * (1.0 + abs(f_current)):
                    roundoff_iters += 1
                    if float(np.linalg.norm(gradient(candidate))) < grad_norm:
                        accepted = True
                    break
            step *= 0.5
        if not accepted:
            break
        w = candidate
        dev = contribution_deviation(w)
        if dev < best_dev:
            best_w, best_dev = w, dev

    if best_dev < _RISK_PARITY_STALL_TOL:
        return np.asarray(best_w / best_w.sum(), dtype=np.float64)
    raise RuntimeError(
        f"risk_parity_weights: Newton iteration did not reach the risk-contribution "
        f"tolerance {_RISK_PARITY_TOL} within {_RISK_PARITY_MAX_ITER} iterations "
        f"(best deviation {best_dev:.3e} exceeds the {_RISK_PARITY_STALL_TOL} "
        f"fallback tolerance); the covariance matrix may be numerically invalid"
    )


def _validate_kelly_fraction_inputs(expected_return: float, variance: float) -> None:
    if not np.isfinite(expected_return):
        raise ValueError("expected_return must be finite")
    if not np.isfinite(variance):
        raise ValueError("variance must be finite")
    if variance <= 0.0:
        raise ValueError("variance must be strictly positive")


def kelly_fraction(expected_return: float, variance: float, clip: bool = True) -> float:
    """Compute the single-asset Kelly fraction f* = expected_return / variance.

    Args:
        expected_return: Excess expected return E[r] - r_f.
        variance: Return variance sigma^2 (must be strictly positive).
        clip: If True (default), clamp the result to [-1, 1] — a leverage
            cap, not part of Kelly's (1956) original formula, which has no
            such bound (a high-edge, low-variance input can legitimately
            call for a fraction outside [-1, 1] under the textbook
            criterion). Pass `clip=False` for the unclamped textbook value.

    Returns:
        Kelly fraction; clamped to [-1, 1] if `clip` is True.

    Raises:
        ValueError: If `expected_return` or `variance` is NaN or infinite,
            or `variance` is not strictly positive.
    """
    _validate_kelly_fraction_inputs(expected_return, variance)
    return _kelly_fraction(expected_return, variance, clip)


def _kelly_fraction(expected_return: float, variance: float, clip: bool) -> float:
    fraction = expected_return / variance
    if not clip:
        return float(fraction)
    return float(np.clip(fraction, -1.0, 1.0))
