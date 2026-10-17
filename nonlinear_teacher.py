"""Full nonlinear stochastic growth teacher, solved by endogenous-grid iteration.
NumPy CPU only. Importing this module does not solve or write anything.
Run ``python3 nonlinear_teacher.py`` to solve, audit and save the teacher.
"""
import os
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_variable] = '1'
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PARAMETERS = dict(beta=.96, sigma=2., alpha=.36, delta=.08, rho=.9, innovation_std=.03)
PARAMETERS['kss'] = (PARAMETERS['alpha']/(1/PARAMETERS['beta']-1+PARAMETERS['delta']))**(1/(1-PARAMETERS['alpha']))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rouwenhorst(n=7, rho=.9, innovation_std=.03):
    probability = (1+rho)/2
    transition = np.array([[probability, 1-probability], [1-probability, probability]])
    for size in range(3, n+1):
        old = transition
        transition = np.zeros((size, size))
        transition[:-1, :-1] += probability*old
        transition[:-1, 1:] += (1-probability)*old
        transition[1:, :-1] += (1-probability)*old
        transition[1:, 1:] += probability*old
        transition[1:-1] /= 2
    log_std = innovation_std/np.sqrt(1-rho**2)
    logz = np.linspace(-np.sqrt(n-1)*log_std, np.sqrt(n-1)*log_std, n)
    # Solve stationarity with an explicit unit-mass row, rather than simulation.
    stationary_system = transition.T-np.eye(n)
    stationary_system[-1] = 1.
    stationary_rhs = np.zeros(n); stationary_rhs[-1] = 1.
    stationary = np.linalg.solve(stationary_system, stationary_rhs)
    require(np.max(abs(transition.sum(axis=1)-1)) < 1e-12, 'Markov rows do not sum to one')
    require(np.min(stationary) > 0. and np.max(abs(stationary@transition-stationary)) < 1e-12,
            'Invalid stationary distribution')
    return transition, logz, np.exp(logz), stationary


class GrowthTeacher:
    def __init__(self, k_grid, consumption_grid, P, logz, z, stationary, parameters=None):
        self.k_grid = np.asarray(k_grid)
        self.consumption_grid = np.asarray(consumption_grid)
        self.P, self.logz, self.z, self.stationary = map(np.asarray, (P, logz, z, stationary))
        self.parameters = dict(PARAMETERS if parameters is None else parameters)

    def consumption(self, k, z_index):
        capital, state = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(z_index, dtype=int))
        if np.any(capital < self.k_grid[0]) or np.any(capital > self.k_grid[-1]):
            raise ValueError('Capital is outside the solved grid; extrapolation is forbidden')
        if np.any(state < 0) or np.any(state >= len(self.z)):
            raise ValueError('Productivity index is outside the finite Markov state space')
        result = np.empty(capital.size)
        flat_capital, flat_state = capital.ravel(), state.ravel()
        for j in range(len(self.z)):
            selected = flat_state == j
            result[selected] = np.interp(flat_capital[selected], self.k_grid, self.consumption_grid[j])
        return result.reshape(capital.shape)

    def resources(self, k, z_index):
        capital = np.asarray(k, dtype=float)
        return self.z[np.asarray(z_index, dtype=int)]*capital**self.parameters['alpha']+(1-self.parameters['delta'])*capital

    def next_capital(self, k, z_index):
        return self.resources(k, z_index)-self.consumption(k, z_index)

    def prices(self, k, z_index):
        capital = np.asarray(k, dtype=float)
        z = self.z[np.asarray(z_index, dtype=int)]
        alpha = self.parameters['alpha']
        return dict(rental_rate=alpha*z*capital**(alpha-1), wage=(1-alpha)*z*capital**alpha)


def load_teacher(path=None):
    with np.load(ROOT/'nonlinear_teacher.npz' if path is None else path, allow_pickle=False) as saved:
        return GrowthTeacher(saved['k_grid'], saved['consumption_grid'], saved['P'], saved['logz'],
                             saved['z'], saved['stationary'], json.loads(str(saved['parameters_json'])))


_DEFAULT_TEACHER = None


def consumption(k, z_index, teacher=None):
    """Evaluate saved consumption. Loading is lazy; import performs no file reads."""
    global _DEFAULT_TEACHER
    if teacher is None:
        if _DEFAULT_TEACHER is None:
            _DEFAULT_TEACHER = load_teacher()
        teacher = _DEFAULT_TEACHER
    return teacher.consumption(k, z_index)


def next_capital(k, z_index, teacher=None):
    """Evaluate next capital from the resource constraint and saved policy."""
    global _DEFAULT_TEACHER
    if teacher is None:
        if _DEFAULT_TEACHER is None:
            _DEFAULT_TEACHER = load_teacher()
        teacher = _DEFAULT_TEACHER
    return teacher.next_capital(k, z_index)


def solve_teacher(points, deadline, initial=None):
    p = PARAMETERS
    transition, logz, z, stationary = rouwenhorst(7, p['rho'], p['innovation_std'])
    # Quadratic spacing gives more capital resolution near the lower boundary.
    k_grid = p['kss']*(.02+(5.-.02)*np.linspace(0., 1., points)**2)
    resources = z[:, None]*k_grid**p['alpha']+(1-p['delta'])*k_grid
    returns = p['alpha']*z[:, None]*k_grid**(p['alpha']-1)+1-p['delta']
    if initial is None:
        consumption = resources-.95*k_grid
    else:
        consumption = np.vstack([initial.consumption(k_grid, j) for j in range(7)])
    history = []
    started = time.perf_counter()
    for iteration in range(1, 3001):
        if time.perf_counter() > deadline:
            raise RuntimeError('The 60-second solver pilot budget expired; no teacher was certified')
        expected_marginal = p['beta']*(transition@(consumption**(-p['sigma'])*returns))
        endogenous_consumption = expected_marginal**(-1/p['sigma'])
        wealth = k_grid+endogenous_consumption
        # Invert z*k^alpha+(1-delta)*k=wealth by vector bisection.
        low = np.zeros_like(wealth)
        high = wealth/(1-p['delta'])
        for _ in range(42):
            mid = (low+high)/2
            above = z[:, None]*mid**p['alpha']+(1-p['delta'])*mid > wealth
            high = np.where(above, mid, high)
            low = np.where(above, low, mid)
        endogenous_capital = (low+high)/2
        require(np.all(np.diff(endogenous_capital, axis=1) > 0.), 'Endogenous capital is not increasing')
        require(np.all(endogenous_capital[:, -1] >= k_grid[-1]), 'Upper grid requires forbidden extrapolation')
        updated = np.vstack([np.interp(k_grid, endogenous_capital[j], endogenous_consumption[j]) for j in range(7)])
        # The numerical capital floor imposes a borrowing constraint only when binding.
        updated = np.minimum(updated, resources-k_grid[0])
        relative_change = float(np.max(abs(updated-consumption)/np.maximum(consumption, 1e-12)))
        consumption = updated
        if iteration % 50 == 0 or relative_change < 2e-11:
            history.append(dict(iteration=iteration, maximum_relative_policy_change=relative_change))
        if relative_change < 2e-11:
            break
    else:
        raise RuntimeError('Nonlinear policy iteration did not converge')
    require(np.all(consumption > 0.), 'Non-positive consumption')
    teacher = GrowthTeacher(k_grid, consumption, transition, logz, z, stationary, p)
    return teacher, dict(points=points, iterations=iteration, seconds=time.perf_counter()-started,
        maximum_relative_policy_change=relative_change, history=history, capital_spacing='quadratic',
        capital_floor_over_kss=.02, capital_ceiling_over_kss=5., bisection_steps=42)


def euler_diagnostics(teacher, k, states):
    p = teacher.parameters
    current = teacher.consumption(k, states)
    next_k = teacher.next_capital(k, states)
    require(np.all(current > 0.) and np.all(next_k >= teacher.k_grid[0]) and np.all(next_k <= teacher.k_grid[-1]),
            'Fresh states violate feasibility or leave solved capital grid')
    future_consumption = np.vstack([teacher.consumption(next_k, j) for j in range(7)])
    future_returns = p['alpha']*teacher.z[:, None]*next_k**(p['alpha']-1)+1-p['delta']
    expected_marginal = np.sum(teacher.P[states].T*future_consumption**(-p['sigma'])*future_returns, axis=0)
    implied_consumption = (p['beta']*expected_marginal)**(-1/p['sigma'])
    relative = implied_consumption/current-1.
    return dict(samples=len(k), consumption_equivalent_relative_error_max=float(np.max(abs(relative))),
        consumption_equivalent_relative_error_rms=float(np.sqrt(np.mean(relative**2))),
        consumption_sd=float(current.std()),
        consumption_equivalent_error_rms_normalized_by_c_sd=float(np.sqrt(np.mean((implied_consumption-current)**2))/current.std()),
        consumption_equivalent_error_max_normalized_by_c_sd=float(np.max(abs(implied_consumption-current))/current.std()),
        consumption_equivalent_relative_error_quantiles=np.quantile(abs(relative), [.5, .9, .99, 1.]).tolist(),
        minimum_consumption=float(current.min()), minimum_next_capital_over_kss=float(next_k.min()/p['kss']),
        maximum_next_capital_over_kss=float(next_k.max()/p['kss']),
        capital_floor_bindings=int(np.sum(next_k <= teacher.k_grid[0]*(1+1e-8))),
        minimum_resource_margin=float((teacher.resources(k, states)-current).min()))


def simulate(teacher, count=10000, seed=71003):
    rng = np.random.default_rng(seed)
    capital, state = teacher.parameters['kss'], 3
    capitals, consumptions, state_counts = [], [], np.zeros(7, dtype=int)
    for _ in range(count):
        c = float(teacher.consumption(capital, state))
        require(c > 0., 'Simulation consumption is infeasible')
        capital = float(teacher.next_capital(capital, state))
        require(teacher.k_grid[0] < capital < teacher.k_grid[-1], 'Simulation reached capital boundary')
        capitals.append(capital); consumptions.append(c); state_counts[state] += 1
        state = int(rng.choice(7, p=teacher.P[state]))
    return dict(steps=count, seed=seed, initial_capital_over_kss=1., initial_productivity_index=3,
        minimum_capital_over_kss=float(min(capitals)/teacher.parameters['kss']),
        maximum_capital_over_kss=float(max(capitals)/teacher.parameters['kss']),
        minimum_consumption=float(min(consumptions)), state_counts=state_counts.tolist(),
        boundary_hits=0, feasible=True)


def sample_states(teacher, count, seed):
    rng = np.random.default_rng(seed)
    k = rng.uniform(.2, 2.5, count)*teacher.parameters['kss']
    states = rng.choice(7, size=count, p=teacher.stationary)
    return k, states


def certify_nonlinearity(teacher):
    train_k, train_states = sample_states(teacher, 8192, 71004)
    test_k, test_states = sample_states(teacher, 10000, 71005)
    train_y, test_y = teacher.consumption(train_k, train_states), teacher.consumption(test_k, test_states)
    results = {}
    for kind in ['level_ols', 'log_ols']:
        train_capital = train_k if kind == 'level_ols' else np.log(train_k)
        test_capital = test_k if kind == 'level_ols' else np.log(test_k)
        train_design = np.column_stack([np.ones(len(train_k)), train_capital, teacher.logz[train_states]])
        test_design = np.column_stack([np.ones(len(test_k)), test_capital, teacher.logz[test_states]])
        target = train_y if kind == 'level_ols' else np.log(train_y)
        coefficients = np.linalg.lstsq(train_design, target, rcond=None)[0]
        prediction = test_design@coefficients
        if kind == 'log_ols':
            prediction = np.exp(prediction)
        rms = float(np.sqrt(np.mean((prediction-test_y)**2)))
        results[kind] = dict(coefficients=coefficients.tolist(), test_rmse=rms,
            test_rmse_normalized_by_consumption_sd=rms/float(test_y.std()),
            test_r2=float(1-np.mean((prediction-test_y)**2)/np.var(test_y)),
            test_max_absolute_error=float(np.max(abs(prediction-test_y))))
    kss = teacher.parameters['kss']; step = .01*kss
    c_minus, c_zero, c_plus = [float(teacher.consumption(kss+offset*step, 3)) for offset in [-1, 0, 1]]
    curvature = (c_plus-2*c_zero+c_minus)/step**2
    return dict(train_samples=8192, train_seed=71004, validation_samples=10000, validation_seed=71005,
        sample_domain='Capital uniform [0.2,2.5]*kss; productivity drawn from the exact stationary seven-state Markov distribution.',
        baselines=results, curvature=dict(capital=kss, productivity_index=3, step=step,
            second_derivative=curvature, dimensionless_second_derivative=curvature*kss**2/c_zero,
            consumption_minus=c_minus, consumption_center=c_zero, consumption_plus=c_plus),
        validation_consumption_sd=float(test_y.std()))


def main():
    started = time.perf_counter()
    deadline = started+60.
    coarse, coarse_stats = solve_teacher(1001, deadline)
    fine, fine_stats = solve_teacher(2001, deadline, initial=coarse)
    k, states = sample_states(fine, 10000, 71001)
    diagnostics = euler_diagnostics(fine, k, states)
    relative_difference = (fine.consumption(k, states)-coarse.consumption(k, states))/fine.consumption(k, states)
    absolute_difference = fine.consumption(k, states)-coarse.consumption(k, states)
    fine_consumption_sd = float(fine.consumption(k, states).std())
    refinement = dict(samples=10000, seed=71001,
        consumption_relative_difference_max=float(np.max(abs(relative_difference))),
        consumption_relative_difference_rms=float(np.sqrt(np.mean(relative_difference**2))),
        consumption_sd=fine_consumption_sd,
        consumption_difference_rms_normalized_by_c_sd=float(np.sqrt(np.mean(absolute_difference**2))/fine_consumption_sd),
        consumption_difference_max_normalized_by_c_sd=float(np.max(abs(absolute_difference))/fine_consumption_sd))
    simulation = simulate(fine)
    nonlinearity = certify_nonlinearity(fine)
    gate = min(.01, min(value['test_rmse_normalized_by_consumption_sd']
                        for value in nonlinearity['baselines'].values())/3)
    teacher_gate_pass = max(diagnostics['consumption_equivalent_error_max_normalized_by_c_sd'],
                            refinement['consumption_difference_max_normalized_by_c_sd']) < gate/10
    passed = diagnostics['consumption_equivalent_relative_error_max'] < 1e-4 and refinement['consumption_relative_difference_max'] < 1e-4
    report = dict(parameters=PARAMETERS, model=dict(name='Nonlinear stochastic growth with fixed unit labor endowment',
        utility='c^(1-sigma)/(1-sigma)', resource_constraint="c+k_next=z*k^alpha+(1-delta)*k",
        competitive_prices='r=alpha*z*k^(alpha-1), w=(1-alpha)*z*k^alpha, labor=1',
        euler_equation='c^(-sigma)=beta*sum_zprime P[z,zprime]*c(k_next,zprime)^(-sigma)*(alpha*zprime*k_next^(alpha-1)+1-delta)',
        optimality='Strictly concave utility and production give the interior Euler condition and resource feasibility. With transversality, these characterize the optimal allocation. The numerical checks verify feasibility, Euler residuals, refinement, and a bounded simulation; they are not an analytic proof of interpolation accuracy or transversality over the unbounded state space.',
        nonlinear=True, perturbation=False, log_linearization=False, labor_endogenous=False,
        finite_markov_benchmark='Exact seven-state transition matrix; a Rouwenhorst approximation to continuous log productivity, not an exact solution of the continuous AR(1).'),
        markov=dict(method='Rouwenhorst', states=7, P=fine.P.tolist(), logz=fine.logz.tolist(),
            z=fine.z.tolist(), stationary=fine.stationary.tolist(),
            stationary_log_mean=float(fine.stationary@fine.logz),
            stationary_log_std=float(np.sqrt(fine.stationary@(fine.logz**2))),
            implied_first_order_log_autocorrelation=float(np.sum(fine.stationary[:, None]*fine.P*fine.logz[:, None]*fine.logz[None, :])/(fine.stationary@(fine.logz**2)))),
        solver=dict(method='Endogenous-grid Euler time iteration with 42-step vector bisection of nonlinear resources and piecewise-linear policy interpolation.',
            coarse=coarse_stats, fine=fine_stats, convergence_relative_tolerance=2e-11,
            sources=['https://python.quantecon.org/os_egm.html', 'https://dynamics.quantecon.org/egm_policy_iter.html']),
        validation=dict(seed=71001, domain='10,000 fresh capital draws uniform[0.2,2.5]*kss and independent stationary productivity draws.',
            euler=diagnostics, refinement=refinement, simulation=simulation,
            tolerance_target=1e-4, passed_target=passed,
            pilot_network_gate_normalized_by_c_sd=gate,
            teacher_error_budget_normalized_by_c_sd=gate/10,
            teacher_error_below_pilot_network_gate_tenth=teacher_gate_pass,
            euler_error_definition='abs((beta*E[c_next^(-sigma)*gross_return])^(-1/sigma)/c_current-1), evaluated at actual next capital from the interpolated policy.'),
        nonlinearity=nonlinearity,
        limitations=['Policy is numerically solved and interpolated, not an analytic exact solution.',
            'Certification covers the declared core domain, the finite productivity chain, and one fixed simulation.',
            'The numerical capital floor/ceiling bound the solver. No extrapolation is allowed by the API.',
            'The fixed labor endowment removes an endogenous labor decision; wages support the same allocation.',
            'The deterministic steady capital is a reference scale, not the stochastic steady-state mean.'],
        runtime_seconds=time.perf_counter()-started, numpy_version=np.__version__)
    np.savez_compressed(ROOT/'nonlinear_teacher.npz', k_grid=fine.k_grid,
        consumption_grid=fine.consumption_grid, next_capital_grid=np.vstack([fine.next_capital(fine.k_grid, j) for j in range(7)]),
        P=fine.P, logz=fine.logz, z=fine.z, stationary=fine.stationary,
        parameters_json=np.array(json.dumps(PARAMETERS)), **{key:np.array(value) for key,value in PARAMETERS.items()})
    report['artifact_sha256'] = hashlib.sha256((ROOT/'nonlinear_teacher.npz').read_bytes()).hexdigest()
    (ROOT/'nonlinear_teacher.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(runtime_seconds=report['runtime_seconds'], validation=report['validation'],
        nonlinearity=nonlinearity), indent=2))


if __name__ == '__main__':
    main()
