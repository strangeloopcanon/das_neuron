"""RBC numerical teacher with labor and convex capital installation costs.

Run this file to solve, check and save. Import has no solve/write side effects.
This verifies first-order equations, not global Bellman optimality.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from nonlinear_teacher import rouwenhorst

ROOT = Path(__file__).resolve().parent
PARAMETERS = dict(alpha=.36,beta=.96,delta=.08,sigma=2.,phi=1.,kappa=2.,
    rho=.9,innovation_std=.03,labor_rho=.7,labor_innovation_std=.04,nss=.33)
p = PARAMETERS
p['kss'] = p['nss']*(p['alpha']/(1/p['beta']-1+p['delta']))**(1/(1-p['alpha']))
p['yss'] = p['kss']**p['alpha']*p['nss']**(1-p['alpha'])
p['css'] = p['yss']-p['delta']*p['kss']
p['chi'] = (1-p['alpha'])*p['yss']/(p['nss']**(1+p['phi'])*p['css']**p['sigma'])


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def markov_chain():
    A, az, _, ast = rouwenhorst(7, PARAMETERS['rho'], PARAMETERS['innovation_std'])
    B, eta, _, bst = rouwenhorst(3, PARAMETERS['labor_rho'], PARAMETERS['labor_innovation_std'])
    features = np.array([(x,e) for x in az for e in eta])
    return np.kron(A,B), features, np.kron(ast,bst)


def static_policy(k, state, kp, features, parameters):
    """Unique static labor/resource root conditional on next capital.

    Output rises with positive labor; consumption from labor optimality falls.
    Labor has no artificial hours cap. Investment may be negative.
    """
    p = parameters
    k,state,kp = np.broadcast_arrays(np.asarray(k,float),np.asarray(state,int),np.asarray(kp,float))
    z = np.exp(features[state,0]); eta = features[state,1]
    investment = kp-(1-p['delta'])*k
    installation = .5*p['kappa']*k*(investment/k-p['delta'])**2
    expense = investment+installation
    B = z*k**p['alpha']
    D = ((1-p['alpha'])*B/(p['chi']*np.exp(eta)))**(1/p['sigma'])
    exponent = (p['alpha']+p['phi'])/p['sigma']
    low = np.full(k.shape,1e-10); high = np.full(k.shape,32.)
    n = np.full(k.shape,p['nss'])
    for _ in range(20):
        y = B*n**(1-p['alpha']); c = D*n**(-exponent)
        error = y-c-expense
        high = np.where(error > 0.,n,high); low = np.where(error > 0.,low,n)
        derivative = ((1-p['alpha'])*y+exponent*c)/n
        proposal = n-error/derivative
        candidate = np.where((proposal > low)&(proposal < high),proposal,.5*(low+high))
        n = np.where(abs(error) < 2e-14*np.maximum(1.,abs(expense)),n,candidate)
    y = B*n**(1-p['alpha']); c = y-expense
    q = 1+p['kappa']*(investment/k-p['delta'])
    return dict(c=c,labor=n,investment=investment,next_capital=kp,q=q,output=y)


class RicherTeacher:
    def __init__(self,k_grid,next_capital_grid,P,state_features,stationary,parameters=None):
        self.k_grid = np.asarray(k_grid); self.next_capital_grid = np.asarray(next_capital_grid)
        self.P = np.asarray(P); self.state_features = np.asarray(state_features)
        self.logz = self.state_features[:,0]; self.z = np.exp(self.logz)
        self.eta = self.state_features[:,1]; self.stationary = np.asarray(stationary)
        self.parameters = dict(PARAMETERS if parameters is None else parameters)

    def policy(self,k,state):
        k,state = np.broadcast_arrays(np.asarray(k,float),np.asarray(state,int))
        if np.any(k < self.k_grid[0]) or np.any(k > self.k_grid[-1]):
            raise ValueError('Capital outside solved grid; extrapolation forbidden')
        if np.any(state < 0) or np.any(state >= len(self.z)):
            raise ValueError('Shock index outside finite Markov state space')
        kp = np.empty(k.size)
        for j in range(len(self.z)):
            mask = state.ravel() == j
            kp[mask] = np.interp(k.ravel()[mask],self.k_grid,self.next_capital_grid[j])
        return static_policy(k,state,kp.reshape(k.shape),self.state_features,self.parameters)

    def sample_states(self,count,seed):
        rng = np.random.default_rng(seed)
        return (rng.uniform(.5,1.8,count)*self.parameters['kss'],
                rng.choice(len(self.z),count,p=self.stationary))


def load_teacher(path=None):
    with np.load(ROOT/'richer_teacher.npz' if path is None else path,allow_pickle=False) as saved:
        return RicherTeacher(saved['k_grid'],saved['next_capital_grid'],saved['P'],
            saved['state_features'],saved['stationary'],json.loads(str(saved['parameters_json'])))


_DEFAULT = None


def sample_states(count,seed,teacher=None):
    global _DEFAULT
    if teacher is None:
        if _DEFAULT is None:
            _DEFAULT = load_teacher()
        teacher = _DEFAULT
    return teacher.sample_states(count,seed)


def solve_teacher(points,deadline,initial=None):
    p = PARAMETERS
    P,features,stationary = markov_chain()
    grid = p['kss']*np.geomspace(.08,4.,points)
    k = np.broadcast_to(grid,(len(P),points))
    states = np.broadcast_to(np.arange(len(P))[:,None],k.shape)
    if initial is None:
        kp = np.clip(.95*k+.05*p['kss'],grid[0],grid[-1])
    else:
        kp = initial.policy(k,states)['next_capital']
    history = []; start = time.perf_counter()
    for iteration in range(1,2001):
        require(time.perf_counter() < deadline,'Numerical teacher exceeded bounded CPU budget')
        old = static_policy(k,states,kp,features,p)
        phi_k = .5*p['kappa']*(p['delta']**2-(old['investment']/k)**2)
        gross_return = p['alpha']*old['output']/k+(1-p['delta'])*old['q']-phi_k
        require(np.all(old['c'] > 0.) and np.all(gross_return > 0.),'Invalid marginal continuation value')
        M = p['beta']*(P@(old['c']**(-p['sigma'])*gross_return))
        # EGM: hold kprime fixed and invert current capital, preserving q>0.
        low = .02*k; high = (p['kappa']/(p['kappa']-1)-1e-9)*k
        for _ in range(40):
            mid = .5*(low+high)
            q = 1+p['kappa']*(k/mid-1)
            c = (q/M)**(1/p['sigma'])
            n = ((1-p['alpha'])*np.exp(features[:,0,None])*mid**p['alpha']/
                (p['chi']*np.exp(features[:,1,None])*c**p['sigma']))**(1/(p['phi']+p['alpha']))
            y = np.exp(features[:,0,None])*mid**p['alpha']*n**(1-p['alpha'])
            i = k-(1-p['delta'])*mid
            error = y-c-i-.5*p['kappa']*mid*(i/mid-p['delta'])**2
            high = np.where(error > 0.,mid,high); low = np.where(error > 0.,low,mid)
        endogenous = .5*(low+high)
        require(np.all(np.diff(endogenous,axis=1) > 0.),'Nonmonotone endogenous grid')
        updated = np.vstack([np.interp(grid,endogenous[j],grid) for j in range(len(P))])
        # Constant endpoint interpolation imposes finite numerical capital bounds.
        change = float(np.max(abs(updated-kp)/p['kss']))
        kp = updated
        if iteration % 50 == 0 or change < 2e-10:
            history.append(dict(iteration=iteration,maximum_absolute_kprime_change_over_kss=change))
        if change < 2e-10:
            break
    else:
        raise RuntimeError('Policy time iteration did not converge')
    return RicherTeacher(grid,kp,P,features,stationary,p),dict(points=points,
        iterations=iteration,seconds=time.perf_counter()-start,maximum_kprime_change_over_kss=change,
        history=history,capital_spacing='geometric',capital_domain_over_kss=[.08,4.])


def summarize(x):
    x = np.asarray(x)
    return dict(maximum=float(np.max(abs(x))),rms=float(np.sqrt(np.mean(x*x))),
        absolute_quantiles=np.quantile(abs(x),[.5,.9,.99,1.]).tolist())


def diagnostics(teacher,count=12000,seed=81003,domain=(.5,1.8)):
    p = teacher.parameters; rng = np.random.default_rng(seed)
    k = rng.uniform(*domain,count)*p['kss']; states = rng.integers(len(teacher.P),size=count)
    current = teacher.policy(k,states); kp = current['next_capital']
    require(np.all(kp > teacher.k_grid[0]) and np.all(kp < teacher.k_grid[-1]),'Fresh capital left interior domain')
    expected = np.zeros(count)
    for j in range(len(teacher.P)):
        future = teacher.policy(kp,j)
        phi_k = .5*p['kappa']*(p['delta']**2-(future['investment']/kp)**2)
        R = p['alpha']*future['output']/kp+(1-p['delta'])*future['q']-phi_k
        expected += teacher.P[states,j]*future['c']**(-p['sigma'])*R
    lhs = current['c']**(-p['sigma'])*current['q']; rhs = p['beta']*expected
    labor_lhs = p['chi']*np.exp(teacher.eta[states])*current['labor']**p['phi']
    labor_rhs = current['c']**(-p['sigma'])*(1-p['alpha'])*current['output']/current['labor']
    installation = .5*p['kappa']*k*(current['investment']/k-p['delta'])**2
    resource = current['c']+current['investment']+installation-current['output']
    capital = current['next_capital']-((1-p['delta'])*k+current['investment'])
    q = current['q']-(1+p['kappa']*(current['investment']/k-p['delta']))
    return dict(samples=count,seed=seed,capital_domain_over_kss=list(domain),
        shock_sampling='uniform across all 21 finite shock states',euler_relative=summarize(rhs/lhs-1),
        euler_consumption_equivalent_relative=summarize((rhs/lhs)**(-1/p['sigma'])-1),
        labor_relative=summarize(labor_rhs/labor_lhs-1),resource_absolute=summarize(resource),
        capital_absolute=summarize(capital),q_absolute=summarize(q),
        minimum_consumption=float(current['c'].min()),minimum_labor=float(current['labor'].min()),
        maximum_labor=float(current['labor'].max()),minimum_q=float(current['q'].min()),
        next_capital_domain_over_kss=[float(kp.min()/p['kss']),float(kp.max()/p['kss'])])


def simulation(teacher,steps=10000,seed=81004):
    rng = np.random.default_rng(seed); p = teacher.parameters
    k = np.full(100,p['kss']); states = rng.choice(len(teacher.P),100,p=teacher.stationary)
    min_k,max_k,min_c,min_n,max_n,min_q = np.inf,-np.inf,np.inf,np.inf,-np.inf,np.inf
    counts = np.zeros(len(teacher.P),int)
    for _ in range(steps//100):
        pol = teacher.policy(k,states)
        min_c = min(min_c,float(pol['c'].min())); min_n = min(min_n,float(pol['labor'].min()))
        max_n = max(max_n,float(pol['labor'].max())); min_q = min(min_q,float(pol['q'].min()))
        counts += np.bincount(states,minlength=len(counts))
        k = pol['next_capital']
        min_k = min(min_k,float(k.min())); max_k = max(max_k,float(k.max()))
        require(np.all(k > teacher.k_grid[0]) and np.all(k < teacher.k_grid[-1]),'Simulation hit numerical boundary')
        states = (rng.random(len(k))[:,None] > np.cumsum(teacher.P[states],axis=1)).sum(axis=1)
    require(min_c > 0. and min_n > 0. and min_q > 0.,'Simulation infeasible')
    return dict(paths=100,transitions_per_path=steps//100,total_transitions=steps,seed=seed,
        minimum_consumption=min_c,labor_range=[min_n,max_n],minimum_q=min_q,
        capital_range_over_kss=[min_k/p['kss'],max_k/p['kss']],state_counts=counts.tolist(),
        numerical_boundary_hits=0,feasible=True)


def structural_checks(teacher):
    """Independent finite differences and deterministic steady-state equations."""
    p = teacher.parameters; rng = np.random.default_rng(81007)
    k = rng.uniform(.5,1.8,1000)*p['kss']
    i = rng.uniform(-.03,.2,1000)*k
    h = 1e-5*k
    def cost(capital,investment):
        return .5*p['kappa']*capital*(investment/capital-p['delta'])**2
    fd_k = (cost(k+h,i)-cost(k-h,i))/(2*h)
    fd_i = (cost(k,i+h)-cost(k,i-h))/(2*h)
    analytic_k = .5*p['kappa']*(p['delta']**2-(i/k)**2)
    analytic_i = p['kappa']*(i/k-p['delta'])
    central_state = int(np.argmin(np.sum(teacher.state_features**2,axis=1)))
    ss = static_policy(p['kss'],central_state,p['kss'],teacher.state_features,p)
    ss_labor = p['chi']*p['nss']**p['phi']-p['css']**(-p['sigma'])*(1-p['alpha'])*p['yss']/p['nss']
    ss_euler = p['beta']*(p['alpha']*p['yss']/p['kss']+1-p['delta'])-1
    require(np.max(abs(fd_k-analytic_k)) < 1e-8,'Installation partial k derivative failed')
    require(np.max(abs(fd_i-analytic_i)) < 1e-8,'Installation partial i derivative failed')
    require(np.max(abs(teacher.P.sum(axis=1)-1)) < 1e-12,'Invalid product Markov row mass')
    require(np.max(abs(teacher.stationary@teacher.P-teacher.stationary)) < 1e-12,'Invalid stationary law')
    return dict(installation_partial_k_finite_difference_error=summarize(fd_k-analytic_k),
        installation_partial_i_finite_difference_error=summarize(fd_i-analytic_i),
        finite_difference_samples=1000,finite_difference_seed=81007,
        steady_state=dict(k=p['kss'],c=p['css'],labor=p['nss'],chi=p['chi'],
            static_policy_consumption_error=float(ss['c']-p['css']),
            static_policy_labor_error=float(ss['labor']-p['nss']),
            labor_equation_absolute_error=float(abs(ss_labor)),euler_relative_error=float(abs(ss_euler))),
        markov_maximum_row_mass_error=float(np.max(abs(teacher.P.sum(axis=1)-1))),
        markov_stationarity_maximum_error=float(np.max(abs(teacher.stationary@teacher.P-teacher.stationary))))


def main():
    started = time.perf_counter()
    coarse,coarse_info = solve_teacher(240,started+60.)
    pilot_seconds = time.perf_counter()-started
    print(json.dumps(dict(stage='measured_pilot',seconds=pilot_seconds,solver=coarse_info)),flush=True)
    require(pilot_seconds < 60.,'First measured pilot exceeded 60 seconds')
    fine,fine_info = solve_teacher(960,started+180.,initial=coarse)
    coarse_diag = diagnostics(coarse,12000,81002)
    fine_diag = diagnostics(fine,12000,81002)
    extended_diag = diagnostics(fine,6000,81005,(.25,2.5))
    k,states = fine.sample_states(12000,81006)
    a,b = coarse.policy(k,states),fine.policy(k,states)
    refinement = {name:dict(absolute=summarize(a[name]-b[name]),relative=summarize(a[name]/b[name]-1))
        for name in ('c','labor','investment','next_capital','q')}
    require(fine_diag['euler_relative']['maximum'] < .002,'Refined Euler residual too large')
    require(fine_diag['labor_relative']['maximum'] < 1e-5,'Labor residual too large')
    np.savez_compressed(ROOT/'richer_teacher.npz',k_grid=fine.k_grid,next_capital_grid=fine.next_capital_grid,
        P=fine.P,state_features=fine.state_features,stationary=fine.stationary,z=fine.z,logz=fine.logz,
        parameters_json=json.dumps(fine.parameters),coarse_k_grid=coarse.k_grid,
        coarse_next_capital_grid=coarse.next_capital_grid)
    report = dict(schema_version=1,status='numerically_solved_and_checked',parameters=PARAMETERS,
        model=dict(production='y=exp(logz)*k^alpha*n^(1-alpha)',
            utility='c^(1-sigma)/(1-sigma)-chi*exp(eta)*n^(1+phi)/(1+phi)',
            resource='c+i+(kappa/2)*k*(i/k-delta)^2=y',capital='kprime=(1-delta)*k+i',
            labor_foc='chi*exp(eta)*n^phi=c^-sigma*(1-alpha)*y/n',
            investment_foc='q=1+kappa*(i/k-delta)',
            euler='c^-sigma*q=beta*E[cprime^-sigma*(MPKprime+(1-delta)*qprime-Phi_k_prime)]',
            installation_partial_k_at_fixed_investment='Phi_k=(kappa/2)*(delta^2-(i/k)^2)',
            derivation='Envelope V_k=mu*(MPK-Phi_k)+beta*(1-delta)*E[V_k_prime]; investment FOC beta*E[V_k_prime]=mu*q.',
            constraints='k>0,c>0,n>0; labor has no hours cap; investment can be negative; q positive on audited domains'),
        numerical_scope=dict(method='endogenous-grid policy time iteration with intertemporal Markov expectations and nested current-capital root',
            qualification='Converged first-order equations with fresh residual and grid checks; global Bellman optimality was not established.',
            shocks='7 productivity x 3 labor-disutility Rouwenhorst product chain; 21 discrete states',
            core_sample_domain_over_kss=[.5,1.8],capital_grid_domain_over_kss=[.08,4.],
            boundary_rule='Numerical capital lower/upper constraints; no extrapolation; core diagnostics and simulation remain interior',
            interpolation='piecewise-linear next-capital policy; labor and resource equations solved at each query'),
        measured_pilot_seconds=pilot_seconds,coarse_solver=coarse_info,fine_solver=fine_info,
        coarse_diagnostics=coarse_diag,fine_diagnostics=fine_diag,extended_diagnostics=extended_diag,
        grid_refinement=refinement,simulation=simulation(fine),structural_checks=structural_checks(fine),
        provenance=dict(source='richer_teacher.py',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            artifact='richer_teacher.npz',artifact_sha256=hashlib.sha256((ROOT/'richer_teacher.npz').read_bytes()).hexdigest(),
            rouwenhorst_source='nonlinear_teacher.py',rouwenhorst_source_sha256=hashlib.sha256((ROOT/'nonlinear_teacher.py').read_bytes()).hexdigest(),
            numpy_version=np.__version__,execution='local NumPy CPU; no paid work, publication, or training'),
        total_seconds=time.perf_counter()-started)
    (ROOT/'richer_teacher.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(stage='complete',seconds=report['total_seconds'],fine_solver=fine_info,
        fine_diagnostics=fine_diag,simulation=report['simulation'])),flush=True)


if __name__ == '__main__':
    main()
