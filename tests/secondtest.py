import quickemcee as qmc
import numpy as np
import tsaopy
from scipy.optimize import dual_annealing


# numerical solver
def rk4(f, x, dx):
    """Do."""
    k1 = dx*f(x)
    k2 = dx*f(x+0.5*k1)
    k3 = dx*f(x+0.5*k2)
    k4 = dx*f(x+k3)
    return x + (k1+k4+2*k2+2*k3)/6


def solve_ivp(f, x0, t0tf, dt):
    """Do."""
    P = x0
    t, tf = t0tf
    x, v = P
    result = [[t, x, v]]
    while t < tf:
        P = rk4(f, P, dt)
        t = t + dt
        x, v = P
        result.append([t, x, v])
    return np.array(result)


# simulation
a1, b1 = .3, 1.0
deriv = lambda X: np.array([X[1], -a1*X[1]-b1*X[0]])

# # # events
# set up event 1
np.random.seed(3354)

ivpsol = solve_ivp(deriv, (1.0, .0), (.0, 30.0), 0.1)

t1, x1, v1 = ivpsol[:, 0], ivpsol[:, 1], ivpsol[:, 2]

x1 += np.random.normal(.0, 0.1, 301) + np.random.uniform(0.95, 1.05, 301) \
    + np.random.uniform(-.05, .05, 301)

v1 += np.random.normal(.0, 0.1, 301) + np.random.uniform(-.03, .03, 301)

event1_params = {'x0': (1.0, qmc.utils.normal_prior(1.0, 5.0)),
                 'v0': (0.0, qmc.utils.normal_prior(0.0, 5.0)),
                 'ep': (0.0, qmc.utils.normal_prior(0.0, 5.0))
                 }

event1 = tsaopy.events.Event(params=event1_params, t_data=t1, x_data=x1,
                             x_sigma=.15, v_data=v1, v_sigma=.1)

# set up event 2
ivpsol = solve_ivp(deriv, (-1.0, .5), (.0, 20.0), 0.1)

t2, x2, v2 = ivpsol[:, 0], ivpsol[:, 1], ivpsol[:, 2]

x2 += np.random.normal(.0, 0.1, 201) + np.random.uniform(-.05, .05, 201)

v2 += np.random.normal(.0, 0.1, 201) + np.random.uniform(-.03, .03, 201)

event2_params = {'x0': (1.0, qmc.utils.normal_prior(1.0, 5.0)),
                 'v0': (0.0, qmc.utils.normal_prior(0.0, 5.0))
                 }

event2 = tsaopy.events.Event(params=event2_params, t_data=t2, x_data=x2,
                             x_sigma=.15, v_data=v2, v_sigma=.1)

# set up tsaopy model

ode_coefs = {'a': [(1, .0, qmc.utils.normal_prior(0.0, 5.0))],
             'b': [(1, .0, qmc.utils.normal_prior(0.0, 5.0))]}

tsaopymodel = tsaopy.models.Model(ode_coefs, [event1, event2])

# do mcmc

neg_ll = lambda coords : -tsaopymodel._log_likelihood(coords)
sol = dual_annealing(neg_ll, [(-10, 10) for _ in range(7)])

mcmcmodel = tsaopymodel.setup_mcmc_model()

sampler = mcmcmodel.run_chain(100, 500, 500,
                              init_x=sol.x, workers=10)

flat_samples, samples = sampler.get_chain(flat=True), sampler.get_chain()

labels = tsaopymodel.paramslabels

# traceplots
qmc.utils.traceplots(samples=samples, labels_list=labels)

# cornerplots
qmc.utils.cornerplots(flat_samples=flat_samples, labels_list=labels)

# autocplots
qmc.utils.autocplots(flat_samples=flat_samples, labels_list=labels)

# results
predf1x = lambda coords: tsaopymodel.event_predict(1, coords)[:, 0]
predf1v = lambda coords: tsaopymodel.event_predict(1, coords)[:, 1]

qmc.utils.resultplot(flat_samples, x1, t1, predf1x,
                     plotsamples=50)

qmc.utils.resultplot(flat_samples, x1, t1, predf1x,
                     plotsamples=50, plotmode=True)

qmc.utils.resultplot(flat_samples, v1, t1, predf1v,
                     plotsamples=50)

qmc.utils.resultplot(flat_samples, v1, t1, predf1v,
                     plotsamples=50, plotmode=True)

predf2x = lambda coords: tsaopymodel.event_predict(2, coords)[:, 0]
predf2v = lambda coords: tsaopymodel.event_predict(2, coords)[:, 1]

qmc.utils.resultplot(flat_samples, x2, t2, predf2x,
                     plotsamples=50)

qmc.utils.resultplot(flat_samples, x2, t2, predf2x,
                     plotsamples=50, plotmode=True)

qmc.utils.resultplot(flat_samples, v2, t2, predf2v,
                     plotsamples=50)

qmc.utils.resultplot(flat_samples, v2, t2, predf2v,
                     plotsamples=50, plotmode=True)
