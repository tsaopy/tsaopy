import numpy as np


# numerical solver
def rk4(f, t, x, dt):
    k1 = dt*f(t, x)
    k2 = dt*f(t+0.5*dt, x+0.5*k1)
    k3 = dt*f(t+0.5*dt, x+0.5*k2)
    k4 = dt*f(t+dt, x+k3)
    return x + (k1+k4+2*k2+2*k3)/6


def solve_ivp(f, x0, t0tf, dt):
    t, tf = t0tf
    x, v = x0
    result = [[t, x, v]]
    while t < tf:
        x, v = rk4(f, t, (x, v), dt)
        t = t + dt
        result.append([t, x, v])
    return np.array(result)


# simulation
a1, b1 = .3, 1.0
def pF(t, params):
    A, w, phi = params
    def loc_f(t_loc, n):
        return A/n*np.sin(n*(w*t_loc+phi))
    return loc_f(t, 1) + loc_f(t, 3) + loc_f(t, 5)

F = lambda t: pF(t, [1.0, 1.0, .0])

deriv = lambda t, X: np.array([X[1],
                               F(t)-a1*X[1]-b1*X[0]])

X0 = np.array([1.0,0.0])

result = solve_ivp(deriv, X0, (0, 15.0), .01)

# proc data
t,x,v = result[:,0],result[:,1],result[:,2]

datalen, n_noise, u_noise = len(t), .5, .5
np.random.seed(2345)

x += np.random.uniform(-u_noise, u_noise, datalen) \
    + np.random.normal(.0, n_noise, datalen)
    
v += 0.7 * np.random.uniform(-u_noise, u_noise, datalen) \
    + 0.7 * np.random.normal(.0, n_noise, datalen)

# fitting
from tsaopy import models, events
import quickemcee as qmc

event1 = events.Event({'x0':qmc.utils.normal_prior(1., 1.),
                       'v0':qmc.utils.normal_prior(.0, 5.)},
                      t, x, .7, v, .7)


tsaopymodel = models.BaseModel({'a':[(1, qmc.utils.normal_prior(.0, 5.))],
                                'b':[(1, qmc.utils.normal_prior(.0, 5.))],
                                'f':[(1, qmc.utils.normal_prior(.0, 100.)),
                                     (2, qmc.utils.uniform_prior(.0, 20.)),
                                     (3, qmc.utils.uniform_prior(-np.pi, np.pi))]},
                               [event1],
                               pF)

from scipy.optimize import dual_annealing

def neg_ll(coords):
    return -tsaopymodel._log_likelihood(coords)

bounds = [(0, 100.) for _ in range(tsaopymodel.ndim)]
bounds[4] = (-np.pi, np.pi)

sol = dual_annealing(neg_ll, bounds)

x0 = sol.x

mcmcmodel = tsaopymodel.setup_mcmc_model()
sampler = mcmcmodel.run_chain(300, 200, 500,
                              init_x=x0, workers=10)

flat_samples, samples = sampler.get_chain(flat=True), sampler.get_chain()
labels = tsaopymodel.paramslabels

# traceplots
qmc.utils.traceplots(samples=samples, labels_list=labels)

# cornerplots
qmc.utils.cornerplots(flat_samples=flat_samples, labels_list=labels)

# autocplots
qmc.utils.autocplots(flat_samples=flat_samples, labels_list=labels)

# results
predfx = lambda coords: tsaopymodel.event_predict(1, coords) [:,0]
predfv = lambda coords: tsaopymodel.event_predict(1, coords) [:,1]

qmc.utils.resultplot(flat_samples, x, t, predfx,
                     plotsamples=100)

qmc.utils.resultplot(flat_samples, x, t, predfx,
                     plotmode=True)

qmc.utils.resultplot(flat_samples, v, t, predfv,
                     plotsamples=100)

qmc.utils.resultplot(flat_samples, v, t, predfv,
                     plotmode=True)
