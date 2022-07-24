import quickemcee as qmc
import numpy as np
import tsaopy
from scipy.optimize import minimize

# # # events
# set up event 1

np.random.seed(3354)

t1 = np.linspace(0, 10, 101)
x1 = -.3 * t1 + 10 + np.random.normal(.0, .5, 101)

event1_params = {'x0': qmc.utils.normal_prior(10.0, 5.0),
                 'v0': qmc.utils.normal_prior(0.0, 5.0)
                 }

event1 = tsaopy.events.Event(event1_params, t1, x1, 1.5)

# set up event 2

t2 = np.linspace(0, 10, 101)
x2 = .5 * t2 + 5 + np.random.normal(.0, .5, 101)

event2_params = {'x0': qmc.utils.normal_prior(5.0, 5.0),
                 'v0': qmc.utils.normal_prior(0.0, 5.0)
                 }

event2 = tsaopy.events.Event(event2_params, t2, x2, 1.5)

# set up tsaopy model

ode_coefs = {'a': [(1, qmc.utils.normal_prior(0.0, 5.0))],
             'b': [(1, qmc.utils.normal_prior(0.0, 5.0))]}

tsaopymodel = tsaopy.models.Model(ode_coefs, [event1, event2])

# do mcmc

neg_ll = lambda coords : -tsaopymodel._log_likelihood(coords)

sol = minimize(neg_ll, np.zeros(6))

mcmcmodel = tsaopymodel.setup_mcmc_model()

sampler = mcmcmodel.run_chain(100, 2000, 2000,
                              init_x=sol.x, workers=1)

flat_samples, samples = sampler.get_chain(flat=True), sampler.get_chain()

labels = tsaopymodel.paramslabels

# traceplots
qmc.utils.traceplots(samples=samples, labels_list=labels)

# cornerplots
qmc.utils.cornerplots(flat_samples=flat_samples, labels_list=labels)

# autocplots
qmc.utils.autocplots(flat_samples=flat_samples, labels_list=labels)

# results
predf1 = lambda coords: tsaopymodel.event_predict(1, coords)
predf2 = lambda coords: tsaopymodel.event_predict(2, coords)

qmc.utils.resultplot(flat_samples, x1, t1, predf1,
                     plotsamples=100)

qmc.utils.resultplot(flat_samples, x1, t1, predf1,
                     plotmode=True)

qmc.utils.resultplot(flat_samples, x2, t2, predf2,
                     plotsamples=100)

qmc.utils.resultplot(flat_samples, x2, t2, predf2,
                     plotmode=True)
