import numpy as np
import quickemcee as qmc
import tsaopy
from scipy.optimize import minimize

# make data
np.random.seed(1020)
n_noise, u_noise = .1, .3
ndata = 301
t_data = np.linspace(0, 30, ndata)
x_data = np.cos(t_data)*np.exp(-.1*t_data)
v_data = np.gradient(x_data, t_data)
x_data += (abs(x_data)*np.random.uniform(-u_noise, u_noise) +
           np.random.normal(0, n_noise, ndata))
v_data += (abs(v_data)*np.random.uniform(-u_noise, u_noise) +
           np.random.normal(0, n_noise, ndata))*.7

# new solution
ev1_params = {'x0': qmc.utils.normal_prior(1, 5),
              'v0': qmc.utils.normal_prior(0, 5)}


# custom log likelihood
def base_ctom_ll(pred, data, sigma, log_f):
    s2 = sigma**2 + data**2 * np.exp(2*log_f)
    return -.5 * np.sum((data - pred) ** 2 / s2 + np.log(s2))


def custom_ll(_self, pred, ll_vals):
    log_fx, log_fv = ll_vals
    predx, predv = pred
    return (base_ctom_ll(predx, _self.x_data, _self.x_sigma, log_fx)
            + base_ctom_ll(predv, _self.v_data, _self.v_sigma, log_fv))


ctom_llparams = {'log_fx': qmc.utils.uniform_prior(-10, 1),
                 'log_fv': qmc.utils.uniform_prior(-20, 5)}

events = [tsaopy.events.Event(ev1_params, t_data, x_data, .05, v_data, .05,
                              custom_ll, ctom_llparams)]

ode_coefs = {'a': [(1, qmc.utils.normal_prior(0, 5))],
             'b': [(1, qmc.utils.normal_prior(0, 5))]}

mymodel = tsaopy.models.BaseModel(ode_coefs, events)
mysampler = mymodel.setup_mcmc_model()
ini_vals = (0, 0, 1, 0, -1, -1)
faux = mymodel.neg_ll
ini_vals = minimize(faux, ini_vals).x
emcee_ensemble = mysampler.run_chain(100, 300, 200,
                                     init_x=ini_vals)

labels = mymodel.paramslabels
flat_chain = emcee_ensemble.get_chain(flat=True)

# 
qmc.utils.cornerplots(flat_chain, labels)
