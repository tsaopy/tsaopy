#######################################################################
#                           setup preliminar
#######################################################################

# ### imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
from time import time

import emcee
import corner

from oscadsf2py import simulation

# ### datos de inicializacion
# importar serie temporal
filename = 'osc05C0-03_processed_trans_sigma50.txt'
data_t, data_x = np.loadtxt(filename, usecols=0),\
    np.loadtxt(filename, usecols=1)
# obtener parametros basicos
t_step, t0, tf, N_data, x0_ini = data_t[1]-data_t[0], data_t[0],\
                         data_t[-1], len(data_x), data_x[0]
# paso de integracion
dt = t_step/10.0

# ### priors
# estimacion del valor omega0 al cuadrado
w0sq_guess = 1.0
# construccion del array de priors
# IMPORTANTE : las dimensiones del array de priors define posteriormente
# la cantidad de terminos de la ec de mov que considera emcee
prior = [x0_ini, 0.0, [0.0], [w0sq_guess, 0.0, 0.0]]
# estimacion del rango de variabilidad de los parametros
prior_bounds = [  # x0
                (0.5*x0_ini, 1.5*x0_ini),
                # v0
                (-0.2, 0.2),
                # disipacion
                (-0.1, 0.1),
                # potencial
                (0, 2*w0sq_guess),
                (-0.1, 0.1),
                (-0.9, 0.9)]


# ### funciones auxiliares
# tomar array de arrays y aplanarlo en array 1D
# [A, B] --> [A[0],A[1],...,A[-1],B[0],...]
# y devolver un array con 'info' sobre la forma de los arrays
# saltea los primeros 2 elem destinados a x0 y v0
def wrapper_aux_function(p_m):
    return np.hstack(p_m), [len(_) for _ in p_m[2:]]


# tomar el array aplanado + la info y reconstruir el array original
def dewrap_aux_function(array_elem_, wrap_info_):
    current_index, dewrapped_array = 0, []
    for id_ in range(len(wrap_info_)):
        dewrapped_array.append(array_elem_[current_index:(current_index
                               + wrap_info_[id_])])
        current_index = current_index + wrap_info_[id_]
    return dewrapped_array


# wrapear los priors anteriores
priors_wraped, main_wrap_info = wrapper_aux_function(prior)
na, nb = main_wrap_info
# definir la dimension del espacio de parametros de emcee
ndim = len(priors_wraped)

#######################################################################
#                           setup de emcee
#######################################################################


# definicion del modelo
def model(_coefs_):
    x0_simu, v0_simu = _coefs_[0], _coefs_[1]
    _coefs_aux = dewrap_aux_function(_coefs_[2:], main_wrap_info)
    A_simu, B_simu = _coefs_aux[0], _coefs_aux[1]
    return simulation([x0_simu, v0_simu], A_simu, B_simu, dt, 10*N_data,
                      na, nb)[::10]


# incerteza 1 sigma sobre los datos x importados
data_sigma = 2e-2


# definicion de la func likelihood
def log_likelihood(_coefs_):
    prediction = model(_coefs_)
    ll = -0.5*np.sum(((prediction-data_x)/data_sigma)**2)
    return (ll)


# testeo para q cada parametro del modelo caiga dentro del rango acept
def coef_test(_x, _n):
    if _x < prior_bounds[_n][1] and _x > prior_bounds[_n][0]:
        return True
    else:
        return False


# log prior segun la docu de emcee, si todos los params pasan el test
# de las priors -> devolver 0, si no -inf
def log_prior(_coefs_):
    for lp_n in range(ndim):
        if not coef_test(_coefs_[lp_n], lp_n):
            return -np.inf
            break
    else:
        return 0.0


# log prob segun la docu de emcee
def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + log_likelihood(theta))


# # ###### OPCIONAL EN ALGUNOS CASOS ES MEJOR SACARLO
# # ### busqueda de los coefs iniciales para iniciar la cadena mcmc
# # definir la funcion a minimizar
# def neg_ll(*args):
#     return -log_likelihood(*args)


# # optimizar por metodo con cotas, usando los priors
# soln = minimize(neg_ll, priors_wraped, method='Powell',
#                 bounds=prior_bounds, options={'maxiter': 300})

# initial = soln.x  # si se saca la opt x scipy dejar linea de abajo
initial = priors_wraped

# parametros de la cadena emcee
n_iter = 10000
n_iter_burn = n_iter//10
n_walkers = 50
p0 = [initial + 1e-7 * np.random.randn(ndim) for i in range(n_walkers)]


# setear el main de emcee
def main(p0, n_walkers, n_iter, ndim, log_probability):
    with Pool(processes=(cpu_count()-2)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim,
                                        log_probability, pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, n_iter_burn, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, n_iter, progress=True)

        return sampler, pos, prob, state


#######################################################################
#                           correr el programa
#######################################################################

start = time()
sampler, pos, prob, state = main(p0, n_walkers, n_iter, ndim,
                                 log_probability)

#######################################################################
#               extraer resultados del sampler y procesarlos
#######################################################################

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
label_list = ['x0', 'v0'] +\
            ['a'+str(_xd+1) for _xd in range(main_wrap_info[0])] + \
            ['b'+str(_xd+1) for _xd in range(main_wrap_info[1])]

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(label_list[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

autoc_tau = int(max(sampler.get_autocorr_time()))
flat_samples = sampler.get_chain(flat=True, thin=autoc_tau)
sample_truths = [np.mean(flat_samples[:, _]) for _ in range(ndim)]

fig = corner.corner(flat_samples, labels=label_list,
                    quantiles=(0.16, 0.84), show_titles=True,
                    title_fmt='.3g', truths=sample_truths,
                    truth_color='tab:red')

finish = time()

print('')
print('---------------------------------------------------------------')
print('Execution finished. Time elapsed: {:.3f} sec.'.format(finish -
                                                             start))
print('')
