##############################################################################
#                                 PRELIMINARY
##############################################################################
# def functions
import numpy as np

# import data

data_t,data_x = np.loadtxt('testdata.txt',usecols=0),np.loadtxt(
                                                    'testdata.txt',usecols=1)
dt,t0,tf,N_data = data_t[1]-data_t[0],data_t[0],data_t[-1],len(data_x)


### acá ajustar para suavizar el valor inicial es problemático

# # fiteo polinomico
# from numpy.polynomial import Polynomial
# data_x_polyfit = Polynomial.fit(data_t[:10],data_x[:10], deg = 3)
# x0_ini = data_x_polyfit(t0)

# splines cubicos
#from scipy.interpolate import interp1d
#f_x0 = interp1d(data_t[:10],data_x[:10],kind='cubic')
#x0_ini = f_x0(t0)

# # savgol filtro
from scipy.signal import savgol_filter
data_x_savgol =  savgol_filter(data_x[:15],window_length=15,polyorder=2)
x0_ini = data_x_savgol[0]

def wrapper_aux_function(p_m):
    return np.hstack(p_m),[len(_) for _ in p_m[2:]]

def dewrap_aux_function(array_data_,wrap_info_):
    current_index,dewrapped_array = 0,[]
    for id_ in range(len(wrap_info_)):
        dewrapped_array.append(array_data_[current_index:(current_index+
                                                         wrap_info_[id_])])
        current_index = current_index+wrap_info_[id_]
    return dewrapped_array

w0sq_guess = 1.0
# priors : [ x0,v0, [a coefs], [b coefs] ]
prior = [x0_ini,0.0,[0.0],[w0sq_guess,0.0,0.0]]
prior_bounds = [#x0
                (0.8*x0_ini,1.2*x0_ini),
                #v0
                (-0.2,0.2),
                #disipacion
                (-0.1,0.1),
                # potencial
                (0,2*w0sq_guess),
                (-0.1,0.1),
                (-0.9,0.9)]

priors_wraped,main_wrap_info = wrapper_aux_function(prior)
ndim=len(priors_wraped)

import oscadsf2py

def simulation(_coefs_):
    x0_simu,v0_simu = _coefs_[0],_coefs_[1]
    _coefs_aux = dewrap_aux_function(_coefs_[2:],main_wrap_info)
    na,nb = main_wrap_info
    A_simu,B_simu = _coefs_aux[0],_coefs_aux[1]
    return oscadsf2py.simulation([x0_simu,v0_simu],A_simu,B_simu,dt,N_data,na,nb)

##############################################################################
#                                 emcee setup
##############################################################################

data_sigma = 2e-2
def log_likelihood(_coefs_):
    prediction = simulation(_coefs_)
    ll=-0.5*np.sum(((prediction-data_x)/data_sigma)**2)
    return (ll)

def coef_test(_x,_n):
    if _x<prior_bounds[_n][1] and _x>prior_bounds[_n][0]:
        return True
    else:
        return False

def log_prior(_coefs_):
    for lp_n in range(ndim):
        if not coef_test(_coefs_[lp_n],lp_n):
            return -np.inf
            break
    else:
        return 0.0

def log_probability(theta):
    lp=log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + log_likelihood(theta))

from scipy.optimize import minimize
neg_ll = lambda *args: -log_likelihood(*args)

### scipy optimize
# CG
#soln = minimize(neg_ll, initial,method='CG',options={'maxiter':200})
# nelder-mead
soln = minimize(neg_ll, priors_wraped,method='Powell',bounds=prior_bounds
                ,options={'maxiter':300})

initial = soln.x
            
n_iter=10000
n_iter_burn = n_iter//10
n_walkers=200

p0=[initial + 1e-7 * np.random.randn(ndim) for i in range(n_walkers)]
            
import emcee
from multiprocessing import Pool,cpu_count

def main(p0,n_walkers,n_iter,ndim,log_probability):
    with Pool(processes=(cpu_count()-2)) as pool:
        sampler = emcee.EnsembleSampler(n_walkers,ndim, log_probability,
                                        pool=pool)
    
        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, n_iter_burn, progress=True)
        sampler.reset()
    
        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, n_iter, progress=True)
    
        return sampler, pos, prob, state

##############################################################################
#                               do the run
##############################################################################

import matplotlib.pyplot as plt
from time import time

start=time()

sampler, pos, prob, state = main(p0,n_walkers,n_iter,ndim,log_probability)

##############################################################################
#                             process results
##############################################################################

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = [ 'x0','v0']  + \
            ['a'+str(_xd+1) for _xd in range(main_wrap_info[0])] + \
            ['b'+str(_xd+1) for _xd in range(main_wrap_info[1])]
            
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

autoc_tau = int(max(sampler.get_autocorr_time()))
flat_samples = sampler.get_chain(flat=True, thin=autoc_tau)
sample_truths = [np.mean(flat_samples[:,_]) for _ in range(ndim)]
import corner
fig = corner.corner(
    flat_samples, labels=labels, quantiles=(0.16, 0.84),show_titles=True,
    title_fmt='.3g', truths=sample_truths,
    truth_color='tab:red'
);
    
finish=time()

print ( '' )
print ( '--------------------------------------------------------------------------------')
print ( 'Execution finished. Time elapsed:',(finish-start), 'sec.')
print('')
