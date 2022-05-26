import numpy as np

# numerical solver
def rk4(f,x,dx):
    k1 = dx*f(x)
    k2 = dx*f(x+0.5*k1)
    k3 = dx*f(x+0.5*k2)
    k4 = dx*f(x+k3)
    return x + (k1+k4+2*k2+2*k3)/6

def solve_ivp(f,x0,t0tf,dt):
    P = x0
    t,tf = t0tf
    x,v = P
    result = [[t,x,v]]
    while t < tf:
        P = rk4(f,P,dt)
        t = t + dt
        x,v = P
        result.append([t,x,v])
    return np.array(result)

# aux noise function
np.random.seed(205)
u_noise,n_noise = 1e-1,1e-1
noise = lambda : np.random.uniform(-u_noise,u_noise) +\
    np.random.normal(0,n_noise)

# simulation
a1,b1 = 0.3,1.0
deriv = lambda X : np.array([  X[1],  -a1*X[1]-b1*X[0]  ])

X0 = np.array([1.0,0.0])

result = solve_ivp(deriv,X0,(0,30),0.01)

t,x,v = result[:,0],result[:,1],result[:,2]

# fix time series length
n_data = len(t)
n_out = 1000
step = n_data//n_out

x, v, t = x[::step], v[::step], t[::step]
while len(t) > n_out:
    x, v, t = x[:-1], v[:-1], t[:-1]

# add noise
for i in range(n_out):
    x[i] = x[i] + noise()*0.3
    v[i] = v[i] + noise()*0.2

import tsaopy

# load data
data_t,data_x,data_v = t,x,v
data_x_sigma,data_v_sigma = 0.15,0.15

# priors

x0_prior = tsaopy.tools.uniform_prior(0.7,1.3)
v0_prior = tsaopy.tools.uniform_prior(-1.0,1.0)
a1_prior = tsaopy.tools.uniform_prior(-5.0,5.0)
b1_prior = tsaopy.tools.uniform_prior(0.0,5.0)
    
# parameters

x0 = tsaopy.parameters.Fitting(1.0,'x0',1,x0_prior)
v0 = tsaopy.parameters.Fitting(0.0,'v0',1,v0_prior)
a1 = tsaopy.parameters.Fitting(0.0, 'a', 1, a1_prior)
b1 = tsaopy.parameters.Fitting(0.5,'b',1,b1_prior)

parameters = [x0,v0,a1,b1]

# model 2 (velocity)

model2 = tsaopy.models.PVModel(parameters,data_t,data_x,data_v,
                            data_x_sigma,data_v_sigma)

sampler,_,_,_ = model2.setup_sampler(200, 300, 300)
samples, flat_samples = sampler.get_chain(), sampler.get_chain(flat=True)

label_list = model2.params_labels
tsaopy.tools.cornerplots(flat_samples,label_list)
