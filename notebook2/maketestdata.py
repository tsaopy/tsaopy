import numpy as np
import matplotlib.pyplot as plt

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
mu = 1.5
deriv = lambda X : np.array([  X[1],  mu*X[1]*(1-X[0]**2)-X[0]  ])

X0 = np.array([2.0,0.0])

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
    x[i] = x[i] + noise()*1
    v[i] = v[i] + noise()*2
    
# plot results
plt.figure(figsize=(7, 5), dpi=150)
plt.plot(t, x, color = 'tab:red', label='position')
plt.plot(t, v, color='tab:purple', label='velocity')
plt.legend()
plt.show()

# save results
np.savetxt('experiment_data.txt', [[t[_], x[_], v[_]] for _ in range(n_out)])
