import numpy as np
import matplotlib.pyplot as plt
from oscadsf2py import simulationv  # f2py numerical simulation

# set random seed
np.random.seed(0o204)

# aux function to add noise
u_noise, n_noise = 5e-2, 5e-2
def noise():
    return np.random.normal(0, n_noise)\
           + np.random.uniform(-u_noise, u_noise)


# ### simulation parameters
# tf : final time, A : damping coefs, B : potential coefs
# C : mixed coefs matrix, F : driving force, h : integration timestep

tf, A, B, C, F, h = 50, [0.2], [1.0],\
                    np.array([[0.0],
                              [0.0]]), \
                    [0.0, 0.0, 0.0], 1e-4
na, nb, cn, cm, n_data = len(A), len(B), len(C[:,0]), len(C[0,:]), int(tf/h)+1

# init conds
x0, v0 = 1.0, 0.0
# do simulation
xv_results, t = simulationv([x0, v0], A, B, C, F, h, n_data, na, nb, cn, cm),\
       np.linspace(0, tf, n_data)
       
x,v = xv_results[:,0], xv_results[:,1]

# ### thin data arrays to given output length
# definir largo de output

n_out = 1000
step = n_data//n_out

x, v, t = x[::step], v[::step], t[::step]
while len(x) > n_out:
    x, v, t = x[:-1], v[:-1], t[:-1]

# add noise
for _ in range(len(x)):
    x[_] = x[_] + noise()
    v[_] = v[_] + noise()/1.2

# plot data
plt.figure(figsize=(15, 5), dpi=150)
plt.plot(t, x, lw=0.5, color = 'tab:red', label='position')
plt.plot(t, v, color='tab:purple', label='velocity')
plt.legend()
plt.show()

# save data
np.savetxt('experiment_data.txt', [[t[_], x[_], v[_]] for _ in range(len(t))])
