##############################################################################
#                                 PRELIMINARY
##############################################################################

# def functions
import numpy as np
def _deriv(X_,A_,B_):
    x_,v_ = X_[0],X_[1]
    a_ = 0
    for j in range(len(A_)):
        a_ = a_ - A_[j]*v_*(abs(v_))**j
    for j in range(len(B_)):
        a_ = a_ - B_[j]*x_**(j+1)    
    return np.array([v_,a_])

def _rk4(h,X_):
    k1 = h*_deriv(X_)
    k2 = h*_deriv(X_+0.5*k1)
    k3 = h*_deriv(X_+0.5*k2)
    k4 = h*_deriv(X_+k3)
    return (X_+(k1+k4+2*k2+2*k3)/6)

### make data
np.random.seed(33525)

# noise gen function
u_noise,n_noise = 1e-2,2e-2
noise = lambda : np.random.normal(0,n_noise)+np.random.uniform(-u_noise,u_noise)

# simu
A,B = [0.0],[1.0,0.0,0.5]
dt,x0 = 1e-4,1.0+noise()
t,x,X,E = [0],[x0],np.array([x0,0]),np.sum(B)
tf = 30

_deriv_coefs = lambda X_ : _deriv(X_,A,B)

def _rk4_c(h,X_):
    k1 = h*_deriv_coefs(X_)
    k2 = h*_deriv_coefs(X_+0.5*k1)
    k3 = h*_deriv_coefs(X_+0.5*k2)
    k4 = h*_deriv_coefs(X_+k3)
    return (X_+(k1+k4+2*k2+2*k3)/6)

while t[-1]<tf:
    t.append(t[-1]+dt)
    X = _rk4_c(dt,X)
    x.append(X[0]+noise())

t,x = t[::300],x[::300]
np.savetxt('testdata.txt',[[t[_],x[_]] for _ in range(len(t))])
    
from matplotlib import pyplot as plt

plt.figure(figsize=(7,5),dpi=150)
plt.plot(t,x,color='tab:red')
plt.show()