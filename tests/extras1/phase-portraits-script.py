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

# simulation
k,g = 1.0,0.5
deriv1 = lambda X : np.array([  X[1],  np.sin(X[0])*(k*np.cos(X[0])-g)])

b1_,b3_ = -0.374, 0.298
deriv2 = lambda X : np.array([  X[1], -b1_*X[0]-b3_*X[0]**3] )

b1,b3,b5 = -0.484, 0.526, -0.0808
deriv3 = lambda X : np.array([  X[1], -b1*X[0]-b3*X[0]**3-b5*X[0]**5] )

deriv = [deriv1,deriv2,deriv3]

init_conds = [[1.0,0.05],[1.0,0.1],[1.0,0.2],[1.0,0.3],
              [1.0,0.5],[1.0,0.6],[1.0,0.8]]
colours = ['mediumslateblue','royalblue','tab:cyan','tab:green',
           'gold','tab:orange','tab:red']
model_labels = ['original equation','3rd order tsaopy',
                '5th order tsaopy']
plot_labels = ['$\ddot{x}-\sin(x)\cos(x)+0.5\sin(x)=0$',
               '$\ddot{x}-0.374x+0.298x^3=0$',
                '$\ddot{x}-0.484x+0.526x^3-0.0808x^5=0$']
bottom_labels_ycoord = [-0.05,-0.05,-0.1]


fig, axes = plt.subplots(3, figsize=(10, 18), dpi=100, sharex=True)
plt.suptitle('phase portraits',size=24,x=0.3,y=0.92)
for i in range(3):
        ax = axes[i]
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-0.9,0.9)
        ax.set_ylabel(model_labels[i],size=20)
        ax.yaxis.set_label_coords(-0.07, 0.5)
        ax.set_xlabel(plot_labels[i],size=16)
        ax.xaxis.set_label_coords(0.5,bottom_labels_ycoord[i])
        
        ax.plot([-1.95,1.95],[0,0],color='gray',ls='--',lw=0.7)
        ax.plot([0,0],[-0.8,0.8],color='gray',ls='--',lw=0.7)
        ax.plot([1.045,1.045],[-0.4,0.4],color='gray',ls='--',lw=0.7)
        
        for j in range(len(init_conds)):
            x0,color = init_conds[j],colours[j]
            result = solve_ivp(deriv[i],np.array(x0),(0,30),0.01)
            t,x,v = result[:,0],result[:,1],result[:,2]
            ax.plot(x, v, color = color)
            result = solve_ivp(deriv[i],-np.array(x0),(0,30),0.01)
            t,x,v = result[:,0],result[:,1],result[:,2]
            ax.plot(x, v, color = color)

plt.show()
