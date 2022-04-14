# ### imports
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from oscadsf2py import simulation  # integrad numerico importado x f2py

# ajustar semilla de np random
np.random.seed(0o204)

# funcion auxiliar para agregar ruido a los datos
u_noise, n_noise = 5e-2, 5e-2


def noise():
    return np.random.normal(0, n_noise)\
           + np.random.uniform(-u_noise, u_noise)


# ### ajustar parametros de la simulacion
# tf : tiempo final, A : coefs de la fuerza de damping, B : coefs de
# la fuerza conservativa, h : paso de integracion

tf, A, B, C, F, h = 50, [0.2], [1.0],\
                    np.array([[0.0],
                              [0.0]]), \
                    [0.0, 0.0, 0.0], 1e-4
na, nb, cn, cm, n_data = len(A), len(B), len(C[:,0]), len(C[0,:]), int(tf/h)+1
# ### hacer la simulacion
# condiciones iniciales
x0, v0 = 1.0, 0.0
# hacer la simulacion
x, t = simulation([x0, v0], A, B, C, F, h, n_data, na, nb, cn, cm),\
       np.linspace(0, tf, n_data)

# ### definir largo de salida de la serie de datos y agregar ruido
# definir largo de output
n_out = 1000

# recortar la serie para que tenga el largo n_out
step = n_data//n_out
x, t = x[::step], t[::step]
while len(x) > n_out:
    x, t = x[:-1], t[:-1]

# agregar ruido
for _ in range(1,len(x)):
    x[_] = x[_] + noise()
    
### procesar filtrado
gf_sd = 9
x_process = gaussian_filter1d(x, gf_sd)

# ### finalizacion
# ver esquema de los datos
plt.figure(figsize=(15, 5), dpi=150)
plt.plot(t, x, lw=0.5)
plt.plot(t, x_process, color='tab:red')
plt.show()

# guardar los datos
np.savetxt('experiment_data.txt', [[t[_], x[_]] for _ in range(len(t))])
