# ### imports
import numpy as np
import matplotlib.pyplot as plt

from oscadsf2py import simulation  # integrad numerico importado x f2py

# ajustar semilla de np random
np.random.seed(12345)

# funcion auxiliar para agregar ruido a los datos
u_noise, n_noise = 1e-2, 2e-2


def noise():
    return np.random.normal(0, n_noise)\
           + np.random.uniform(-u_noise, u_noise)


# ### ajustar parametros de la simulacion
# tf : tiempo final, A : coefs de la fuerza de damping, B : coefs de
# la fuerza conservativa, h : paso de integracion
tf, A, B, h = 30, [0.0], [1.0, 0.0, 0.5], 1e-4
na, nb, n_data = len(A), len(B), int(tf/h)+1

# ### hacer la simulacion
# condiciones iniciales
x0, v0 = 1.0+noise(), 0.0+noise()
# hacer la simulacion y darle
x, t = simulation([x0, v0], A, B, h, n_data, na, nb),\
       np.linspace(0, tf, n_data)

# ### definir largo de salida de la serie de datos y agregar ruido
# definir largo de output
n_out = 100

# recortar la serie para que tenga el largo n_out
step = n_data//n_out
x, t = x[::step], t[::step]
while len(x) > n_out:
    x, t = x[:-1], t[:-1]

# agregar ruido
for _ in range(len(x)):
    x[_] = x[_] + noise()

# ### finalizacion
# ver esquema de los datos
plt.figure(figsize=(7, 5), dpi=150)
plt.plot(t, x, color='tab:red')
plt.show()

# guardar los datos
np.savetxt('testdata.txt', [[t[_], x[_]] for _ in range(len(t))])
