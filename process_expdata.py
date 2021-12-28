# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# cargar csv y convertir a array de numpy
filename = 'osc05c03.csv'
data_pd = pd.read_csv(filename, sep=';', decimal=',')
data_np = data_pd.to_numpy(dtype=float)

# descartar primeras entradas y elegir columnas
data_np = data_np[10:]
data_t, data_x_long_0, data_x_trans_0 = data_np[:, 0], data_np[:, 2],\
                                    data_np[:, 3]

data_len = len(data_t)

data_t_zero = data_t[0]
for i in range(data_len):
    data_t[i] += -data_t_zero

# aplicar filtros

gf_sd = 55

data_x_long, data_x_trans = gaussian_filter1d(data_x_long_0, gf_sd),\
                            gaussian_filter1d(data_x_trans_0, gf_sd)

# plotear series temporales filtradas
plt.figure(figsize=(7, 5), dpi=150)
plt.plot(data_t, data_x_long_0,lw=0.5,label='Datos')
plt.plot(data_t, data_x_long, color='tab:red',label='Filtrado')
plt.title('Longitudinal')
plt.legend()
plt.show()

plt.figure(figsize=(7, 5), dpi=150)
plt.plot(data_t, data_x_trans_0,lw=0.5,label='Datos')
plt.plot(data_t, data_x_trans, color='tab:red',label='Filtrado')
plt.title('Transversal')
plt.legend()
plt.show()

# guardar datos
np.savetxt(filename[:-4]+'_processed_long.txt',
           [[data_t[_], data_x_long[_]] for _ in range(data_len)])
np.savetxt(filename[:-4]+'_processed_trans.txt',
           [[data_t[_], data_x_trans[_]] for _ in range(data_len)])
