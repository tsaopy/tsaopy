# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# cargar csv y convertir a array de numpy
filename = 'osc05C0-03.csv'
data_pd = pd.read_csv(filename, sep=';', decimal=',')
data_np = data_pd.to_numpy(dtype=float)

# descartar primeras entradas y elegir columnas
data_np = data_np[10:]
data_t, data_x_long, data_x_trans = data_np[:, 0], data_np[:, 2],\
                                    data_np[:, 3]

data_len = len(data_t)

# aplicar filtros
data_x_long, data_x_trans = gaussian_filter1d(data_x_long, 25),\
                            gaussian_filter1d(data_x_trans, 25)

data_x_long, data_x_trans = savgol_filter(data_x_long, 35, 2),\
                            savgol_filter(data_x_trans, 35, 2)

# plotear series temporales
plt.figure(figsize=(7, 5), dpi=150)
plt.plot(data_t, data_x_long, color='tab:red')
plt.show()

plt.figure(figsize=(7, 5), dpi=150)
plt.plot(data_t, data_x_trans, color='tab:purple')
plt.show()

# guardar datos
np.savetxt(filename[:-4]+'_processed_long.txt',
           [[data_t[_], data_x_long[_]] for _ in range(data_len)])
np.savetxt(filename[:-4]+'_processed_trans.txt',
           [[data_t[_], data_x_trans[_]] for _ in range(data_len)])
