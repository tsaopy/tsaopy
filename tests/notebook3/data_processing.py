import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame = pd.read_csv('experiment_data.csv',sep=';',decimal=",")

data_array = data_frame.to_numpy() [10:-10]

data_t, data_y = data_array [:,0], data_array [:,3]

split = 1
data_t, data_y = (data_t-data_t[0])[::split], (data_y-data_y[0])[::split]

plt.figure(figsize=(9,5),dpi=200)
plt.scatter(data_t,data_y,color='black',s=5.0)
plt.show()

np.savetxt('experiment_data.txt', [[data_t[_], data_y[_]] for _ in 
                                   range(len(data_t))])
