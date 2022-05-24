import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import corner
from emcee.autocorr import function_1d

# prior classes
class uniform_prior:
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        p = 1 if x < self.xmax and x > self.xmin else 0
        return p/(self.xmax-self.xmin)

class normal_prior:
    def __init__(self, x0, sigma):
        self.x0 = x0
        self.sigma = sigma

    def __call__(self, x):
        p = np.exp(-0.5*((x-self.x0)/self.sigma)**2)/sqrt(2*np.pi)/self.sigma
        return p

# plotting
def cornerplots(flat_samples,labels_list):
    dim = len(labels_list)
    sample_truths = [np.mean(flat_samples[:, _]) for _ in range(dim)]

    fig = corner.corner(flat_samples, labels=labels_list,
                        quantiles=(0.16, 0.84), show_titles=True,
                        title_fmt='.3g', truths=sample_truths,
                        truth_color='tab:red')
    plt.show()

def traceplots(samples,labels_list):
    dim = len(labels_list)
    fig, axes = plt.subplots(dim, figsize=(10, 7), dpi=100, sharex=True)
    plt.suptitle('parameter traces')
    for i in range(dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.show()

def autocplots(flat_samples,labels_list):
    dim,clen = len(labels_list),len(flat_samples)
    step_slice = clen//100
    aux_dom = range(0,clen,step_slice)
    aux_fl = flat_samples[::step_slice]
    autocfs = np.array([function_1d(aux_fl[:,_])
               for _ in range(dim)])
    fig, axes = plt.subplots(dim, figsize=(10, 7), dpi=200, sharex=True)
    plt.suptitle('autocorrelation functions')
    for i in range(dim):
        ax = axes[i]
        ax.stem(aux_dom,autocfs[i,:],markerfmt='')
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("sample number");
    plt.show()

