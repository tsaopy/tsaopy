import sys
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import corner
from emcee.autocorr import function_1d

from tsaopy.bendtools import test_var_is_number

#                                   PRIOR CLASSES


class uniform_prior:
    """
    This class defines a probability distribution object. Initializing an instance of this class
    requires to imput the PDF parameters as per the usual mathematical convention. The PDF created
    is given normalized.

    An instance of this class can be called to compute the probability of a float, and it only
    takes the float as argument.

    This class defines a uniform probability distribution.
    """

    def __init__(self, xmin, xmax):
        if not test_var_is_number(xmin) or not test_var_is_number(xmax):
            sys.exit("tsaopy priors error: input parameters are not real numbers.")
        if not xmax > xmin:
            sys.exit("tsaopy priors error: upper bound is not greater than lower bound.")
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        p = 1 if x < self.xmax and x > self.xmin else 0
        return p / (self.xmax - self.xmin)


class normal_prior:
    """
    This class defines a probability distribution object. Initializing an instance of this class
    requires to imput the PDF parameters as per the usual mathematical convention. The PDF created
    is given normalized.

    An instance of this class can be called to compute the probability of a float, and it only
    takes the float as argument.

    This class defines a normal(aka Gaussian) probability distribution.
    """

    def __init__(self, x0, sigma):
        if not test_var_is_number(x0) or not test_var_is_number(sigma):
            sys.exit("tsaopy priors error: input parameters are not real numbers.")
        self.x0 = x0
        self.sigma = sigma

    def __call__(self, x):
        p = (np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2) / sqrt(2 * np.pi) / self.sigma)
        return p


#                                   PLOTTING


def cornerplots(flat_samples, labels_list):
    """
    This function makes cornerplots for a given sample and a list of labels for each parameter.

    The middle red line on each PDF marks the mean which in general is NOT the same as the mode,
    except on some particular PDFs such as the normal distribution.
    The dashed grey lines mark the 16/84 and 84/16 quantiles which indicate the SD in a normal
    distribution.
    """
    dim = len(labels_list)
    sample_truths = [np.mean(flat_samples[:, _]) for _ in range(dim)]

    corner.corner(
        flat_samples,
        labels=labels_list,
        quantiles=(0.16, 0.84),
        show_titles=True,
        title_fmt=".3g",
        truths=sample_truths,
        truth_color="tab:red",
    )
    plt.show()


def traceplots(samples, labels_list):
    """
    This function makes traceplots for each parameter of a given sample. Notice that in this
    case the samples object is not given flattened.

    A trace plot shows the evolution of each walker for a parameter during an MCMC run. This is
    used for analyzing the convergence of an MCMC chain, or to diagnose problems in a not
    converging chain.
    """
    dim = len(labels_list)
    fig, axes = plt.subplots(dim, figsize=(10, 7), dpi=100, sharex=True)
    plt.suptitle("parameter traces")
    for i in range(dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()


def autocplots(flat_samples, labels_list):
    """
    This function plots the autocorrelation function for each parameter of a given sample using the
    function_1d method in emcee.autocorr. It is used to asses the convergence of an MCMC chain.

    An autocorrelation function that quickly drops from 1 to 0, and keeps oscillating around 0
    afterwards suggests that the samples might come from a converged chain. It is not a final
    answer and you shoud try other tests at the same time.

    If the autocorrelation function does not show the behaviour described above you cannot trust
    that the chain has converged and therefore that your results are acceptable.
    """
    dim, clen = len(labels_list), len(flat_samples)
    step_slice = clen // 100
    aux_dom = range(0, clen, step_slice)
    aux_fl = flat_samples[::step_slice]
    autocfs = np.array([function_1d(aux_fl[:, _]) for _ in range(dim)])
    fig, axes = plt.subplots(dim, figsize=(10, 7), dpi=200, sharex=True)
    plt.suptitle("autocorrelation functions")
    for i in range(dim):
        ax = axes[i]
        ax.stem(aux_dom, autocfs[i, :], markerfmt="")
        ax.set_ylabel(labels_list[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("sample number")
    plt.show()
