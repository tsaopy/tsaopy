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

    Defines a probability distribution object. An instance of this class can be
    called to compute the probability density of a float, and it only takes the
    float as argument.

    The uniform prior distribution is defined as in the usual mathematical
    convention and is given normalized.

    """

    def __init__(self, xmin, xmax):
        """

        Initialice the instance.

        Args:
        xmin(int or float): lower bound of the uniform distribution.
        xmax(int or float): upper bound of the uniform distribution.

        Returns:
        tsaopy pdf object instance.

        """
        if not test_var_is_number(xmin) or not test_var_is_number(xmax):
            sys.exit("tsaopy priors error: input parameters are not real "
                     "numbers.")
        if not xmax > xmin:
            sys.exit("tsaopy priors error: upper bound is not greater than "
                     "lower bound.")
        self.xmin = xmin
        self.xmax = xmax

    def __call__(self, x):
        """

        Compute the probability of a real valued variable.

        Args:
        x(int or float)

        Returns:
        p(x) as per the mathematical definition of the PDF.

        """
        p = 1 if x < self.xmax and x > self.xmin else 0
        return p / (self.xmax - self.xmin)


class normal_prior:
    """

    Define a probability distribution object. An instance of this class can be
    called to compute the probability density of a float, and it only takes the
    float as argument.

    The normal (aka Gaussian) prior distribution is defined as in the usual
    mathematical convention and is given normalized.

    """

    def __init__(self, x0, sigma):
        """

        Initialice the instance.

        Args:
        x0(int or float): central value of the normal distribution.
        simga(int or float): standard deviation of the normal distribution.

        Returns:
        tsaopy pdf object instance.

        """
        if not test_var_is_number(x0) or not test_var_is_number(sigma):
            sys.exit("tsaopy priors error: input parameters are not real "
                     "numbers.")
        self.x0 = x0
        self.sigma = sigma

    def __call__(self, x):
        """

        Compute the probability of a real valued variable.

        Args:
        x(int or float)

        Returns:
        p(x) as per the mathematical definition of the PDF.

        """
        p = (np.exp(-0.5 * ((x - self.x0) / self.sigma) ** 2)) / sqrt(
            2 * np.pi) / self.sigma
        return p


#                                   PLOTTING


def cornerplots(flat_samples, labels_list):
    """

    Make cornerplots given a sample and a list of labels for each parameter.

    The middle red line on each PDF marks the median, which is also the central
    value reported above the plot. The dashed grey lines mark the 16/84 and
    84/16 quantiles, which indicate the SD in a normal distribution.

    Args:
    flat_samples(array): flattened array of samples.
    labels_list(list of strings): list with the label for each parameter of the
    samples.

    Returns:
    Displays created figures.

    """
    dim = len(labels_list)
    sample_medians = [np.median(flat_samples[:, _]) for _ in range(dim)]

    corner.corner(
        flat_samples,
        labels=labels_list,
        quantiles=(0.16, 0.84),
        show_titles=True,
        title_fmt=".3g",
        truths=sample_medians,
        truth_color="tab:red",
    )
    plt.show()


def traceplots(samples, labels_list):
    """

    Make traceplots for each parameter of a given sample. Notice that in this
    case the samples object is not given flattened.

    A trace plot shows the evolution of each walker for a parameter during an
    MCMC run. This is used for analyzing the convergence of an MCMC chain, or
    to diagnose problems in a not converging chain.

    Args:
    samples(array): non flattened array of samples.
    labels_list(list of strings): list with the label for each parameter of the
    samples.

    Returns:
    Displays created figures.

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

    Plot the autocorrelation function for each parameter of a given sample
    using the function_1d callable provided in emcee.autocorr. It is used to
    asses the convergence of an MCMC chain.

    An autocorrelation function that quickly drops from 1 to 0, and keeps
    oscillating around 0 afterwards suggests that the samples might come from a
    converged chain. It is not a final answer and you shoud try other tests at
    the same time.

    If the autocorrelation function does not show the behaviour described above
    you cannot trust that the chain has converged and therefore that your
    results are acceptable.

    Args:
    flat_samples(array): flattened array of samples.
    labels_list(list of strings): list with the label for each parameter of the
    samples.

    Returns:
    Displays created figures.

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
