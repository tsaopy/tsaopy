import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from matplotlib import pyplot as plt
import emcee

from tsaopy.f2pyauxmod import simulation, simulationv
from tsaopy.bendtools import fitparams_info, params_array_shape, test_params_are_ok

# model classes


class PModel:
    """
    This class builds tsaopy model objects. This object condenses all necessary variables to set up
    the ODE according to the parameters you provided plus the MCMC configuration to do the fitting.

    It includes some QOL methods suchs as changing the initial values of the chain, the time step
    for the simulations, the number of CPU cores used during simulations, and type of emcee moves
    used in the MCMC chain. It also includes some plotting methods.

    This model class only uses x(t) information for the fitting.
    """

    def __init__(self, parameters, t_data, x_data, x_unc):
        test_params_are_ok(parameters)

        self.parameters = parameters
        self.t_data = t_data
        self.x_data = x_data
        self.x_unc = x_unc

        self.datalen = len(t_data)
        self.t0 = t_data[0]
        self.tsplit = 4
        self.dt = (t_data[-1] - self.t0) / (self.datalen - 1) / self.tsplit

        self.params_to_fit = [p for p in parameters if not p.fixed]
        self.ndim = len(self.params_to_fit)
        self.ptf_ini_values = [p.value for p in self.params_to_fit]
        self.ptf_info, self.params_labels = fitparams_info(self.params_to_fit)
        self.priors_array = [p.prior for p in self.params_to_fit]

        self.parray_shape = params_array_shape(self.parameters)
        self.alens = (self.parray_shape[2][0],
                      self.parray_shape[3][0],
                      self.parray_shape[4][0],
                      self.parray_shape[4][1])

        self.mcmc_moves = None
        self.cpu_cores = cpu_count() - 2

    # simulations

    def setup_simulation_arrays(self, coords):

        na, nb, cn, cm, = self.alens
        x0v0, A, B, C, F = (np.zeros(2), np.zeros(na), np.zeros(nb),
                            np.zeros((cn, cm)), np.zeros(3))

        for _ in self.parameters:
            if _.ptype == "x0":
                x0v0[0] = _.value
            elif _.ptype == "v0":
                x0v0[1] = _.value
            elif _.ptype == "a":
                A[_.index - 1] = _.value
            elif _.ptype == "b":
                B[_.index - 1] = _.value
            elif _.ptype == "c":
                q = _.index
                C[(q[0] - 1, q[1] - 1)] = _.value
            elif _.ptype == "f":
                F[_.index - 1] = _.value

        results = [x0v0, A, B, C, F]
        ptf_index_info = self.ptf_info

        for q in ptf_index_info:
            i = ptf_index_info.index(q)
            results[q[0]][q[1]] = coords[i]

        return results

    def predict(self, coords):

        dt, tsplit, datalen = self.dt, self.tsplit, self.datalen
        na, nb, cn, cm, = self.alens

        x0v0_simu, A_simu, B_simu, C_simu, F_simu = self.setup_simulation_arrays(coords)

        return simulation(x0v0_simu, A_simu, B_simu, C_simu, F_simu,
                          dt, tsplit * datalen, na, nb, cn, cm)[::tsplit]

    # mcmc stuff

    def log_prior(self, coefs):
        result = 1
        for i in range(self.ndim):
            prob = self.priors_array[i](coefs[i])
            if prob <= 0:
                return -np.inf
            else:
                result = result * prob
        return np.log(result)

    def log_likelihood(self, coefs):
        prediction = self.predict(coefs)
        if not np.isfinite(prediction[-1]):
            return -np.inf
        ll = - 0.5 * np.sum(((prediction - self.x_data) / self.x_unc) ** 2)
        return ll

    def log_probability(self, coefs):
        lp = self.log_prior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(coefs)

    def setup_sampler(self, n_walkers, burn_iter, main_iter):

        p0 = [self.ptf_ini_values + 1e-7 * np.random.randn(self.ndim)
              for i in range(n_walkers)]

        with Pool(processes=self.cpu_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_probability,
                                            moves=self.mcmc_moves, pool=pool)

            print("")
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            sampler.reset()

            print("")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

            return sampler, pos, prob, state

    # tools

    def update_initvals(self, newinivalues):
        self.ptf_ini_values = newinivalues

    def set_mcmc_moves(self, moves):
        self.mcmc_moves = moves

    def set_cpu_cores(self, cores):
        self.cpu_cores = cores

    def update_tsplit(self, newtsplit):
        self.tsplit = newtsplit
        self.dt = (self.t_data[-1] - self.t0) / (self.datalen - 1) / self.tsplit

    def neg_ll(self, coords):
        return -self.log_probability(coords)

    # plots

    def plot_measurements(self, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
            )
        plt.legend()
        plt.show()

    def plot_simulation(self, coords, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
            )
        plt.plot(
            self.t_data, self.predict(coords), color="tab:red", label="x simulation"
            )
        plt.legend()
        plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PVModel(PModel):
    """
    Similar to the PModel, this model ajusts the parameters to both x(t) and v(t) data.
    """

    def __init__(self, parameters, t_data, x_data, v_data, x_unc, v_unc):
        super().__init__(parameters, t_data, x_data, x_unc)
        self.v_data = v_data
        self.v_unc = v_unc

    def predict(self, coords):

        dt, tsplit, datalen = self.dt, self.tsplit, self.datalen
        na, nb, cn, cm, = self.alens

        x0v0_simu, A_simu, B_simu, C_simu, F_simu = self.setup_simulation_arrays(coords)

        return simulationv(x0v0_simu, A_simu, B_simu, C_simu, F_simu,
                           dt, tsplit * datalen, na, nb, cn, cm)[::tsplit]

    def log_likelihood(self, coefs):
        prediction = self.predict(coefs)
        predx, predv = prediction[:, 0], prediction[:, 1]
        if not np.isfinite(predv[-1]):
            return -np.inf
        ll = - 0.5 * np.sum(((predx - self.x_data) / self.x_unc) ** 2) \
             - 0.5 * np.sum(((predv - self.v_data) / self.v_unc) ** 2)
        return ll

    def log_probability(self, coefs):
        lp = self.log_prior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(coefs)

    def setup_sampler(self, n_walkers, burn_iter, main_iter):

        p0 = [self.ptf_ini_values + 1e-7 * np.random.randn(self.ndim)
              for i in range(n_walkers)]

        with Pool(processes=self.cpu_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, self.ndim, self.log_probability,
                                            moves=self.mcmc_moves, pool=pool)

            print("")
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            sampler.reset()

            print("")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

            return sampler, pos, prob, state

    ###################################################################

    # tools

    def neg_ll(self, coords):
        return -self.log_probability(coords)

    # plots

    def plot_measurements(self, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
        )
        plt.scatter(
            self.t_data, self.v_data, color="tab:blue", s=1.0, label="v measurements"
        )
        plt.legend()
        plt.show()

    def plot_measurements_x(self, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
        )
        plt.legend()
        plt.show()

    def plot_measurements_v(self, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.v_data, color="tab:blue", s=1.0, label="v measurements"
        )
        plt.legend()
        plt.show()

    def plot_simulation(self, coords, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
        )
        plt.scatter(
            self.t_data, self.v_data, color="tab:blue", s=1.0, label="v measurements"
        )
        plt.plot(
            self.t_data,
            self.predict(coords)[:, 0],
            color="tab:red",
            label="x simulation",
        )
        plt.plot(
            self.t_data,
            self.predict(coords)[:, 1],
            color="tab:purple",
            label="v simulation",
        )
        plt.legend()
        plt.show()

    def plot_simulation_x(self, coords, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data, color="black", s=1.0, label="x measurements"
        )
        plt.plot(
            self.t_data,
            self.predict(coords)[:, 0],
            color="tab:red",
            label="x simulation",
        )
        plt.legend()
        plt.show()

    def plot_simulation_v(self, coords, figsize=(7, 5), dpi=150):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.v_data, color="tab:blue", s=1.0, label="v measurements"
        )
        plt.plot(
            self.t_data,
            self.predict(coords)[:, 1],
            color="tab:purple",
            label="v simulation",
        )
        plt.legend()
        plt.show()
