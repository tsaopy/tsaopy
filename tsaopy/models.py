import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from matplotlib import pyplot as plt
import emcee

from tsaopy._f2pyauxmod import simulation, simulationv
from tsaopy._bendtools import (fitparams_info, params_array_shape,
                               test_params_are_ok)

# model classes


class PModel:
    """
    Build `tsaopy` model object.

    This object condenses all necessary variables to set up the ODE according
    to the parameters provided plus the MCMC configuration to do the fitting.

    It includes some QOL methods suchs as changing the initial values of the
    chain, the time step for the simulations, the number of CPU cores used
    during simulations, and type of emcee moves used in the MCMC chain. It also
    includes some plotting methods.

    This class only uses x(t) information for the fitting.
    """

    def __init__(self, parameters, t_data, x_data, x_unc):
        """
        Parameters
        ----------
        parameters : list
            the list of parameters that will be considered in the model. There
            must be at least three parameters including the initial conditions
            and one ODE coefficient. There can't be repeated parameters (two
            parameters having the same ptype and index).
        t_data : array
            array with the time axis of the measurements..
        x_data : array
            array with the position measurements.
        x_unc : float or int, or array
            uncertainty of your measurements. It can be either a single number
            representing the uncertainty of all measurements or an array of the
            same length as x_data with a unique value for each measurement.
        """
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
        self.mcmc_initvals = [p.value for p in self.params_to_fit]
        self.ptf_info, self.params_labels = fitparams_info(self.params_to_fit)
        self.priors_array = [p.prior for p in self.params_to_fit]

        self.parray_shape = params_array_shape(self.parameters)
        self.alens = (self.parray_shape[0][0],
                      self.parray_shape[1][0],
                      self.parray_shape[2][0],
                      self.parray_shape[2][1])

        self.fx_fix = False
        for p in self.params_to_fit:
            if p.ptype == 'log_fx':
                self.fx_fix = True
                self.log_fx_loc = self.params_to_fit.index(p)

        self.mcmc_moves = None
        if cpu_count() > 2:
          self.cpu_cores = cpu_count() - 2
        else:
          self.cpu_cores = 1

    # simulations

    def _setup_simulation_arrays(self, coords):
        """Set up the parameters array used by a simulation."""
        na, nb, cn, cm, = self.alens
        scalars, A, B, C, F = (np.zeros(3), np.zeros(na), np.zeros(nb),
                               np.zeros((cn, cm)), np.zeros(3))

        for p in self.parameters:
            if p.ptype == "ep":
                scalars[0] = p.value
            if p.ptype == "x0":
                scalars[1] = p.value
            elif p.ptype == "v0":
                scalars[2] = p.value
            elif p.ptype == "a":
                A[p.index - 1] = p.value
            elif p.ptype == "b":
                B[p.index - 1] = p.value
            elif p.ptype == "c":
                q = p.index
                C[(q[0] - 1, q[1] - 1)] = p.value
            elif p.ptype == "f":
                F[p.index - 1] = p.value

        results = [scalars, A, B, C, F]
        ptf_index_info = self.ptf_info

        for q in ptf_index_info:
            if q is not None:
                i = ptf_index_info.index(q)
                results[q[0]][q[1]] = coords[i]

        return results

    def _predict(self, coords):
        """Compute x(t) for a set of parameter values."""
        dt, tsplit, datalen = self.dt, self.tsplit, self.datalen
        na, nb, cn, cm, = self.alens

        epx0v0_simu, A_simu, B_simu, C_simu, F_simu = (
                                        self._setup_simulation_arrays(coords))

        ep_simu, x0v0_simu = epx0v0_simu[0], epx0v0_simu[1:3]

        return simulation(x0v0_simu, A_simu, B_simu, C_simu, F_simu,
                          dt, tsplit * datalen, na, nb, cn, cm
                          )[::tsplit] + ep_simu

    # mcmc stuff

    def _log_prior(self, coefs):
        """Compute the logarithmic prior of a set of parameter values."""
        result = 1
        for i in range(self.ndim):
            prob = self.priors_array[i](coefs[i])
            if prob <= 0:
                return -np.inf
            else:
                result = result * prob
        return np.log(result)

    def _log_likelihood(self, coefs):
        """Compute the logarithmic likelihood of a set of parameter values."""
        prediction = self._predict(coefs)
        if not np.isfinite(prediction[-1]):
            return -np.inf

        if self.fx_fix:
            log_fx = coefs[self.log_fx_loc]
            s2 = self.x_unc ** 2 + prediction ** 2 * np.exp(2 * log_fx)
            ll = - 0.5 * np.sum((prediction - self.x_data) ** 2 / s2 +
                                np.log(s2))
        else:
            ll = - 0.5 * np.sum(((prediction - self.x_data) / self.x_unc) ** 2)
        return ll

    def _log_probability(self, coefs):
        """Compute the logarithmic probabilty of a set of parameter values."""
        lp = self._log_prior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(coefs)

    def setup_sampler(self, n_walkers, burn_iter, main_iter):
        """
        Set up the `emcee` Sampler object and run the MCMC chain.

        See `emcee` docs for more details.

        Parameters such as the number of CPU cores and `emcee` moves used by
        the sampler can be changed from the model attributes before running
        this method. See the full docs of the model classes for more details.

        Parameters
        ----------
        n_walkers : int
            the number of walkers in the MCMC chain. See `emcee` docs for more
            details.
        burn_iter : int
            the number of steps that your chain will do during the burn in
            phase. The samples produced during burn in phase are discarded.
        main_iter : int
            the number of steps that your chain will do during the production
            phase. The samples produced during production phase are saved in
            the sampler and can be extracted for later analysis.

        Returns
        -------
        sampler : emcee Sampler instance
            Returns the `emcee` ensemble sampler after running MCMC. See
            `emcee` docs for more details.
        """
        p0 = [self.mcmc_initvals + 1e-7 * np.random.randn(self.ndim)
              for i in range(n_walkers)]

        with Pool(processes=self.cpu_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, self.ndim,
                                            self._log_probability,
                                            moves=self.mcmc_moves, pool=pool)

            print("")
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            sampler.reset()

            print("")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

            return sampler

    # tools

    def update_initvals(self, newinivalues):
        """
        Update the starting values of the MCMC chain.

        Update the initial values of the MCMC chain stored in the `tsaopy`
        model instance `mcmc_initvals` attribute. Default is a list with the
        value attribute for each parameter supplied to the model at
        initialization.

        This attribute is supplied to the `emcee` Sampler object when
        `setup_sampler`.

        Parameters
        ----------
        newinivalues : list or array
            the new values for the initial  values of the MCMC chain. The
            elements must be passed in the same order than the parameters
            arg that was passed when initializing the model.
        """
        self.mcmc_initvals = newinivalues

    def set_mcmc_moves(self, moves):
        """
        Change the `emcee` moves used in the MCMC run.

        Set the mcmc_moves attribute in the `tsaopy` model instance. Default is
        None. This attribute is supplied to the `emcee` Sampler object when the
        setup_sampler method of the `tsaopy` model object is called. It's
        possible to run MCMC chains for the same model with different moves by
        updating the attribute with this method before each chain is run.

        Parameters
        ----------
        moves : emcee moves instance
            the `emcee` moves instance to be supplied to the sampler.


        """
        self.mcmc_moves = moves

    def set_cpu_cores(self, cores):
        """
        Set the number of CPU cores used by the emcee sampler.

        Set the cpu_cores attribute in the `tsaopy` model instance. Default is
        the total number of cores in the system, obtained with
        `multiprocessing.cpu_count`, minus two, or one if `cpu_count` returns 2
        or less.

        This attribute is supplied to the emcee Sampler object when you call
        the setup_sampler method of the tsaopy model object.

        Parameters
        ----------
        cores : int
            number of CPU cores to be used by the `emcee` sampler.
        """
        self.cpu_cores = cores

    def update_tsplit(self, newtsplit):
        """
        Change the integration time in simulations.

        Set the `tsplit` attribute in the `tsaopy` model instance, which is
        used to set the time step for the numerical integrations.

        The time step in the numerical integration is computed as

        dt = (tf - t0)/(ndata-1)/tsplit

        This means that the time step is obtained by dividing the difference
        between consecutive t values over `tsplit`. Default value is 4.
        Users that want to improve computing time may reduce this attribute
        to 3, 2, or 1, at the expense of a possible precission loss in the
        numerical simulations.

        Parameters
        ----------
        newtsplit : int
            the new value for tsplit that will also update the integration
            time step.
        """
        self.tsplit = newtsplit
        self.dt = (self.t_data[-1] - self.t0) / (
                                                self.datalen - 1) / self.tsplit

    def neg_ll(self, coords):
        """
        Compute the negative logarithmic likelihood.

        Compute the negative logarithmic likelihood for a set of parameter
        values for the defined model. This is best used when optimizing the
        initial values with an external optimizer.

        Parameters
        ----------
        coords : list or array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.

        Returns
        -------
        float
            value of `neg_ll` for the given parameter values.

        Examples
        -------
            f_to_minimize = my_model.neg_ll
            external_function_minimizer(f_to_minimize, *args)
        """
        return -self._log_likelihood(coords)

    # plots

    def plot_measurements(self, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,x(t)) series provided to the model.

        Parameters
        ----------
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
            )
        plt.legend()
        plt.show()

    def plot_simulation(self, coords, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,x(t)) series provided to the model,
        and a lineplot of a simulation using values provided in the coords arg.

        Parameters
        ----------
        coords : list or array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
            )
        plt.plot(
            self.t_data, self._predict(coords),
            color="tab:red", label="x simulation"
            )
        plt.legend()
        plt.show()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PVModel(PModel):
    """
    Build `tsaopy` model object.

    This object condenses all necessary variables to set up the ODE according
    to the parameters provided plus the MCMC configuration to do the fitting.

    It includes some QOL methods suchs as changing the initial values of the
    chain, the time step for the simulations, the number of CPU cores used
    during simulations, and type of emcee moves used in the MCMC chain. It also
    includes some plotting methods.

    This class fits the parameters to both x(t) and v(t) data.
    """

    def __init__(self, parameters, t_data, x_data, v_data, x_unc, v_unc):
        """
        Parameters
        ----------
        parameters : list
            the list of parameters that will be considered in the model. There
            must be at least three parameters including the initial conditions
            and one ODE coefficient. There can't be repeated parameters (two
            parameters having the same ptype and index).
        t_data : array
            array with the time axis of the measurements.
        x_data : array
            array with the position measurements.
        v_data : array
            array with the velocity measurements. Note that there can't be a
            scale factor between x(t) and v(t), v(t) must be exactly equal
            to the time derivative of x(t).
        x_unc : float or int, or array
            uncertainty of the x(t) measurements. It can be either a single
            number representing the uncertainty of all measurements or an array
            of the same length as `x_data` with a unique value for each
            measurement.
        v_unc : float or int, or array
            same as `x_unc` but for v(t) measurements.
        """
        super().__init__(parameters, t_data, x_data, x_unc)
        self.v_data = v_data
        self.v_unc = v_unc
        self.fv_fix = False
        for p in self.params_to_fit:
            if p.ptype == 'log_fv':
                self.fv_fix = True
                self.log_fv_loc = self.params_to_fit.index(p)

    def _predict(self, coords):
        """Compute x(t) and v(t) for a set of parameter values."""
        dt, tsplit, datalen = self.dt, self.tsplit, self.datalen
        na, nb, cn, cm, = self.alens

        epx0v0_simu, A_simu, B_simu, C_simu, F_simu = (
                                        self._setup_simulation_arrays(coords))

        ep_simu, x0v0_simu = epx0v0_simu[0], epx0v0_simu[1:]

        simu_result = simulationv(x0v0_simu, A_simu, B_simu, C_simu, F_simu,
                                  dt, tsplit * datalen, na, nb, cn,
                                  cm)[::tsplit]

        simu_result[:, 0] = simu_result[:, 0] + ep_simu
        return simu_result

    def _log_likelihood(self, coefs):
        """Compute the logarithmic likelihood of a set of parameter values."""
        prediction = self._predict(coefs)
        predx, predv = prediction[:, 0], prediction[:, 1]
        if not np.isfinite(predv[-1]):
            return -np.inf

        if self.fx_fix:
            log_fx = coefs[self.log_fx_loc]
            s2_x = self.x_unc ** 2 + predx ** 2 * np.exp(2 * log_fx)
            ll = - 0.5 * np.sum((predx - self.x_data) ** 2 / s2_x +
                                np.log(s2_x))
        else:
            ll = - 0.5 * np.sum(((predx - self.x_data) / self.x_unc) ** 2)

        if self.fv_fix:
            log_fv = coefs[self.log_fv_loc]
            s2_v = self.v_unc ** 2 + predv ** 2 * np.exp(2 * log_fv)
            ll = ll - 0.5 * np.sum((predv - self.v_data) ** 2 / s2_v +
                                   np.log(s2_v))
        else:
            ll = ll - 0.5 * np.sum(((predv - self.v_data) / self.v_unc) ** 2)

        return ll

    def _log_probability(self, coefs):
        """Compute the logarithmic probability of a set of parameter values."""
        lp = self._log_prior(coefs)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(coefs)

    def setup_sampler(self, n_walkers, burn_iter, main_iter):
        """
        Set up the `emcee` Sampler object and run the MCMC chain.

        See `emcee` docs for more details.

        Parameters such as the number of CPU cores and `emcee` moves used by
        the sampler can be changed from the model attributes before running
        this method. See the full docs of the model classes for more details.

        Parameters
        ----------
        n_walkers : int
            the number of walkers in the MCMC chain. See `emcee` docs for more
            details.
        burn_iter : int
            the number of steps that your chain will do during the burn in
            phase. The samples produced during burn in phase are discarded.
        main_iter : int
            the number of steps that your chain will do during the production
            phase. The samples produced during production phase are saved in
            the sampler and can be extracted for later analysis.

        Returns
        -------
        sampler : emcee Sampler instance
            Returns the `emcee` ensemble sampler after running MCMC. See
            `emcee` docs for more details.
        """
        p0 = [self.mcmc_initvals + 1e-7 * np.random.randn(self.ndim)
              for i in range(n_walkers)]

        with Pool(processes=self.cpu_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, self.ndim,
                                            self._log_probability,
                                            moves=self.mcmc_moves, pool=pool)

            print("")
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            sampler.reset()

            print("")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

            return sampler

    # tools

    def neg_ll(self, coords):
        """See docs for PModel."""
        return -self._log_likelihood(coords)

    # plots

    def plot_measurements(self, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing both the (t,x(t)) and (t,v(t)) series
        provided to the model.

        Parameters
        ----------
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
        )
        plt.scatter(
            self.t_data, self.v_data,
            color="tab:blue", s=1.0, label="v measurements"
        )
        plt.legend()
        plt.show()

    def plot_measurements_x(self, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,x(t)) series provided to the model.

        Parameters
        ----------
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
        )
        plt.legend()
        plt.show()

    def plot_measurements_v(self, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,v(t)) series provided to the model.

        Parameters
        ----------
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.v_data,
            color="tab:blue", s=1.0, label="v measurements"
        )
        plt.legend()
        plt.show()

    def plot_simulation(self, coords, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing both the (t,x(t)) and (t,v(t)) series
        provided to the model, and lineplots of a simulation using values
        provided in the coords arg.

        Parameters
        ----------
        coords : list or array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
        )
        plt.scatter(
            self.t_data, self.v_data,
            color="tab:blue", s=1.0, label="v measurements"
        )
        plt.plot(
            self.t_data,
            self._predict(coords)[:, 0],
            color="tab:red",
            label="x simulation",
        )
        plt.plot(
            self.t_data,
            self._predict(coords)[:, 1],
            color="tab:purple",
            label="v simulation",
        )
        plt.legend()
        plt.show()

    def plot_simulation_x(self, coords, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,x(t)) series provided to the model,
        and a lineplot of a simulation using values provided in the coords arg.

        Parameters
        ----------
        coords : list or array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.x_data,
            color="black", s=1.0, label="x measurements"
        )
        plt.plot(
            self.t_data,
            self._predict(coords)[:, 0],
            color="tab:red",
            label="x simulation",
        )
        plt.legend()
        plt.show()

    def plot_simulation_v(self, coords, figsize=(7, 5), dpi=100):
        """
        Make a scatterplot showing the (t,v(t)) series provided to the model,
        and a lineplot of a simulation using values provided in the coords arg.

        Parameters
        ----------
        coords : list or array
            values for each parameter. The elements must be passed in the same
            order than the parameters arg that was passed when initializing the
            model.
        figsize : tuple, optional
            proportions of the image passed to pyplot. The default is (7, 5).
        dpi : TYPE, optional
            dots per inch passed to pyplot. The default is 100.

        Returns
        -------
        Displays created figures.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        plt.scatter(
            self.t_data, self.v_data,
            color="tab:blue", s=1.0, label="v measurements"
        )
        plt.plot(
            self.t_data,
            self._predict(coords)[:, 1],
            color="tab:purple",
            label="v simulation",
        )
        plt.legend()
        plt.show()
