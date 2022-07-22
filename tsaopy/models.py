"""Main submodule of the library."""
import numpy as np
import emcee
from Multiprocessing import Pool
import quickemcee as qemc


#           Aux stuff, raises etc
class ModelInitException(Exception):
    """Custom exception for Model instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Model object.')
        super().__init__(msg)


#           tsaopy scripts
class Model:
    """Main object of the library."""

    def __init__(self, ode_coefs, events):
        """Do."""
        # run tests
        for i in ode_coefs:
            try:
                ilist = ode_coefs[i]
                # check that every parameter has 3 elements in its touple
                for coef in ilist:
                    if len(coef) < 3:
                        raise ModelInitException('some ODE coef is missing '
                                                 'parameters.')
                    if len(coef) > 3:
                        raise ModelInitException('some ODE coef has too many '
                                                 'parameters.')
                    # check that for every parameter p(x) > 0
                    else:
                        _, x, p = coef
                        if not p(x) > .0:
                            raise ModelInitException('some ODE coef initial '
                                                     'guess does not return '
                                                     'a positive value for its'
                                                     ' prior.')
            # general exception
            except Exception as exception:
                raise ModelInitException('ode coefs dict has something wrong.'
                                         ) from exception

        # a and b 1D vectors
        # do

        # f vector
        if 'f' not in ode_coefs:
            self.using_f = False
        elif 'f' in ode_coefs:
            self.using_f = True

    def _ode_arrays(self, ode_coefs):
        alen, blen, cn, cm = self.alen, self.blen, self.cn, self.cm
        A, B, F, C = (np.zeros(alen), np.zeros(blen), np.zeros(3),
                      np.zeros((cn, cm)))
        return A, B, F, C

    def _log_likelihood(self, coords):
        """Compute log likelihood."""
        odecoefs_ndim = self.odecoefs_ndim
        ode_coefs, event_params = (coords[:odecoefs_ndim],
                                   coords[odecoefs_ndim:])

        A, B, F, C = self._ode_arrays(ode_coefs)

        last_n = 0
        result = .0
        # iterate over all events
        for event in self.events:
            x0v0 = event_params[last_n:last_n + 2]
            last_n += 2
            if event.using_ep:
                ep = event_params[last_n]
                last_n += 1
            else:
                ep = None
            if event.using_logfx:
                logfx = event_params[last_n]
                last_n += 1
            else:
                logfx = None
            if event.using_logfv:
                logfv = event_params[last_n]
                last_n += 1
            else:
                logfv = None

            result += event._log_likelihood(self, A, B, F, C, x0v0,
                                            ep, logfx, logfv)

    def _log_prior(self, coords):
        """Compute log prior."""
        odecoefs_ndim = self.odecoefs_ndim
        odecoords = coords[:odecoefs_ndim]
        result = 1
        for i, p in enumerate(self.odepriors):
            prob = p(odecoords[i])
            if prob <= .0:
                return -np.inf
            result *= prob
        result = np.log(result)

        last_n = odecoefs_ndim
        # iterate over all events
        for event in self.events:
            event_ndim = event.ndim
            event_coords = coords[last_n:last_n + event_ndim]
            last_n += event_ndim
            result += event._log_prior(event_coords)

        return result

    def _log_probability(self, coords):
        """Compute log probability."""
        lp = self._log_prior(coords)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(coords)

    def run_mcmc_chain(self, nwalkers, burn_iter, main_iter,
                       init_x=None, moves=None, workers=1):
        """
        Instance an `emcee` Ensemble Sambler and run an MCMC chain with it.

        Parameters
        ----------
        nwalkers : int
            number of walkers.
        burn_iter : int
            the number of steps that the chain will do during the burn in
            phase. The samples produced during burn in phase are discarded.
        main_iter : int
            the number of steps that the chain will do during the production
            phase. The samples produced during production phase are saved in
            the sampler and can be extracted for later analysis.
        init_x : array, optional
            1D array of length ndim with an initial guess for the parameters
            values. When set as None uses all zeroes. The default is None.
        moves : emcee moves object, optional
            `emcee` moves object. The default is None.
        workers : int, optional
            Parallelize the computing by setting up a pool of workers of size
            workers. The default is 1.
        Returns
        -------
        sampler : emcee Ensemble Sampler object
            The instanced `emcee` sampler for which the chain is run.
        """
        ndim = self.ndim

        if init_x is None:
            init_x = np.zeros(ndim)

        p0 = [init_x + 1e-7 * np.random.randn(ndim)
              for i in range(nwalkers)]

        if workers == 1:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            self._log_probability,
                                            moves=moves)
            print("")
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            sampler.reset()

            print("")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, main_iter, progress=True)

            return sampler

        elif workers > 1:
            with Pool(processes=workers) as pool:
                sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                self._log_probability,
                                                moves=moves,
                                                pool=pool)
                print("")
                print("Running burn-in...")
                p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
                sampler.reset()

                print("")
                print("Running production...")
                pos, prob, state = sampler.run_mcmc(p0, main_iter,
                                                    progress=True)
                pool.close()
            # outside with-as
            return sampler
