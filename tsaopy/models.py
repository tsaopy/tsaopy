"""Main submodule of the library."""
import numpy as np


#           Aux stuff, raises etc
class ModelInitException(Exception):
    """Custom exception for Event instance init."""

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

    def log_likelihood(self, coords):
        """Compute log likelihood for all events."""
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
            result += event.log_likelihood(self, A, B, F, C, x0v0,
                                           ep, logfx, logfv)

    def log_prior(self, coords):
        """Compute log prior for all events."""
        odecoefs_ndim = self.odecoefs_ndim
        odecoords = coords[:odecoefs_ndim]
        result = 1
        for i, p in enumerate(self.priors):
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
            result += event.log_prior(event_coords)

        return result
