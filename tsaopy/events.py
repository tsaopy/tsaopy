"""tsaopy Events submodule."""
import numpy as np
from tsaopy._f2pyauxmod import simulation, simulationv


#           Aux stuff, raises etc
class EventInitException(Exception):
    """Custom exception for Event instance init."""

    def __init__(self, rtext):
        """Edit init."""
        msg = rtext + (' Exception ocurred while trying to instance a tsaopy '
                       'Event object.')
        super().__init__(msg)


def _ll(pred, data, sigma):
    return -.5 * np.sum(((pred - data) / sigma) ** 2)


def _ll_logf(pred, data, sigma, logf):
    s2 = sigma ** 2 + pred ** 2 * np.exp(2 * logf)
    return -.5 * np.sum((pred - data) ** 2 / s2 + np.log(s2))


#           tsaopy scripts
class Event:
    """
    tsaopy Event class. Used to later instance tsaopy Model objects.

    The idea behind the Event class is to make it straightforward to fit the
    same equation of motion to different sets of measurements at the same time.

    Suppose we observed the same system several times with different initial
    conditions. Then we will set up an Event instace for each of our experiment
    runs. We will group them all together later when creating a Model class
    instance. With this Model instance we will be fitting all the different
    sets of measurements with the same equation of motion, but different
    initial conditions, equilibrium points, etc.
    """

    def __init__(self, params, t_data, x_data, x_sigma,
                 v_data=None, v_sigma=None):
        """Instance parameters."""
        #           TO DO HERE: improve error handling ...
        # make sure that x0 and v0 are in params
        try:
            if 'x0' not in params:
                raise EventInitException('no x0 param when building Event.')
            if 'v0' not in params:
                raise EventInitException('no v0 param when building Event.')
        except Exception as exception:
            raise EventInitException("couldn't assert x0 and v0 keys were incl"
                                     "uded in params dict.") from exception

        # make sure that all params have 2 values
        for param in params:
            try:
                if len(params[param]) < 2:
                    raise EventInitException('missing value in a param.')
                elif len(params[param]) > 2:
                    raise EventInitException('too many values in a param.')
            except Exception as exception:
                raise EventInitException("couldn't assert all params have a "
                                         "guess and a prior.") from exception

        # check that all guesses return a positive value for the prior
        for param in params:
            try:
                x, p = params[param]
                if not np.isfinite(p(x)):
                    raise ValueError("initial guess for a param returned nan "
                                     "or inf for its prior.")
                if not p(x) > .0:
                    raise ValueError("initial guess for a param didn't return "
                                     "a positive value for its prior.")
            except Exception as exception:
                raise EventInitException("couldn't check that initial guess "
                                         "returns a positive prior value for "
                                         "some parameter.") from exception

        # do check for v data usage
        if v_data is not None:
            if v_sigma is None:
                raise EventInitException("v_data array given but no v_sigma.")
            self.using_v_data = True
        else:
            self.using_v_data = False

        # check x and t have the same lengths, x and v same shape
        try:
            if not len(t_data) == len(x_data):
                raise ValueError('x_data and t_data have different lengths.')
            if self.using_v_data:
                if not x_data.shape == v_data.shape:
                    raise ValueError('x_data and v_data have different shapes.'
                                     )
        except Exception as exception:
            raise EventInitException("couldn't assert all input arrays have "
                                     "compatible shapes.") from exception

        # check x and t have finite float values
        try:
            if not np.isfinite(x_data).all():
                raise ValueError('x_data has non finite values.')
            if not np.isfinite(t_data).all():
                raise ValueError('t_data has non finite values.')
            if self.using_v_data:
                if not np.isfinite(v_data).all():
                    raise ValueError('v_data has non finite values.')
        except Exception as exception:
            raise EventInitException("couldn't assert all values in input "
                                     "arrays are finite numbers.") \
                                                            from exception

        #       Define core attributes ~~~~
        self.ndim = len(params)
        self.tsplit = 2
        self.datalen = len(t_data)
        self.dt = (t_data[-1] - t_data[0]) / (self.datalen - 1)

        # attribute params
        self.params = params

        # attibute priors
        self.priors = [params['x0'][-1],
                       params['v0'][-1]]

        # do checks for ep
        if 'ep' in params:
            self.using_ep = True
            self.priors.append(params['ep'][-1])
        else:
            self.using_ep = False

        # do checks for fx
        if 'log_fx' in params:
            self.using_logfx = True
            self.priors.append(params['log_fx'][-1])
        else:
            self.using_logfx = False

        # do checks for fv
        if 'log_fv' in params:
            # necessary check
            if not self.using_v_data:
                raise EventInitException("log_fv is in params but there is no "
                                         "v data.")
            self.using_logfv = True
            self.priors.append(params['log_fv'][-1])
        else:
            self.using_logfv = False

        # save observations & sigma data
        self.t_data = t_data
        self.x_data = x_data
        self.x_sigma = x_sigma
        if self.using_v_data:
            self.v_data = v_data
            self.v_sigma = v_sigma

    def predict(self, A, B, F, C, x0v0, ep):
        """Compute tsaopy prediction using coefs and iniconds."""
        use_ep, use_v = self.using_ep, self.using_v_data
        tsplit = self.tsplit
        dt, datalen = self.dt / tsplit, self.datalen * tsplit

        na, nb = len(A), len(B)
        cn, cm = C.shape

        if not use_ep and not use_v:
            return simulation(x0v0, A, B, C, F,
                              dt, datalen, na, nb, cn, cm)[::tsplit]
        if use_ep and not use_v:
            return simulation(x0v0, A, B, C, F,
                              dt, datalen, na, nb, cn,
                              cm)[::tsplit] + ep
        if not use_ep and use_v:
            return simulationv(x0v0, A, B, C, F,
                               dt, datalen, na, nb, cn, cm)[::tsplit]
        if use_ep and use_v:
            arr = np.zeros((self.datalen, 2))
            arr[:, 0] += ep
            return simulationv(x0v0, A, B, C, F,
                               dt, datalen, na, nb, cn, cm)[::tsplit] + arr

    def _log_prior(self, event_coords):
        """Compute log prior for event parameters."""
        result = 1
        for i, p in enumerate(self.priors):
            prob = p(event_coords[i])
            if prob <= .0:
                return -np.inf
            result *= prob
        return np.log(result)

    def _log_likelihood(self, A, B, F, C, x0v0,
                        ep, log_fx, log_fv):
        """Compute log likelihood for event parameters and ODE coefs arrays."""
        pred = self.predict(A, B, F, C, x0v0, ep)

        if not np.isfinite(pred).all():
            return -np.inf

        use_v, use_fx, use_fv = (self.using_v_data, self.using_logfx,
                                 self.using_logfv)

        # not using v data cases
        if not use_v and not use_fx:
            return _ll(pred, self.x_data, self.x_sigma)
        elif not use_v and use_fx:
            return _ll_logf(pred, self.x_data, self.x_sigma, log_fx)

        predx, predv = pred[:, 0], pred[:, 1]
        # using v data cases
        if use_v and not use_fx and not use_fv:
            return (_ll(predx, self.x_data, self.x_sigma)
                    + _ll(predv, self.v_data, self.v_sigma))
        elif use_v and use_fx and not use_fv:
            return (_ll_logf(predx, self.x_data, self.x_sigma, log_fx)
                    + _ll(predv, self.v_data, self.v_sigma))
        elif use_v and not use_fx and use_fv:
            return (_ll(predx, self.x_data, self.x_sigma)
                    + _ll_logf(predv, self.v_data, self.v_sigma, log_fv))
        elif use_v and use_fx and use_fv:
            return (_ll_logf(predx, self.x_data, self.x_sigma, log_fx)
                    + _ll_logf(predv, self.v_data, self.v_sigma, log_fv))
