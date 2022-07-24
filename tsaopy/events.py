"""tsaopy Events submodule."""
import numpy as np
from tsaopy._f2pyauxmod import simulation, simulationv


#           Aux stuff, raises etc
class EventInitException(Exception):
    """Custom exception for Event instance init."""

    def __init__(self, rtext):
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
        """
        Parameters
        ----------
        params : dict
            dictionary containing the parameters relevant in the event. It's
            necessary to always include x0 and v0. Optionally one can use ep
            for the equilibrium point, and log_fx and log_fv for the f term in
            the uncertainty (see docs for more info).

            Each parameter in the dict should be defined with its key ('x0,
            'v0', 'ep', 'log_fx', 'log_fv') and the value should be the prior
            for that parameter.
        t_data : array
            array containing the time values. Values must be evenly spread.
        x_data : array
            array containing the position measurements. Must be of the same
            length as t_data.
        x_sigma : float or array
            uncertainty of the measurements. Can be a float with a unique value
            for all measurements, or an array of the same shape as x_data with
            a unique value for each value of x_data.
        v_data : array, optional
            array containing the velocity measurements. Must be of the same
            length as t_data.
        v_sigma : float or array, optional
            uncertainty of the measurements. Can be a float with a unique value
            for all measurements, or an array of the same shape as v_data with
            a unique value for each value of v_data. Necessary if v_data is
            being used.

        Examples
        --------
            import numpy as np
            import tsaopy
            import quickemcee as qmc

            t = np.linspace(0,10,101)
            x = np.cos(t) + np.random.normal(.0, .3, 101) # data with noise
            x_noise = .3

            x0_prior = qmc.utils.normal_prior(1.0, 10.0)
            v0_prior = qmc.utils.normal_prior(.0, 10.0)

            params_dict = {
                           'x0': x0_prior,
                           'v0': v0_prior
                           }

            event1 = tsaopy.events.Event(params_dict, t, x, x_noise)
        """
        #           TO DO HERE: improve error handling ...
        # make sure that x0 and v0 are in params
        try:
            if 'x0' not in params:
                raise EventInitException('no x0 param when building Event.')
            if 'v0' not in params:
                raise EventInitException('no v0 param when building Event.')
        except Exception as exception:
            raise EventInitException("couldn't verify x0 and v0 keys were incl"
                                     "uded in params dict.") from exception

        # test if priors don't return a positive float when called
        for param in params:
            try:
                x, p = np.random.normal(.0, 100.0), params[param]
                if not np.isfinite(p(x)):
                    raise ValueError("prior for a param return nan or inf when"
                                     " called with a random float.")
                if p(x) < .0:
                    raise ValueError("prior for a param returned a negative va"
                                     "lue when called with a random float.")
            except Exception as exception:
                raise EventInitException("couldn't verify that a random float "
                                         "returns a positive value for some "
                                         "parameter prior.") from exception

        # do check for v data usage
        if v_data is not None:
            if v_sigma is None:
                raise EventInitException("v_data is being used but v_sigma is "
                                         "missing.")
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
            raise EventInitException("couldn't verify that all input arrays "
                                     "have compatible shapes.") from exception

        # check arrays have finite float values
        try:
            if not np.isfinite(x_data).all():
                raise ValueError('x_data has non finite values.')
            if not np.isfinite(t_data).all():
                raise ValueError('t_data has non finite values.')
            if self.using_v_data:
                if not np.isfinite(v_data).all():
                    raise ValueError('v_data has non finite values.')
        except Exception as exception:
            raise EventInitException("couldn't verify that all values in input"
                                     " arrays are finite numbers.") \
                                                            from exception

        #       Define core attributes ~~~~
        self.ndim = len(params)
        self.tsplit = 2
        self.datalen = len(t_data)
        self.dt = (t_data[-1] - t_data[0]) / (self.datalen - 1)

        # attribute params
        self.params = params

        # attibute priors
        self.priors = [params['x0'],
                       params['v0']]

        # do checks for ep
        if 'ep' in params:
            self.using_ep = True
            self.priors.append(params['ep'])
        else:
            self.using_ep = False

        # do checks for fx
        if 'log_fx' in params:
            self.using_logfx = True
            self.priors.append(params['log_fx'])
        else:
            self.using_logfx = False

        # do checks for fv
        if 'log_fv' in params:
            # necessary check
            if not self.using_v_data:
                raise EventInitException("log_fv is in params but there is no "
                                         "v data.")
            self.using_logfv = True
            self.priors.append(params['log_fv'])
        else:
            self.using_logfv = False

        # save observations & sigma data
        self.t_data = t_data
        self.x_data = x_data
        self.x_sigma = x_sigma
        if self.using_v_data:
            self.v_data = v_data
            self.v_sigma = v_sigma

    def _predict(self, A, B, F, C, x0v0, ep):
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
        pred = self._predict(A, B, F, C, x0v0, ep)

        # discard diverging simulations
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
