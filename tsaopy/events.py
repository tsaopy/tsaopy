"""tsaopy Events submodule."""
import numpy as np
from tsaopy._f2pyauxmod import simulation


def _base_ll(pred, data, sigma):

    # discard diverging simulations
    if not np.isfinite(pred).all():
        return -np.inf

    return -.5 * np.sum(((pred - data) / sigma) ** 2)


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
                 v_data=None, v_sigma=None, log_likelihood=None,
                 ll_params=None):
        """

        Parameters
        ----------
        params : dict
            dictionary containing the parameters relevant in the event. It's
            necessary to always include either x0 and v0 (initial conditions)
            or tt (transient state time). If transient state is considered and
            initial conditions not, then by default x0 and v0 are set to 0.
            Ideally, use transient state time in driven oscillators.
            Optionally one can use ep for the equilibrium point for a series
            where it's equilibrium point is shifted from 0.

            Each entry in the dictionary should be defined with its key ('tt',
            'x0', 'v0', 'ep') and the value should be the prior for that parame
            ter.
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
        log_likelihood : callable, optional
            Optional parameter for including a custom logarithmic likelihood.
            See docs on how to set it up. The default is None.
        ll_params : dict, optional
            A dictionary with extra parameters used by the custom log likelihoo
            d. Use a label for keys and a prior for the value. The default is
            None.

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

            params_dict = {'x0': x0_prior,
                           'v0': v0_prior
                           }

            event1 = tsaopy.events.Event(params_dict, t, x, x_noise)

        """
        #           TO DO HERE: error handling

        #       Define core attributes ~~~~
        self.ndim = len(params)
        self.tsplit = 2
        self.datalen = len(t_data)
        self.dt = np.float128((t_data[-1] - t_data[0]) / (self.datalen - 1))

        # attibute priors
        self.priors = []

        # do checks in coefs
        # tt
        if 'tt' in params:
            self.using_tt = True
            self.priors.append(params['tt'])
        else:
            self.using_tt = False

        # x0v0
        if ('x0' in params and 'v0' in params):
            self.using_x0v0 = True
            self.priors.append(params['x0'])
            self.priors.append(params['v0'])
        else:
            self.using_x0v0 = False

        # ep
        if 'ep' in params:
            self.using_ep = True
            self.priors.append(params['ep'])
        else:
            self.using_ep = False

        # save observations & sigma data
        self.t0, self.tf = t_data[0], t_data[-1]
        self.x_data = x_data
        self.x_sigma = x_sigma

        if not (v_data is None or v_sigma is None):
            self.fit_to_v = True
            self.v_data = v_data
            self.v_sigma = v_sigma
        else:
            self.fit_to_v = False

        # log likelihood
        if log_likelihood is None:
            self.use_custom_ll = False
        else:
            self.use_custom_ll = True
            self.custom_ll = log_likelihood

        if ll_params is None:
            self.custom_ll_params = False
        else:
            self.custom_ll_params = True
            self.custom_ll_params_labels = [_ for _ in ll_params]
            self.priors += [ll_params[_] for _ in ll_params]
            self.cllp_ndim = len(ll_params)
            self.ndim += self.cllp_ndim

    #  methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _predict(self, A, B, C, df, df_params, tt, x0v0, ep):
        """Compute tsaopy prediction using coefs and iniconds."""
        fit_to_v = self.fit_to_v

        tsplit = self.tsplit
        dt = self.dt / tsplit

        t0, tf = self.t0 - tt, self.tf
        if not self.using_tt and self.using_x0v0:
            datalen = (self.datalen - 1) * tsplit + 1
        elif self.using_tt:
            datalen = int((tf - t0) / dt) + 1
            if datalen < (self.datalen - 1) * tsplit + 1:
                datalen += 1

        na, nb = len(A), len(B)
        cn, cm = C.shape
        nf = 2 * (datalen - 1) + 1
        t_array = np.linspace(t0, tf, nf)
        F = df(t_array, df_params)

        if cn == 0 or cm == 0:
            cn, cm, C = 1, 1, np.zeros(1)

        pred = simulation(a_in=A, b_in=B, c_in=C, f_in=F, x_in=x0v0,
                          na=na, nb=nb, cn=cn, cm=cm, nf=nf,
                          datalen=datalen, dt=dt)[::tsplit][-self.datalen:]
        predx, predv = pred[:, 0], pred[:, 1]

        if not fit_to_v:
            return predx + ep
        elif fit_to_v:
            return predx + ep, predv

    def _default_ll(self, pred):
        """Default log likelihood."""
        if not self.fit_to_v:
            return _base_ll(pred, self.x_data, self.x_sigma)
        elif self.fit_to_v:
            return (_base_ll(pred[0], self.x_data, self.x_sigma)
                    + _base_ll(pred[1], self.v_data, self.v_sigma))

    def _log_likelihood(self, A, B, C, df, df_params,
                        tt, x0v0, ep,
                        ll_params):
        """Compute log likelihood for event parameters and ODE coefs arrays."""
        pred = self._predict(A, B, C, df, df_params, tt, x0v0, ep)

        if not self.use_custom_ll:
            return self._default_ll(pred)
        elif self.use_custom_ll and self.custom_ll_params:
            return self.custom_ll(self, pred, ll_params)
        elif self.use_custom_ll and not self.custom_ll_params:
            return self.custom_ll(self, pred)

    def _log_prior(self, event_coords):
        """Compute log prior for event parameters."""
        result = 1
        for i, p in enumerate(self.priors):
            prob = p(event_coords[i])
            if prob <= .0:
                return -np.inf
            result *= prob
        return np.log(result)
